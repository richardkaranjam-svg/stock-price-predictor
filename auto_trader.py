"""
auto_trader.py — V3+ Institutional-Grade Paper Trading Engine.

Resolves Issues: #4, #5, #8, #15, #18
  - 15:45 ET auto-flatten & 15:30 ET entry cutoff
  - High-signal windows (09:30–11:00, 14:30–15:30) & -2.0% daily kill-switch
  - EV gate: P_win >= 0.58 AND ev_atr_value > 0
  - Short selling guard via broker.check_short_eligibility()
  - Dynamic ATR stops (1.50x TP, 0.75x SL) & 1R breakeven ratchet
  - Version/checksum verification against env_versions in artifact
  - SQLite V3+ schema auto-migration on startup
"""
import json
import logging
import sqlite3
import threading
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from broker import (
    get_alpaca_client, get_paper_account, submit_paper_market_order,
    cancel_open_orders, cancel_open_orders_for_symbol, get_available_qty,
    check_short_eligibility, get_short_safe_buying_power,
)
from config import is_market_open, is_high_signal_window
from data_loader import create_features, download_intraday_data, download_spy_intraday, get_news_summary
from predict import get_latest_stock_data, predict_v3
from db_migration import migrate_paper_trades_schema
import quant_core

DB_PATH = Path(__file__).resolve().parent / 'paper_trades.db'
MODEL_PATH = Path(__file__).resolve().parent / 'stock_price_model.pkl'
METADATA_PATH = Path(__file__).resolve().parent / 'model_metadata.json'
DAILY_MAX_LOSS_PCT = 0.02

# V3+ Trading Parameters
MIN_CONFIDENCE = 0.58
TP_ATR_MULT = 1.50
SL_ATR_MULT = 0.75
RATCHET_TRIGGER_R = 1.0  # Ratchet triggers at +1R (= +SL_ATR_MULT)
RATCHET_NEW_SL_ATR = 0.10  # New SL after ratchet: entry +/- 0.10 × ATR

logger = logging.getLogger(__name__)


class PaperTrader:

    def __init__(self):
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.ticker = 'AAPL'
        self.interval_seconds = 1200
        self.threshold = 0.005
        self.quantity = 1
        self.slippage_pct = 0.0015
        # V3+ dynamic state
        self._entry_price: Optional[float] = None
        self._entry_atr: Optional[float] = None
        self._entry_side: Optional[str] = None  # 'long' or 'short'
        self._tp_price: Optional[float] = None
        self._sl_price: Optional[float] = None
        self._ratchet_triggered: bool = False
        self._init_db()
        self._check_env_versions()

    def _connect(self):
        return sqlite3.connect(DB_PATH)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                ticker TEXT NOT NULL,
                action TEXT NOT NULL,
                predicted_return REAL,
                current_price REAL,
                order_id TEXT,
                order_status TEXT,
                error TEXT
            )''')
            # V2 columns
            columns = {row[1] for row in conn.execute('PRAGMA table_info(paper_trades)')}
            if 'reason' not in columns:
                conn.execute('ALTER TABLE paper_trades ADD COLUMN reason TEXT')
            if 'fill_price' not in columns:
                conn.execute('ALTER TABLE paper_trades ADD COLUMN fill_price REAL')
            if 'slippage' not in columns:
                conn.execute('ALTER TABLE paper_trades ADD COLUMN slippage REAL')
            # V3+ schema migration
            migrate_paper_trades_schema(conn)

            conn.execute('''CREATE TABLE IF NOT EXISTS account_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                equity REAL NOT NULL,
                starting_equity REAL NOT NULL,
                profit_loss REAL NOT NULL
            )''')

    def _check_env_versions(self):
        """Compare local library versions against artifact's env_versions (Tweak 5)."""
        if not MODEL_PATH.exists():
            return
        try:
            raw = joblib.load(MODEL_PATH)
            if not isinstance(raw, dict):
                return
            train_versions = raw.get('env_versions', {})
            if not train_versions:
                return

            import sklearn
            import lightgbm as lgb
            local_versions = {
                'numpy': np.__version__,
                'scikit-learn': sklearn.__version__,
                'lightgbm': lgb.__version__,
            }
            for lib, train_ver in train_versions.items():
                local_ver = local_versions.get(lib)
                if local_ver and local_ver != train_ver:
                    logger.warning(
                        f"⚠ Version mismatch for {lib}: trained={train_ver}, local={local_ver}"
                    )
        except Exception as e:
            logger.warning(f"Could not verify env_versions: {e}")

    def _record(self, ticker, action, predicted_return=None, current_price=None,
                order_id=None, order_status=None, error=None, reason=None,
                fill_price=None, slippage=None,
                atr_at_entry=None, ev_atr_value=None,
                take_profit_price=None, stop_loss_price=None,
                ratchet_triggered=0, confidence=None, model_version='V3+'):
        with self._connect() as conn:
            conn.execute(
                '''INSERT INTO paper_trades
                (created_at, ticker, action, predicted_return, current_price,
                 order_id, order_status, error, reason, fill_price, slippage,
                 atr_at_entry, ev_atr_value, take_profit_price, stop_loss_price,
                 ratchet_triggered, confidence, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (datetime.now(timezone.utc).isoformat(), ticker, action,
                 predicted_return, current_price, order_id, order_status,
                 error, reason, fill_price, slippage,
                 atr_at_entry, ev_atr_value, take_profit_price, stop_loss_price,
                 ratchet_triggered, confidence, model_version),
            )

    def _record_account_snapshot(self):
        account = get_paper_account()
        equity = float(account['equity'])
        with self._connect() as conn:
            row = conn.execute(
                'SELECT starting_equity FROM account_snapshots ORDER BY id LIMIT 1'
            ).fetchone()
            starting_equity = float(row[0]) if row else equity
            profit_loss = equity - starting_equity
            conn.execute(
                '''INSERT INTO account_snapshots
                (created_at, equity, starting_equity, profit_loss)
                VALUES (?, ?, ?, ?)''',
                (datetime.now(timezone.utc).isoformat(), equity, starting_equity, profit_loss),
            )
        return {'equity': equity, 'starting_equity': starting_equity, 'profit_loss': profit_loss}

    def _get_position(self, ticker):
        """Return current position qty and average entry price."""
        client = get_alpaca_client()
        try:
            pos = client.get_open_position(ticker)
            return {
                'qty': float(pos.qty),
                'avg_entry_price': float(pos.avg_entry_price),
                'qty_available': float(pos.qty_available),
            }
        except Exception:
            return {'qty': 0.0, 'avg_entry_price': 0.0, 'qty_available': 0.0}

    def _position_quantity(self, ticker):
        """Backward-compatible helper for legacy trading logic/tests."""
        return self._get_position(ticker)['qty']

    def _get_fill_price(self, order_id, max_wait=5):
        """Poll Alpaca for fill price after order submission."""
        client = get_alpaca_client()
        for _ in range(max_wait):
            try:
                order = client.get_order_by_id(order_id)
                if order.filled_avg_price is not None:
                    return float(order.filled_avg_price)
                if str(getattr(order.status, 'value', order.status)) in ('filled', 'canceled', 'expired'):
                    return None
            except Exception:
                pass
            time.sleep(1)
        return None

    def _check_daily_kill_switch(self) -> bool:
        """Halt trading for the day if daily loss reaches -2.0%."""
        try:
            account = get_paper_account()
            equity = float(account['equity'])
            today_utc = datetime.now(timezone.utc).date().isoformat()
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT equity FROM account_snapshots WHERE substr(created_at, 1, 10) = ? ORDER BY id ASC LIMIT 1",
                    (today_utc,)
                ).fetchone()
            if row is None:
                return False
            start_equity = float(row[0])
            if start_equity <= 0:
                return False
            daily_pnl_pct = (equity - start_equity) / start_equity
            return daily_pnl_pct <= -DAILY_MAX_LOSS_PCT
        except Exception:
            return False

    def _check_eod_flatten(self, ticker) -> Optional[dict]:
        """15:45 ET auto-flatten: force-close any open position.

        Also blocks new entries after 15:30 ET.
        """
        now_et = quant_core.get_et_now()
        mins_since_open = (now_et.hour - 9) * 60 + (now_et.minute - 30)

        # 15:45 ET auto-flatten (375 mins since open)
        if mins_since_open >= 375:
            position = self._get_position(ticker)
            if position['qty'] != 0:
                try:
                    cancelled = cancel_open_orders_for_symbol(ticker)
                    side = 'sell' if position['qty'] > 0 else 'buy'
                    qty = abs(position['qty'])
                    order = submit_paper_market_order(ticker, side, qty)
                    self._entry_price = None
                    self._entry_atr = None
                    self._entry_side = None
                    self._ratchet_triggered = False
                    self._record(
                        ticker, f'eod_flatten_{side}',
                        current_price=float(position['avg_entry_price']),
                        order_id=order['id'], order_status=order['status'],
                        reason='EOD_1545_AUTO_FLATTEN',
                    )
                    return {
                        'action': f'eod_flatten_{side}',
                        'reason': 'EOD_1545_AUTO_FLATTEN',
                        'order': order, 'cancelled_orders': cancelled,
                        'performance': self._record_account_snapshot(),
                    }
                except Exception as e:
                    self._record(ticker, 'eod_flatten_error', error=str(e),
                                 reason='EOD_1545_AUTO_FLATTEN')

        return None

    def run_once(self):
        ticker = self.ticker

        # 1. Market-hours guard
        if not is_market_open():
            self._record(ticker, 'market_closed', reason='US market is closed (09:30–16:00 ET Mon–Fri)')
            return {'action': 'market_closed', 'reason': 'market is closed'}

        # 2. 15:45 ET auto-flatten (must happen BEFORE any other logic)
        eod_result = self._check_eod_flatten(ticker)
        if eod_result is not None:
            return eod_result

        # 3. 15:30 ET entry cutoff
        now_et = quant_core.get_et_now()
        mins_since_open = (now_et.hour - 9) * 60 + (now_et.minute - 30)
        entry_blocked = mins_since_open > 360  # > 15:30 ET

        # 4. Daily kill-switch
        if self._check_daily_kill_switch():
            self._record(ticker, 'daily_kill_switch', reason=f'daily loss limit -{DAILY_MAX_LOSS_PCT:.0%} reached')
            return {'action': 'daily_kill_switch', 'reason': 'daily kill switch activated'}

        # 5. High-signal session window filter
        if not is_high_signal_window():
            self._record(ticker, 'hold', reason='outside high-signal session (09:30-11:00 & 14:30-15:30 ET)')
            return {'action': 'hold', 'reason': 'midday filter active'}

        # 6. Get V3+ prediction
        try:
            pred = predict_v3(ticker)
        except Exception as e:
            self._record(ticker, 'error', error=str(e), reason='prediction_failed')
            return {'action': 'error', 'reason': str(e)}

        predicted_class = pred['predicted_class']
        confidence = pred['confidence']
        current_price = pred['current_price']
        atr_at_entry = pred['atr_at_entry']
        ev_atr_value = pred['ev_atr_value']
        tp_price = pred['take_profit_price']
        sl_price = pred['stop_loss_price']

        position = self._get_position(ticker)
        position_qty = position['qty']
        avg_entry_price = position['avg_entry_price']
        qty_available = position['qty_available']

        # ══════════════════════════════════════════════════════════
        # PRIORITY 1: Dynamic ATR Stops & 1R Breakeven Ratchet
        # ══════════════════════════════════════════════════════════
        if position_qty != 0 and self._entry_price is not None and self._entry_atr is not None:
            if self._entry_side == 'long':
                unrealised_atr = (current_price - self._entry_price) / self._entry_atr if self._entry_atr > 0 else 0
            else:  # short
                unrealised_atr = (self._entry_price - current_price) / self._entry_atr if self._entry_atr > 0 else 0

            # --- 1R Ratchet: tighten stop to near breakeven ---
            if not self._ratchet_triggered and unrealised_atr >= RATCHET_TRIGGER_R * SL_ATR_MULT:
                self._ratchet_triggered = True
                if self._entry_side == 'long':
                    self._sl_price = self._entry_price + RATCHET_NEW_SL_ATR * self._entry_atr
                else:
                    self._sl_price = self._entry_price - RATCHET_NEW_SL_ATR * self._entry_atr
                self._record(
                    ticker, 'ratchet', current_price=current_price,
                    reason='RATCHET_1R_BREAKEVEN',
                    atr_at_entry=self._entry_atr, ratchet_triggered=1,
                    take_profit_price=self._tp_price, stop_loss_price=self._sl_price,
                )

            # --- Check TP ---
            tp_hit = False
            sl_hit = False
            if self._entry_side == 'long':
                tp_hit = current_price >= self._tp_price
                sl_hit = current_price <= self._sl_price
            else:
                tp_hit = current_price <= self._tp_price
                sl_hit = current_price >= self._sl_price

            if tp_hit:
                side = 'sell' if position_qty > 0 else 'buy'
                qty = abs(position_qty)
                try:
                    cancelled = cancel_open_orders_for_symbol(ticker)
                    order = submit_paper_market_order(ticker, side, qty)
                    fill_price = self._get_fill_price(order['id'])
                    self._record(
                        ticker, side, current_price=current_price,
                        order_id=order['id'], order_status=order['status'],
                        reason='DYNAMIC_ATR_TAKE_PROFIT',
                        fill_price=fill_price, atr_at_entry=self._entry_atr,
                        take_profit_price=self._tp_price, stop_loss_price=self._sl_price,
                        ratchet_triggered=int(self._ratchet_triggered), confidence=confidence,
                    )
                    self._entry_price = None
                    self._entry_atr = None
                    self._entry_side = None
                    self._ratchet_triggered = False
                    return {'action': side, 'reason': 'DYNAMIC_ATR_TAKE_PROFIT',
                            'order': order, 'performance': self._record_account_snapshot()}
                except Exception as e:
                    self._record(ticker, side, error=str(e), reason='DYNAMIC_ATR_TAKE_PROFIT')
                    raise

            if sl_hit:
                side = 'sell' if position_qty > 0 else 'buy'
                qty = abs(position_qty)
                try:
                    cancelled = cancel_open_orders_for_symbol(ticker)
                    order = submit_paper_market_order(ticker, side, qty)
                    fill_price = self._get_fill_price(order['id'])
                    self._record(
                        ticker, side, current_price=current_price,
                        order_id=order['id'], order_status=order['status'],
                        reason='DYNAMIC_ATR_STOP_LOSS',
                        fill_price=fill_price, atr_at_entry=self._entry_atr,
                        take_profit_price=self._tp_price, stop_loss_price=self._sl_price,
                        ratchet_triggered=int(self._ratchet_triggered), confidence=confidence,
                    )
                    self._entry_price = None
                    self._entry_atr = None
                    self._entry_side = None
                    self._ratchet_triggered = False
                    return {'action': side, 'reason': 'DYNAMIC_ATR_STOP_LOSS',
                            'order': order, 'performance': self._record_account_snapshot()}
                except Exception as e:
                    self._record(ticker, side, error=str(e), reason='DYNAMIC_ATR_STOP_LOSS')
                    raise

        # ══════════════════════════════════════════════════════════
        # PRIORITY 2: ML Model Decision with V3+ Gates
        # ══════════════════════════════════════════════════════════

        # Entry cutoff after 15:30 ET
        if entry_blocked:
            self._record(ticker, 'hold', current_price=current_price,
                         reason='ENTRY_BLOCKED_AFTER_1530', confidence=confidence)
            return {'action': 'hold', 'reason': 'entry blocked after 15:30 ET'}

        # Confidence gate
        if confidence < MIN_CONFIDENCE:
            self._record(ticker, 'hold', current_price=current_price,
                         reason=f'LOW_CONFIDENCE_{confidence:.0%}', confidence=confidence)
            return {'action': 'hold', 'reason': f'confidence {confidence:.0%} < {MIN_CONFIDENCE:.0%}'}

        # EV gate
        if predicted_class != 0 and ev_atr_value <= 0:
            self._record(ticker, 'hold', current_price=current_price,
                         reason='EV_GATE_REJECTED', confidence=confidence,
                         ev_atr_value=ev_atr_value, atr_at_entry=atr_at_entry)
            return {'action': 'hold', 'reason': f'EV_GATE_REJECTED (EV={ev_atr_value:.4f})'}

        # --- Determine action ---
        action = 'hold'
        reason = ''

        if predicted_class == 1:
            if position_qty < 0:
                # Close short first, then go long
                action = 'buy'
                reason = f'close short + go long (conf={confidence:.0%}, EV={ev_atr_value:.4f})'
            elif position_qty == 0:
                action = 'buy'
                reason = f'go long (conf={confidence:.0%}, EV={ev_atr_value:.4f})'
            else:
                action = 'hold'
                reason = f'already long (conf={confidence:.0%})'

        elif predicted_class == -1:
            # Short selling guard (Gap 3, Tweak 4)
            if position_qty > 0:
                # Close long first; only reverse into short if eligible
                short_check = check_short_eligibility(ticker)
                safe_bp = get_short_safe_buying_power(0.80)
                order_value = current_price * self.quantity
                can_short = (
                    short_check['allowed']
                    and order_value <= safe_bp
                )
                if can_short:
                    action = 'sell'
                    reason = f'close long + go short (conf={confidence:.0%}, EV={ev_atr_value:.4f})'
                else:
                    # Close the long but do NOT open a new short
                    action = 'sell'
                    reason = f'close long only (short guard blocked: {short_check.get("reason", "BP")}), conf={confidence:.0%}'
            elif position_qty == 0:
                # Attempt to short
                short_check = check_short_eligibility(ticker)
                if not short_check['allowed']:
                    self._record(
                        ticker, 'hold', current_price=current_price,
                        reason='SHORT_GUARD_BLOCKED', confidence=confidence,
                        ev_atr_value=ev_atr_value, atr_at_entry=atr_at_entry,
                    )
                    return {
                        'action': 'hold',
                        'reason': f'SHORT_GUARD_BLOCKED: {short_check["reason"]}',
                    }

                # Check buying power
                safe_bp = get_short_safe_buying_power(0.80)
                order_value = current_price * self.quantity
                if order_value > safe_bp:
                    self._record(
                        ticker, 'hold', current_price=current_price,
                        reason='SHORT_GUARD_BLOCKED', confidence=confidence,
                    )
                    return {
                        'action': 'hold',
                        'reason': f'SHORT_GUARD_BLOCKED: order ${order_value:.0f} > safe BP ${safe_bp:.0f}',
                    }

                action = 'sell'
                reason = f'go short (conf={confidence:.0%}, EV={ev_atr_value:.4f})'
            else:
                action = 'hold'
                reason = f'already short (conf={confidence:.0%})'

        else:
            action = 'hold'
            reason = f'model predicted CHOP (conf={confidence:.0%})'

        if action == 'hold':
            self._record(ticker, action, current_price=current_price,
                         reason=reason, confidence=confidence,
                         ev_atr_value=ev_atr_value, atr_at_entry=atr_at_entry)
            return {'action': action, 'reason': reason,
                    'performance': self._record_account_snapshot()}

        # ══════════════════════════════════════════════════════════
        # EXECUTE ORDER
        # ══════════════════════════════════════════════════════════
        try:
            # If we're reversing a position, close existing first
            if position_qty != 0:
                close_side = 'sell' if position_qty > 0 else 'buy'
                close_qty = abs(position_qty)
                cancel_open_orders_for_symbol(ticker)
                close_order = submit_paper_market_order(ticker, close_side, close_qty)
                self._record(
                    ticker, close_side, current_price=current_price,
                    order_id=close_order['id'], order_status=close_order['status'],
                    reason=f'close_before_{"long" if action == "buy" else "short"}',
                    confidence=confidence,
                )
                # If short guard blocked the reversal, stop after closing
                if 'close long only' in reason or 'close short only' in reason:
                    self._entry_price = None
                    self._entry_atr = None
                    self._entry_side = None
                    self._ratchet_triggered = False
                    return {
                        'action': close_side, 'reason': reason,
                        'order': close_order,
                        'performance': self._record_account_snapshot(),
                    }
                time.sleep(0.5)  # Brief pause for fill

            # Place new entry order
            cancelled = cancel_open_orders_for_symbol(ticker)
            order = submit_paper_market_order(ticker, action, self.quantity)
            fill_price = self._get_fill_price(order['id'])
            actual_slippage = None
            if fill_price is not None:
                actual_slippage = abs(fill_price - current_price) / current_price

            # Set V3+ dynamic state
            self._entry_price = fill_price if fill_price else current_price
            self._entry_atr = atr_at_entry
            self._entry_side = 'long' if action == 'buy' else 'short'
            self._ratchet_triggered = False

            if self._entry_side == 'long':
                self._tp_price = self._entry_price + TP_ATR_MULT * atr_at_entry
                self._sl_price = self._entry_price - SL_ATR_MULT * atr_at_entry
            else:
                self._tp_price = self._entry_price - TP_ATR_MULT * atr_at_entry
                self._sl_price = self._entry_price + SL_ATR_MULT * atr_at_entry

            self._record(
                ticker, action, current_price=current_price,
                order_id=order['id'], order_status=order['status'],
                reason=reason, fill_price=fill_price, slippage=actual_slippage,
                confidence=confidence, ev_atr_value=ev_atr_value,
                atr_at_entry=atr_at_entry,
                take_profit_price=self._tp_price, stop_loss_price=self._sl_price,
            )
            return {
                'action': action, 'reason': reason,
                'order': order, 'cancelled_orders': cancelled,
                'fill_price': fill_price, 'slippage': actual_slippage,
                'performance': self._record_account_snapshot(),
            }
        except Exception as e:
            self._record(ticker, action, current_price=current_price,
                         error=str(e), reason=reason, confidence=confidence)
            raise

    def _loop(self):
        while not self.stop_event.is_set():
            cycle_started = time.monotonic()
            try:
                self.run_once()
            except Exception as error:
                self._record(self.ticker, 'error', error=str(error))
            remaining = self.interval_seconds - (time.monotonic() - cycle_started)
            self.stop_event.wait(max(0, remaining))

    def start(self, ticker='AAPL', interval_seconds=1200, threshold=0.005, quantity=1, slippage_pct=0.0015):
        with self.lock:
            if self.thread and self.thread.is_alive():
                return False
            self.ticker = ticker.upper()
            self.interval_seconds = max(1200, int(interval_seconds))
            self.threshold = max(0.0, float(threshold))
            self.quantity = max(1, int(quantity))
            self.slippage_pct = max(0.0, float(slippage_pct))
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
            return True

    def stop(self):
        self.stop_event.set()
        return True

    def status(self):
        running = bool(self.thread and self.thread.is_alive() and not self.stop_event.is_set())
        with self._connect() as conn:
            row = conn.execute(
                '''SELECT equity, starting_equity, profit_loss, created_at
                   FROM account_snapshots ORDER BY id DESC LIMIT 1'''
            ).fetchone()
            conn.row_factory = sqlite3.Row
            latest_row = conn.execute(
                'SELECT * FROM paper_trades ORDER BY id DESC LIMIT 1'
            ).fetchone()
        performance = None
        if row:
            performance = {
                'equity': row[0], 'starting_equity': row[1],
                'profit_loss': row[2], 'updated_at': row[3],
            }
        latest = dict(latest_row) if latest_row else None
        return {'running': running, 'ticker': self.ticker, 'threshold': self.threshold,
                'slippage_pct': self.slippage_pct,
                'interval_seconds': self.interval_seconds, 'quantity': self.quantity,
                'performance': performance, 'latest_decision': latest}

    def history(self, limit=50):
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                'SELECT * FROM paper_trades ORDER BY id DESC LIMIT ?', (limit,)
            ).fetchall()
        return [dict(row) for row in rows]


paper_trader = PaperTrader()
