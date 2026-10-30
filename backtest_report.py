"""
backtest_report.py — Standalone generic backtesting & performance reporting engine.

Measures whether trading signals are profitable AFTER realistic transaction costs
(commission + slippage). Generic across any ticker and market asset.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytz

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════
# VERDICT THRESHOLD CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

FAIL_MIN_TRADES = 100
FAIL_MIN_PROFIT_FACTOR = 1.0
FAIL_MIN_EXPECTANCY = 0.0  # expectancy <= 0 is FAIL
WEAK_MIN_PROFIT_FACTOR = 1.2
WEAK_MAX_DRAWDOWN_PCT = 20.0


# ═══════════════════════════════════════════════════════════════════════════
# JSON SERIALIZATION HELPER
# ═══════════════════════════════════════════════════════════════════════════

def _json_default(obj: Any) -> Any:
    """Serialize numpy / pandas objects and infinities to standard JSON."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isinf(obj):
            return 999999.0 if obj > 0 else -999999.0
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    return str(obj)


# ═══════════════════════════════════════════════════════════════════════════
# 1. RUN BACKTEST
# ═══════════════════════════════════════════════════════════════════════════

def run_backtest(
    signals: pd.DataFrame,
    bars: Dict[str, pd.DataFrame],
    cost_per_share: float = 0.01,
    slippage_bps: float = 2.0,
    risk_per_trade: float = 0.005,
    horizon_bars: int = 12,
    tp_atr_mult: float = 1.50,
    sl_atr_mult: float = 0.75,
    initial_capital: float = 100_000.0,
    allow_overlap: bool = False,
    max_concurrent_positions: int = 5,
    max_leverage: float = 1.0,
) -> Dict[str, Any]:
    """Run event-driven backtest on signals across arbitrary tickers.

    Parameters
    ----------
    signals : pd.DataFrame
        Signals with columns:
          - 'timestamp': tz-aware UTC timestamp
          - 'ticker': ticker symbol (str)
          - 'side': in {'long', 'short', 'flat'}
          - 'confidence': optional float
    bars : dict[str, pd.DataFrame]
        OHLCV DataFrames indexed by tz-aware timestamp.
        Optionally contains an 'ATR' column for triple-barrier evaluation.
    cost_per_share : float
        Commission charged per share on entry AND exit (e.g. 0.01 = $0.01/share).
    slippage_bps : float
        Slippage in basis points applied to entry AND exit prices (2.0 = 0.02%).
    risk_per_trade : float
        Fraction of initial capital at risk per trade (e.g. 0.005 = 0.5%).
    horizon_bars : int
        Maximum holding period in bars.
    tp_atr_mult : float
        Take-profit multiplier if ATR is present.
    sl_atr_mult : float
        Stop-loss multiplier if ATR is present.
    initial_capital : float
        Account starting capital in dollars.
    allow_overlap : bool
        If False (default), only one open position per ticker at a time.
        If True, allow multiple concurrent positions for the same ticker.
    max_concurrent_positions : int
        Max simultaneous open positions across all tickers (default 5).
        New signals are skipped when this cap is reached.
    max_leverage : float
        Maximum portfolio leverage (default 1.0 = no margin borrowing).
        Limits shares to floor(max_leverage * equity / entry_price).

    Returns
    -------
    dict with:
      - 'metrics': output of compute_metrics(trades)
      - 'trades': pd.DataFrame of all trade executions
      - 'verdict': 'PASS' | 'WEAK' | 'FAIL'
      - 'reasons': list of failing/weak reason descriptions
      - unpacked metric keys for direct top-level access
    """
    # Standardize bars
    std_bars: Dict[str, pd.DataFrame] = {}
    for ticker_raw, b_df in bars.items():
        if b_df is None or b_df.empty:
            continue
        t_key = str(ticker_raw).strip().upper()
        df = b_df.copy()

        # Ensure tz-aware UTC index
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()

        # Standardize column names to Title Case if lowercase
        col_map = {}
        for c in df.columns:
            c_lower = str(c).lower()
            if c_lower in {'open', 'high', 'low', 'close', 'volume', 'vwap'}:
                col_map[c] = c_lower.capitalize() if c_lower != 'vwap' else 'VWAP'
            elif c_lower == 'atr':
                col_map[c] = 'ATR'
        if col_map:
            df = df.rename(columns=col_map)

        std_bars[t_key] = df

    # Prepare trades DataFrame schema
    trade_columns = [
        'ticker', 'signal_time', 'entry_time', 'exit_time', 'side',
        'shares', 'entry_price', 'fill_entry_price', 'exit_price', 'fill_exit_price',
        'gross_pnl', 'net_pnl', 'costs', 'cost_per_share', 'slippage_bps',
        'return_pct', 'r_multiple', 'exit_reason', 'bars_held', 'confidence',
        'size_capped',
    ]

    # Guard: empty signals or empty bars
    if signals is None or len(signals) == 0 or len(std_bars) == 0:
        empty_trades = pd.DataFrame(columns=trade_columns)
        metrics = compute_metrics(empty_trades, initial_capital=initial_capital)
        v, reasons = verdict(metrics)
        return {
            'metrics': metrics,
            'trades': empty_trades,
            'verdict': v,
            'reasons': reasons,
            **metrics,
        }

    # Standardize signals
    sig_df = signals.copy()
    if 'timestamp' not in sig_df.columns:
        raise ValueError("signals DataFrame must contain a 'timestamp' column.")
    if 'ticker' not in sig_df.columns:
        raise ValueError("signals DataFrame must contain a 'ticker' column.")
    if 'side' not in sig_df.columns:
        raise ValueError("signals DataFrame must contain a 'side' column.")

    sig_df['timestamp'] = pd.to_datetime(sig_df['timestamp'], utc=True)
    sig_df['ticker'] = sig_df['ticker'].astype(str).str.strip().str.upper()
    sig_df['side'] = sig_df['side'].astype(str).str.strip().str.lower()
    sig_df = sig_df.sort_values('timestamp').reset_index(drop=True)

    trades_list: List[Dict[str, Any]] = []
    slippage_rate = float(slippage_bps) / 10000.0

    for _, row in sig_df.iterrows():
        side = row['side']
        if side not in {'long', 'short'}:
            continue  # 'flat' signals produce no trades

        ticker = row['ticker']
        if ticker not in std_bars:
            continue

        b_df = std_bars[ticker]
        sig_time = row['timestamp']
        conf = float(row['confidence']) if 'confidence' in row and pd.notna(row['confidence']) else None

        # Enter at the NEXT bar's open after the signal (no look-ahead)
        next_idx = int(b_df.index.searchsorted(sig_time, side='right'))
        if next_idx >= len(b_df):
            continue  # Signal occurred at or after the last available bar

        entry_time = b_df.index[next_idx]

        # -------------------------------------------------------------------
        # Same-session entry only:
        # Skip signal if entry bar (next_idx) is on a different NY date than signal
        # bar (next_idx - 1), or if signal bar is after 15:30 ET (mins > 360).
        # -------------------------------------------------------------------
        sig_bar_idx = next_idx - 1
        if sig_bar_idx < 0:
            continue

        sig_bar_time = b_df.index[sig_bar_idx]
        sig_dt_et = sig_bar_time.tz_convert('America/New_York')
        entry_dt_et = entry_time.tz_convert('America/New_York')

        if entry_dt_et.date() != sig_dt_et.date():
            continue  # Signal on last bar of day (or overnight gap) -> no trade

        sig_mins_since_open = (sig_dt_et.hour - 9) * 60 + (sig_dt_et.minute - 30)
        if sig_mins_since_open > 360:
            continue  # Signal bar after 15:30 ET cutoff -> no trade

        # Check concurrency / overlap against currently open positions
        # A prior trade is open at entry_time if entry_time <= prior_trade['exit_time']
        active_trades = [
            tr for tr in trades_list
            if tr['entry_time'] <= entry_time <= tr['exit_time']
        ]

        if not allow_overlap:
            # Only one open position per ticker
            if any(tr['ticker'] == ticker for tr in active_trades):
                continue

        if max_concurrent_positions is not None and len(active_trades) >= max_concurrent_positions:
            # Skip signal if global concurrency cap is reached
            continue

        entry_price = float(b_df.iloc[next_idx]['Open'])
        if entry_price <= 0 or np.isnan(entry_price):
            continue

        # -------------------------------------------------------------------
        # Sizing and barriers: ATR of the SIGNAL bar (next_idx - 1), never entry bar
        # -------------------------------------------------------------------
        has_atr = False
        atr_val = 0.0

        if 'ATR' in b_df.columns:
            raw_atr = b_df.iloc[sig_bar_idx]['ATR']
            if pd.notna(raw_atr) and float(raw_atr) > 0:
                atr_val = float(raw_atr)
                has_atr = True
            elif sig_bar_idx > 0:
                prev_atr = b_df.iloc[sig_bar_idx - 1]['ATR']
                if pd.notna(prev_atr) and float(prev_atr) > 0:
                    atr_val = float(prev_atr)
                    has_atr = True

        # Compute dynamic equity from closed trades up to entry_time
        closed_pnl = sum(tr['net_pnl'] for tr in trades_list if tr['exit_time'] < entry_time)
        current_equity = max(0.0, initial_capital + closed_pnl)

        # -------------------------------------------------------------------
        # Sizing and barrier distances (with consistent 1% No-ATR fallback)
        # -------------------------------------------------------------------
        if has_atr:
            stop_distance = sl_atr_mult * atr_val
            tp_distance = tp_atr_mult * atr_val
        else:
            # Enforce the 1% stop that sizing assumes; set tp at (tp_mult/sl_mult)*1%
            stop_pct = 0.01
            stop_distance = entry_price * stop_pct
            tp_distance = (tp_atr_mult / sl_atr_mult) * stop_distance

        stop_distance = max(stop_distance, 0.001)
        dollar_risk = current_equity * risk_per_trade
        risk_shares = max(1, int(dollar_risk / stop_distance))

        # -------------------------------------------------------------------
        # Portfolio-level leverage cap across all open positions
        # -------------------------------------------------------------------
        open_notional = sum(tr['shares'] * tr['entry_price'] for tr in active_trades)
        allowed_notional = max_leverage * current_equity - open_notional

        if allowed_notional <= 0:
            continue  # Skip signal, no portfolio leverage capacity remaining

        max_allowed_shares = int(np.floor(allowed_notional / entry_price))
        shares = min(risk_shares, max_allowed_shares)

        if shares < 1:
            continue  # Skip signal (do not force 1 share)

        size_capped = bool(risk_shares > shares)

        # Costs on entry
        if side == 'long':
            fill_entry_price = entry_price * (1.0 + slippage_rate)
            tp_price = entry_price + tp_distance
            sl_price = entry_price - stop_distance
        else:
            fill_entry_price = entry_price * (1.0 - slippage_rate)
            tp_price = entry_price - tp_distance
            sl_price = entry_price + stop_distance

        # -------------------------------------------------------------------
        # Session Cap: never hold overnight. Exit earlier of horizon_bars or
        # the last bar at/before 15:40 America/New_York on entry day.
        # -------------------------------------------------------------------
        entry_time_et = entry_time.tz_convert('America/New_York')
        entry_date_et = entry_time_et.date()

        end_limit_idx = min(next_idx + horizon_bars, len(b_df))
        valid_bar_indices: List[int] = []

        for j in range(next_idx, end_limit_idx):
            bar_et = b_df.index[j].tz_convert('America/New_York')
            if bar_et.date() != entry_date_et:
                break  # Do not cross overnight boundary
            mins_since_open = (bar_et.hour - 9) * 60 + (bar_et.minute - 30)
            if mins_since_open > 370:  # 15:40 ET cutoff
                break
            valid_bar_indices.append(j)

        if not valid_bar_indices:
            valid_bar_indices = [next_idx]

        # Simulate trajectory through forward valid bars
        exit_price: Optional[float] = None
        exit_time: Optional[pd.Timestamp] = None
        exit_reason: Optional[str] = None
        bars_held: int = 0

        for j in valid_bar_indices:
            bar = b_df.iloc[j]
            high_j = float(bar['High'])
            low_j = float(bar['Low'])
            t_j = b_df.index[j]

            # Intrabar barrier checks with pessimistic tie-breaking (SL first)
            if side == 'long':
                if low_j <= sl_price:
                    exit_price = sl_price
                    exit_time = t_j
                    exit_reason = 'sl'
                    bars_held = (j - next_idx) + 1
                    break
                elif high_j >= tp_price:
                    exit_price = tp_price
                    exit_time = t_j
                    exit_reason = 'tp'
                    bars_held = (j - next_idx) + 1
                    break
            elif side == 'short':
                if high_j >= sl_price:
                    exit_price = sl_price
                    exit_time = t_j
                    exit_reason = 'sl'
                    bars_held = (j - next_idx) + 1
                    break
                elif low_j <= tp_price:
                    exit_price = tp_price
                    exit_time = t_j
                    exit_reason = 'tp'
                    bars_held = (j - next_idx) + 1
                    break

        # If neither barrier triggered: exit at last valid bar close
        if exit_price is None:
            last_j = valid_bar_indices[-1]
            exit_price = float(b_df.iloc[last_j]['Close'])
            exit_time = b_df.index[last_j]
            exit_reason = 'session_cap' if (last_j + 1 < next_idx + horizon_bars) else 'horizon'
            bars_held = (last_j - next_idx) + 1

        # Costs on exit
        if side == 'long':
            fill_exit_price = exit_price * (1.0 - slippage_rate)
            gross_pnl = shares * (exit_price - entry_price)
            net_pnl = shares * (fill_exit_price - fill_entry_price) - 2.0 * shares * cost_per_share
        else:
            fill_exit_price = exit_price * (1.0 + slippage_rate)
            gross_pnl = shares * (entry_price - exit_price)
            net_pnl = shares * (fill_entry_price - fill_exit_price) - 2.0 * shares * cost_per_share

        total_costs = gross_pnl - net_pnl
        trade_risk = shares * stop_distance
        r_multiple = net_pnl / max(trade_risk, 1e-6)
        return_pct = net_pnl / max(shares * entry_price, 1e-6)

        trades_list.append({
            'ticker': ticker,
            'signal_time': sig_time,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'side': side,
            'shares': shares,
            'entry_price': entry_price,
            'fill_entry_price': fill_entry_price,
            'exit_price': exit_price,
            'fill_exit_price': fill_exit_price,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'costs': total_costs,
            'cost_per_share': cost_per_share,
            'slippage_bps': slippage_bps,
            'return_pct': return_pct,
            'r_multiple': r_multiple,
            'exit_reason': exit_reason,
            'bars_held': bars_held,
            'confidence': conf,
            'size_capped': size_capped,
        })

    trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame(columns=trade_columns)

    # Trading days in backtest span
    all_trading_days = None
    if len(trades_df) > 0:
        start_day = pd.to_datetime(trades_df['entry_time'].min()).tz_convert('America/New_York').date()
        end_day = pd.to_datetime(trades_df['exit_time'].max()).tz_convert('America/New_York').date()
        all_bar_dates = {ts.tz_convert('America/New_York').date() for df in std_bars.values() for ts in df.index}
        span_dates = sorted(d for d in all_bar_dates if start_day <= d <= end_day)
        all_trading_days = span_dates if span_dates else [d.date() for d in pd.bdate_range(start_day, end_day)]

    metrics = compute_metrics(trades_df, initial_capital=initial_capital, trading_days=all_trading_days)
    v, reasons = verdict(metrics)

    return {
        'metrics': metrics,
        'trades': trades_df,
        'verdict': v,
        'reasons': reasons,
        **metrics,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. COMPUTE METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(
    trades: Union[pd.DataFrame, List[Dict[str, Any]]],
    initial_capital: float = 100_000.0,
    trading_days: Optional[List[Any]] = None,
    _is_sub: bool = False,
) -> Dict[str, Any]:
    """Compute institutional performance metrics from executed trades.

    trades_per_day and daily Sharpe use ALL trading days in the backtest span
    (days with no exits count as 0 PnL).
    """
    if isinstance(trades, list):
        trades_df = pd.DataFrame(trades)
    else:
        trades_df = trades

    if trades_df is None or len(trades_df) == 0:
        res: Dict[str, Any] = {
            'n_trades': 0,
            'trades_per_day': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'expectancy': 0.0,
            'expectancy_usd': 0.0,
            'expectancy_r': 0.0,
            'profit_factor': 0.0,
            'net_pnl': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe': 0.0,
            'gross_win': 0.0,
            'gross_loss': 0.0,
            'total_costs': 0.0,
        }
        if not _is_sub:
            res['per_ticker'] = {}
        return res

    n_trades = int(len(trades_df))
    net_pnl_vals = trades_df['net_pnl'].astype(np.float64).values
    wins = net_pnl_vals[net_pnl_vals > 0]
    losses = net_pnl_vals[net_pnl_vals < 0]

    n_wins = len(wins)
    win_rate = float(n_wins / n_trades) if n_trades > 0 else 0.0
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

    expectancy_usd = float(np.mean(net_pnl_vals)) if n_trades > 0 else 0.0
    if 'r_multiple' in trades_df.columns:
        expectancy_r = float(trades_df['r_multiple'].astype(np.float64).mean())
    else:
        expectancy_r = 0.0

    gross_win = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
    if gross_loss > 0:
        profit_factor = float(gross_win / gross_loss)
    elif gross_win > 0:
        profit_factor = 999.0
    else:
        profit_factor = 0.0

    net_pnl = float(np.sum(net_pnl_vals))
    total_costs = float(trades_df['costs'].astype(np.float64).sum()) if 'costs' in trades_df.columns else 0.0

    # -----------------------------------------------------------------------
    # Trading days span & daily Sharpe across ALL trading days (0 on empty days)
    # -----------------------------------------------------------------------
    if trading_days is not None and len(trading_days) > 0:
        all_days = sorted(list(trading_days))
    else:
        entry_dates_temp = pd.to_datetime(trades_df['entry_time']).dt.tz_convert('America/New_York').dt.date
        exit_dates_temp = pd.to_datetime(trades_df['exit_time']).dt.tz_convert('America/New_York').dt.date
        min_d = min(entry_dates_temp.min(), exit_dates_temp.min())
        max_d = max(entry_dates_temp.max(), exit_dates_temp.max())
        all_days = [d.date() for d in pd.bdate_range(min_d, max_d)]

    n_days = max(1, len(all_days))
    trades_per_day = float(n_trades / n_days)

    exit_dates = pd.to_datetime(trades_df['exit_time']).dt.tz_convert('America/New_York').dt.date
    daily_pnl_grouped = trades_df.groupby(exit_dates)['net_pnl'].sum().to_dict()
    daily_pnls = np.array([daily_pnl_grouped.get(d, 0.0) for d in all_days], dtype=np.float64)
    daily_returns = daily_pnls / initial_capital

    if len(daily_returns) >= 2 and float(np.std(daily_returns, ddof=1)) > 1e-12:
        sharpe = float(np.mean(daily_returns) / np.std(daily_returns, ddof=1) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Max Drawdown % (equity curve tracking)
    sort_col = 'exit_time' if 'exit_time' in trades_df.columns else 'entry_time'
    sorted_df = trades_df.sort_values(sort_col)
    cum_pnl = sorted_df['net_pnl'].astype(np.float64).cumsum().values
    equity_curve = np.insert(initial_capital + cum_pnl, 0, initial_capital)
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns_pct = (peaks - equity_curve) / np.maximum(peaks, 1e-9) * 100.0
    max_drawdown_pct = float(np.max(drawdowns_pct)) if len(drawdowns_pct) > 0 else 0.0

    metrics: Dict[str, Any] = {
        'n_trades': n_trades,
        'trades_per_day': round(trades_per_day, 2),
        'win_rate': round(win_rate, 4),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'expectancy': round(expectancy_usd, 2),
        'expectancy_usd': round(expectancy_usd, 2),
        'expectancy_r': round(expectancy_r, 4),
        'profit_factor': round(profit_factor, 2),
        'net_pnl': round(net_pnl, 2),
        'max_drawdown_pct': round(max_drawdown_pct, 2),
        'sharpe': round(sharpe, 2),
        'gross_win': round(gross_win, 2),
        'gross_loss': round(gross_loss, 2),
        'total_costs': round(total_costs, 2),
    }

    # Per-ticker breakdown
    if not _is_sub:
        per_ticker: Dict[str, Any] = {}
        if 'ticker' in trades_df.columns:
            for ticker_name, t_df in trades_df.groupby('ticker'):
                per_ticker[str(ticker_name)] = compute_metrics(
                    t_df, initial_capital=initial_capital, trading_days=all_days, _is_sub=True
                )
        metrics['per_ticker'] = per_ticker

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# 3. VERDICT
# ═══════════════════════════════════════════════════════════════════════════

def verdict(metrics: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Determine strategy verdict: 'PASS' | 'WEAK' | 'FAIL' and list reasons.

    Rules:
      FAIL if n_trades < 100 or profit_factor < 1.0 or expectancy <= 0;
      WEAK if profit_factor < 1.2 or max_drawdown_pct > 20;
      else PASS.
    """
    fail_reasons: List[str] = []
    n_trades = int(metrics.get('n_trades', 0))
    profit_factor = float(metrics.get('profit_factor', 0.0))
    expectancy = float(metrics.get('expectancy', 0.0))
    max_drawdown_pct = float(metrics.get('max_drawdown_pct', 0.0))

    if n_trades < FAIL_MIN_TRADES:
        fail_reasons.append(f"n_trades ({n_trades}) < {FAIL_MIN_TRADES}")
    if profit_factor < FAIL_MIN_PROFIT_FACTOR:
        fail_reasons.append(f"profit_factor ({profit_factor:.2f}) < {FAIL_MIN_PROFIT_FACTOR:.1f}")
    if expectancy <= FAIL_MIN_EXPECTANCY:
        fail_reasons.append(f"expectancy ({expectancy:.2f}) <= {FAIL_MIN_EXPECTANCY:.1f}")

    if fail_reasons:
        return 'FAIL', fail_reasons

    weak_reasons: List[str] = []
    if profit_factor < WEAK_MIN_PROFIT_FACTOR:
        weak_reasons.append(f"profit_factor ({profit_factor:.2f}) < {WEAK_MIN_PROFIT_FACTOR:.1f}")
    if max_drawdown_pct > WEAK_MAX_DRAWDOWN_PCT:
        weak_reasons.append(f"max_drawdown_pct ({max_drawdown_pct:.2f}%) > {WEAK_MAX_DRAWDOWN_PCT:.1f}%")

    if weak_reasons:
        return 'WEAK', weak_reasons

    return 'PASS', []


# ═══════════════════════════════════════════════════════════════════════════
# 4. SAVE REPORT
# ═══════════════════════════════════════════════════════════════════════════

def save_report(
    metrics: Dict[str, Any],
    trades: pd.DataFrame,
    out_dir: Union[str, Path],
    initial_capital: float = 100_000.0,
    verdict_override: Optional[str] = None,
    reasons_override: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Write report.json, trades.csv, and equity_curve.png to out_dir."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if verdict_override is not None:
        v = verdict_override
        reasons = reasons_override or []
    else:
        v, reasons = verdict(metrics)

    # 1. report.json
    report_dict = {
        'verdict': v,
        'reasons': reasons,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'metrics': metrics,
    }
    report_file = out_path / 'report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2, default=_json_default)

    # 2. trades.csv
    trades_file = out_path / 'trades.csv'
    trades_df = trades if isinstance(trades, pd.DataFrame) else pd.DataFrame(trades)
    trades_df.to_csv(trades_file, index=False)

    # 3. equity_curve.png
    curve_file = out_path / 'equity_curve.png'
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=False,
                                   gridspec_kw={'height_ratios': [3, 1]})

    if len(trades_df) > 0:
        sort_col = 'exit_time' if 'exit_time' in trades_df.columns else 'entry_time'
        sorted_trades = trades_df.sort_values(sort_col).reset_index(drop=True)
        cum_pnl = sorted_trades['net_pnl'].astype(np.float64).cumsum().values
        equity_vals = np.insert(initial_capital + cum_pnl, 0, initial_capital)
        trade_nums = np.arange(len(equity_vals))

        ax1.plot(trade_nums, equity_vals, color='#2563eb', linewidth=1.8, label='Portfolio Equity')
        ax1.axhline(initial_capital, color='gray', linestyle='--', alpha=0.6, label='Initial Capital')
        ax1.set_ylabel('Equity ($)')
        ax1.set_title(
            f"Backtest Performance — Verdict: {v} | Net PnL: ${metrics.get('net_pnl', 0):,.2f} | "
            f"Win Rate: {metrics.get('win_rate', 0):.1%} | PF: {metrics.get('profit_factor', 0):.2f}"
        )
        ax1.grid(True, linestyle=':', alpha=0.5)
        ax1.legend(loc='upper left')

        peaks = np.maximum.accumulate(equity_vals)
        dd_pct = (peaks - equity_vals) / np.maximum(peaks, 1e-9) * 100.0
        ax2.fill_between(trade_nums, 0, -dd_pct, color='#dc2626', alpha=0.35, label='Drawdown %')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Trade Number')
        ax2.set_ylim(-max(30.0, float(np.max(dd_pct) * 1.2)), 0)
        ax2.grid(True, linestyle=':', alpha=0.5)
        ax2.legend(loc='lower left')
    else:
        ax1.plot([0, 1], [initial_capital, initial_capital], color='gray', linestyle='--')
        ax1.set_title(f"Backtest Performance — Verdict: {v} (No Trades Executed)")
        ax1.set_ylabel('Equity ($)')
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Drawdown (%)')

    fig.tight_layout()
    plt.savefig(curve_file, dpi=140, bbox_inches='tight')
    plt.close(fig)

    return {
        'report': report_file,
        'trades': trades_file,
        'equity_curve': curve_file,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. CLI & BARS CACHE LOADER
# ═══════════════════════════════════════════════════════════════════════════

def load_bars_from_cache(cache_dir: Union[str, Path], tickers: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Scan directory for cached OHLCV files (.pkl, .csv, .parquet)."""
    import joblib

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        raise FileNotFoundError(f"Bars cache directory does not exist: {cache_dir}")

    bars: Dict[str, pd.DataFrame] = {}
    target_tickers = {t.strip().upper() for t in tickers} if tickers else None

    # Scan for common patterns (.cache_{TICKER}_*.pkl, {TICKER}.csv, {TICKER}.pkl)
    for p in cache_path.glob('*'):
        if p.is_dir():
            continue

        name = p.name
        suffix = p.suffix.lower()
        if suffix not in {'.pkl', '.csv', '.parquet'}:
            continue

        # Extract potential ticker
        matched_ticker = None
        cleaned_stem = p.stem.lstrip('.').replace('cache_', '')
        parts = cleaned_stem.split('_')
        candidate = parts[0].upper()
        if target_tickers is None or candidate in target_tickers:
            matched_ticker = candidate

        if not matched_ticker:
            continue

        try:
            if suffix == '.pkl':
                loaded = joblib.load(p)
            elif suffix == '.csv':
                loaded = pd.read_csv(p, index_col=0, parse_dates=True)
            elif suffix == '.parquet':
                loaded = pd.read_parquet(p)
            else:
                continue

            if isinstance(loaded, pd.DataFrame) and not loaded.empty:
                # Prefer existing higher-quality or longer dataset if duplicate
                if matched_ticker not in bars or len(loaded) > len(bars[matched_ticker]):
                    bars[matched_ticker] = loaded
        except Exception:
            # Skip invalid cache files gracefully
            continue

    return bars


def main() -> None:
    parser = argparse.ArgumentParser(description="Institutional signal backtest & reporting engine.")
    parser.add_argument('--signals', type=str, required=True, help="Path to signals CSV file.")
    parser.add_argument('--bars-cache', type=str, required=True, help="Directory containing cached OHLCV bars.")
    parser.add_argument('--out', type=str, default='reports', help="Output directory for reports.")
    parser.add_argument('--cost-per-share', type=float, default=0.01, help="Commission per share on entry and exit.")
    parser.add_argument('--slippage-bps', type=float, default=2.0, help="Slippage in basis points per side.")
    parser.add_argument('--risk-per-trade', type=float, default=0.005, help="Risk fraction per trade.")
    parser.add_argument('--horizon-bars', type=int, default=12, help="Max holding period in bars.")
    parser.add_argument('--initial-capital', type=float, default=100_000.0, help="Account initial capital.")
    parser.add_argument('--allow-overlap', action='store_true', default=False, help="Allow overlapping positions on same ticker.")
    parser.add_argument('--max-concurrent-positions', type=int, default=5, help="Max concurrent positions across all tickers.")
    parser.add_argument('--max-leverage', type=float, default=1.0, help="Max portfolio leverage cap.")

    args = parser.parse_args()

    signals_path = Path(args.signals)
    if not signals_path.exists():
        print(f"Error: signals file not found at {signals_path}", file=sys.stderr)
        sys.exit(1)

    signals = pd.read_csv(signals_path)
    tickers = signals['ticker'].unique().tolist() if 'ticker' in signals.columns else None

    bars = load_bars_from_cache(args.bars_cache, tickers=tickers)
    if not bars:
        print(f"Error: no matching bar data found in {args.bars_cache} for tickers {tickers}", file=sys.stderr)
        sys.exit(1)

    print(f"\nRunning backtest on {len(signals)} signals across {len(bars)} tickers...")
    result = run_backtest(
        signals=signals,
        bars=bars,
        cost_per_share=args.cost_per_share,
        slippage_bps=args.slippage_bps,
        risk_per_trade=args.risk_per_trade,
        horizon_bars=args.horizon_bars,
        initial_capital=args.initial_capital,
        allow_overlap=args.allow_overlap,
        max_concurrent_positions=args.max_concurrent_positions,
        max_leverage=args.max_leverage,
    )

    out_paths = save_report(
        metrics=result['metrics'],
        trades=result['trades'],
        out_dir=args.out,
        initial_capital=args.initial_capital,
    )

    v = result['verdict']
    print(f"\n{'=' * 50}")
    print(f"  VERDICT: {v}")
    if result['reasons']:
        print(f"  Reasons: {', '.join(result['reasons'])}")
    print(f"{'=' * 50}")
    m = result['metrics']
    print(f"  Trades:          {m['n_trades']}")
    print(f"  Win Rate:        {m['win_rate']:.1%}")
    print(f"  Profit Factor:   {m['profit_factor']:.2f}")
    print(f"  Expectancy:      ${m['expectancy']:.2f} ({m['expectancy_r']:.3f} R)")
    print(f"  Net PnL:         ${m['net_pnl']:,.2f}")
    print(f"  Max Drawdown:    {m['max_drawdown_pct']:.2f}%")
    print(f"  Daily Sharpe:    {m['sharpe']:.2f}")
    print(f"\n  Reports written to: {args.out}/")
    print(f"    - {out_paths['report']}")
    print(f"    - {out_paths['trades']}")
    print(f"    - {out_paths['equity_curve']}")


if __name__ == '__main__':
    main()
