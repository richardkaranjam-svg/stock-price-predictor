"""
db_migration.py — Idempotent SQLite schema auto-migration for V3+.

Adds V3+ columns to the paper_trades table and maintains the signals table
for logging model predictions and PaperTrader decisions.
Safe to run multiple times.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

DB_PATH = Path(__file__).resolve().parent / 'paper_trades.db'

V3_COLUMNS = {
    'atr_at_entry': 'REAL',
    'ev_atr_value': 'REAL',
    'take_profit_price': 'REAL',
    'stop_loss_price': 'REAL',
    'ratchet_triggered': 'INTEGER DEFAULT 0',
    'confidence': 'REAL',
    'model_version': "TEXT DEFAULT 'V3+'",
}

SIGNALS_COLUMNS = {
    'timestamp_utc': 'TEXT NOT NULL',
    'ticker': 'TEXT NOT NULL',
    'model_version': 'TEXT',
    'side': 'TEXT',
    'confidence': 'REAL',
    'price': 'REAL',
    'features_hash': 'TEXT',
    'outcome': 'REAL',
    # Where the row came from: 'predict' (predict_v3) or 'trader' (PaperTrader decision).
    'source': 'TEXT',
    # Raw PaperTrader action (e.g. 'buy', 'hold', 'eod_flatten_sell', 'market_closed').
    'action': 'TEXT',
}

# Only these sides get an outcome filled in. 'exit' (closing an existing position)
# and 'none' (market_closed, kill-switch, errors, ratchet) are not new directional
# signals, so scoring them as long/short would pollute the outcome statistics.
OUTCOME_SIDES = ('long', 'short', 'flat')

SIGNALS_CREATE_TABLE_SQL = '''CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc TEXT NOT NULL,
    ticker TEXT NOT NULL,
    model_version TEXT,
    side TEXT,
    confidence REAL,
    price REAL,
    features_hash TEXT,
    outcome REAL,
    source TEXT,
    action TEXT
)'''


def migrate_paper_trades_schema(conn_or_path: Any = None) -> list:
    """Add any missing V3+ columns to the paper_trades table.

    Uses PRAGMA table_info to check existing columns and only adds
    those that are missing. Fully idempotent — safe to run on every
    app startup.

    Parameters
    ----------
    conn_or_path : sqlite3.Connection | str | Path | None
        Active SQLite connection or DB path.

    Returns
    -------
    list of str
        Names of columns that were added (empty if already up to date).
    """
    def _run(conn: sqlite3.Connection) -> list:
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
        existing = {row[1] for row in conn.execute('PRAGMA table_info(paper_trades)')}
        added = []
        for col_name, col_type in V3_COLUMNS.items():
            if col_name not in existing:
                conn.execute(f'ALTER TABLE paper_trades ADD COLUMN {col_name} {col_type}')
                added.append(col_name)
        if added:
            conn.commit()
        return added

    if isinstance(conn_or_path, sqlite3.Connection):
        return _run(conn_or_path)
    else:
        path = conn_or_path or DB_PATH
        with sqlite3.connect(path) as conn:
            return _run(conn)


def migrate_signals_schema(conn_or_path: Any = None) -> list:
    """Create the signals table if missing, and ensure all expected columns exist.

    Fully idempotent — safe to run on every app startup.

    Parameters
    ----------
    conn_or_path : sqlite3.Connection | str | Path | None
        Active SQLite connection or DB path.

    Returns
    -------
    list of str
        Names of columns added if missing.
    """
    def _run(conn: sqlite3.Connection) -> list:
        conn.execute(SIGNALS_CREATE_TABLE_SQL)
        existing = {row[1] for row in conn.execute('PRAGMA table_info(signals)')}
        added = []
        for col_name, col_type in SIGNALS_COLUMNS.items():
            if col_name not in existing:
                conn.execute(f'ALTER TABLE signals ADD COLUMN {col_name} {col_type}')
                added.append(col_name)
        if added:
            conn.commit()
        return added

    if isinstance(conn_or_path, sqlite3.Connection):
        return _run(conn_or_path)
    else:
        path = conn_or_path or DB_PATH
        with sqlite3.connect(path) as conn:
            return _run(conn)


def log_signal(
    conn_or_path: Any = None,
    timestamp_utc: Optional[str] = None,
    ticker: Optional[str] = None,
    side: Optional[str] = None,
    confidence: Optional[float] = None,
    price: Optional[float] = None,
    features_hash: Optional[str] = None,
    outcome: Optional[float] = None,
    source: Optional[str] = None,
    action: Optional[str] = None,
    *,
    model_version: Optional[str] = 'V3+',
    **kwargs,
) -> int:
    """Append a signal row to the signals table.

    Parameters
    ----------
    conn_or_path : sqlite3.Connection | str | Path | None
        Existing SQLite connection or database path. If None, uses DB_PATH.
    timestamp_utc : str, optional
        ISO 8601 UTC timestamp. Defaults to now.
    ticker : str
        Stock ticker symbol.
    model_version : str, optional
        Model version (e.g. 'V3+').
    side : str, optional
        'long', 'short', 'flat' (scored by fill_signal_outcomes), or
        'exit' / 'none' (logged but never scored).
    confidence : float, optional
        Confidence value.
    price : float, optional
        Price at signal time.
    features_hash : str, optional
        Deterministic hash of the input features.
    outcome : float, optional
        Outcome filled later after horizon_bars pass.
    source : str, optional
        'predict' or 'trader'.
    action : str, optional
        Raw PaperTrader action string.

    Returns
    -------
    int
        The row id of the inserted signal.
    """
    if timestamp_utc is None:
        timestamp_utc = datetime.now(timezone.utc).isoformat()

    sql = '''INSERT INTO signals
    (timestamp_utc, ticker, model_version, side, confidence, price, features_hash,
     outcome, source, action)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
    params = (timestamp_utc, ticker, model_version, side, confidence, price, features_hash,
              outcome, source, action)

    if isinstance(conn_or_path, sqlite3.Connection):
        migrate_signals_schema(conn_or_path)
        cur = conn_or_path.execute(sql, params)
        conn_or_path.commit()
        return cur.lastrowid
    else:
        path = conn_or_path or DB_PATH
        with sqlite3.connect(path) as conn:
            migrate_signals_schema(conn)
            cur = conn.execute(sql, params)
            conn.commit()
            return cur.lastrowid


def fill_signal_outcomes(
    conn_or_path: Any = None,
    bars: Optional[Any] = None,
    horizon_bars: int = 12,
) -> int:
    """Fill outcome for pending signals in the signals table after horizon_bars have elapsed.

    Only rows with side in OUTCOME_SIDES and a recorded price are scored.
    Outcome is the fractional return from the logged price to the Close
    horizon_bars later: long = +return, short = -return, flat = raw asset return.

    Parameters
    ----------
    conn_or_path : sqlite3.Connection | str | Path | None
        Database connection or path. Defaults to DB_PATH.
    bars : pd.DataFrame | dict[str, pd.DataFrame] | None
        Bar DataFrame (with DatetimeIndex and Close column) or dict of {ticker: DataFrame}.
        If None, attempts to load bars via get_latest_stock_data.
    horizon_bars : int, default 12
        Number of bars after signal to evaluate outcome.

    Returns
    -------
    int
        Number of signal rows updated with an outcome.
    """
    def _execute(conn: sqlite3.Connection) -> int:
        migrate_signals_schema(conn)
        placeholders = ', '.join('?' for _ in OUTCOME_SIDES)
        cursor = conn.execute(
            'SELECT id, timestamp_utc, ticker, side, price FROM signals '
            f'WHERE outcome IS NULL AND price IS NOT NULL AND price > 0 '
            f'AND side IN ({placeholders})',
            OUTCOME_SIDES,
        )
        rows = cursor.fetchall()
        if not rows:
            return 0

        updated_count = 0
        for row_id, ts_str, ticker, side, entry_price in rows:
            # Resolve bars for this ticker
            b_df = None
            if isinstance(bars, dict):
                b_df = bars.get(ticker)
            elif isinstance(bars, pd.DataFrame):
                b_df = bars
            elif bars is None:
                try:
                    from predict import get_latest_stock_data
                    b_df = get_latest_stock_data(ticker)
                except Exception:
                    b_df = None

            if b_df is None or b_df.empty:
                continue

            # Find Close column
            close_col = next((c for c in ('Close', 'close', 'c') if c in b_df.columns), None)
            if not close_col:
                continue

            # Standardize index to UTC
            idx = b_df.index
            if idx.tz is None:
                idx = idx.tz_localize('UTC')
            else:
                idx = idx.tz_convert('UTC')

            try:
                sig_time = pd.Timestamp(ts_str)
                if sig_time.tz is None:
                    sig_time = sig_time.tz_localize('UTC')
                else:
                    sig_time = sig_time.tz_convert('UTC')
            except Exception:
                continue

            # Locate the bar matching or right after signal
            pos = int(idx.searchsorted(sig_time, side='right')) - 1
            if pos < 0:
                pos = 0

            target_pos = pos + horizon_bars
            if target_pos >= len(b_df):
                # horizon_bars have not passed yet
                continue

            exit_price = float(b_df.iloc[target_pos][close_col])
            base_price = float(entry_price)

            side_str = str(side).lower()
            if side_str == 'short':
                outcome = (base_price - exit_price) / base_price
            else:
                # 'long' return, or raw asset return for 'flat'
                outcome = (exit_price - base_price) / base_price

            conn.execute('UPDATE signals SET outcome = ? WHERE id = ?', (float(outcome), row_id))
            updated_count += 1

        if updated_count > 0:
            conn.commit()
        return updated_count

    if isinstance(conn_or_path, sqlite3.Connection):
        return _execute(conn_or_path)
    else:
        path = conn_or_path or DB_PATH
        with sqlite3.connect(path) as conn:
            return _execute(conn)


def run_migration(db_path: Optional[Path] = None):
    """Run migrations on the database path."""
    target = db_path or DB_PATH
    conn = sqlite3.connect(target)
    try:
        added_trades = migrate_paper_trades_schema(conn)
        added_signals = migrate_signals_schema(conn)
        if added_trades or added_signals:
            print(f"[OK] SQLite migration: paper_trades={added_trades}, signals={added_signals}")
        else:
            print("[OK] SQLite schema already up to date")
    finally:
        conn.close()


if __name__ == '__main__':
    run_migration()
