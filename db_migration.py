"""
db_migration.py — Idempotent SQLite schema auto-migration for V3+.

Adds V3+ columns to the paper_trades table without touching existing
V2 rows. Safe to run multiple times.
"""
import sqlite3
from pathlib import Path

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


def migrate_paper_trades_schema(conn: sqlite3.Connection) -> list:
    """Add any missing V3+ columns to the paper_trades table.

    Uses PRAGMA table_info to check existing columns and only adds
    those that are missing. Fully idempotent — safe to run on every
    app startup.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite connection.

    Returns
    -------
    list of str
        Names of columns that were added (empty if already up to date).
    """
    existing = {row[1] for row in conn.execute('PRAGMA table_info(paper_trades)')}
    added = []
    for col_name, col_type in V3_COLUMNS.items():
        if col_name not in existing:
            conn.execute(f'ALTER TABLE paper_trades ADD COLUMN {col_name} {col_type}')
            added.append(col_name)
    if added:
        conn.commit()
    return added


def run_migration():
    """Run the migration on the default database path."""
    conn = sqlite3.connect(DB_PATH)
    try:
        added = migrate_paper_trades_schema(conn)
        if added:
            print(f"[OK] SQLite migration: added columns {added}")
        else:
            print("[OK] SQLite schema already up to date")
    finally:
        conn.close()


if __name__ == '__main__':
    run_migration()
