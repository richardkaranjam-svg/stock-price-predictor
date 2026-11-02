"""
test_simulate_session.py — Unit & Integration tests for simulate_session.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from simulate_session import generate_synthetic_session_bars, run_session_simulation


def test_generate_synthetic_session_bars():
    """Verify that synthetic session bars cover 19 twenty-minute bars with required OHLCV+ATR columns."""
    bars = generate_synthetic_session_bars(ticker="AAPL", base_price=150.0)
    assert len(bars) == 19
    for col in ("Open", "High", "Low", "Close", "Volume", "VWAP", "ATR"):
        assert col in bars.columns, f"Missing column {col}"

    assert (bars["High"] >= bars["Low"]).all()
    assert (bars["High"] >= bars["Close"]).all()
    assert (bars["Low"] <= bars["Close"]).all()
    assert (bars["ATR"] > 0).all()

    # Time coverage: first bar 09:30, last bar 15:30
    assert bars.index[0].strftime("%H:%M") == "09:30"
    assert bars.index[-1].strftime("%H:%M") == "15:30"


def test_run_session_simulation_isolated(tmp_path: Path):
    """Run an isolated session simulation in a temporary SQLite database and verify DB integrity."""
    test_db = tmp_path / "sim_test.db"
    res = run_session_simulation(ticker="MSFT", db_path=str(test_db), base_price=300.0)

    assert res["ticker"] == "MSFT"
    assert res["n_trades"] >= 1
    assert isinstance(res["total_pnl"], float)
    assert res["n_settled"] > 0
    assert test_db.is_file()

    # Verify SQLite DB tables and rows
    with sqlite3.connect(test_db) as conn:
        sig_count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        assert sig_count >= 18, f"Expected at least 18 signals, got {sig_count}"

        # Verify hash column integrity
        hashes = [r[0] for r in conn.execute("SELECT features_hash FROM signals WHERE features_hash IS NOT NULL").fetchall()]
        assert len(hashes) == sig_count
        assert all(len(h) == 64 for h in hashes), "SHA-256 features hash must be 64 hex characters"

        # Verify settled outcomes
        settled_outcomes = [r[0] for r in conn.execute("SELECT outcome FROM signals WHERE outcome IS NOT NULL").fetchall()]
        assert len(settled_outcomes) == res["n_settled"]
        assert all(isinstance(o, float) for o in settled_outcomes)
