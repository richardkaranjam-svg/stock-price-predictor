"""
test_audit_strategy.py — Unit test for audit_strategy.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from audit_strategy import run_strategy_backtest, generate_html_tear_sheet


def test_audit_strategy_end_to_end(tmp_path: Path):
    # Create minimal trade log CSV
    trades_data = {
        "Entry_Timestamp": ["2024-01-02 09:30", "2024-01-03 10:00", "2024-01-04 11:00"],
        "Exit_Timestamp": ["2024-01-02 16:00", "2024-01-03 15:30", "2024-01-04 16:00"],
        "Side": ["LONG", "SHORT", "LONG"],
        "Net_PnL_USD": [500.0, -200.0, 750.0],
        "R_Multiple": [1.5, -1.0, 2.0],
        "Regime_Period": ["2024 Test", "2024 Test", "2024 Test"],
    }
    csv_file = tmp_path / "test_trades.csv"
    pd.DataFrame(trades_data).to_csv(csv_file, index=False)

    out_dir = tmp_path / "output"
    results = run_strategy_backtest(str(csv_file), output_dir=str(out_dir), export_html=True)

    assert "verdict" in results
    assert "metrics" in results
    assert results["metrics"]["n_trades"] == 3
    assert results["metrics"]["net_pnl"] == 1050.0
    assert results["html_file"] is not None
    assert Path(results["html_file"]).is_file()

    html_content = Path(results["html_file"]).read_text(encoding="utf-8")
    assert "Institutional Performance Tear Sheet" in html_content
    assert "$1,050.00" in html_content


def test_audit_strategy_arbitrary_broker_columns(tmp_path: Path):
    """Ensure broker exports with Profit, Date/Time, Type, currency symbols work seamlessly."""
    broker_trades = {
        "Date/Time": ["2024-02-01 10:00", "2024-02-02 11:30", "2024-02-03 14:00"],
        "Type": ["Buy", "Sell", "Buy"],
        "Profit": ["$1,200.50", "(350.00)", "$850.00"],
        "Commission": [5.0, 5.0, 5.0],
    }
    csv_file = tmp_path / "broker_trades.csv"
    pd.DataFrame(broker_trades).to_csv(csv_file, index=False)

    out_dir = tmp_path / "output_broker"
    results = run_strategy_backtest(str(csv_file), output_dir=str(out_dir), export_html=True)
    assert results["metrics"]["n_trades"] == 3
    assert results["metrics"]["net_pnl"] == pytest.approx(1700.50)
    assert results["html_file"] is not None
    assert Path(results["html_file"]).is_file()
