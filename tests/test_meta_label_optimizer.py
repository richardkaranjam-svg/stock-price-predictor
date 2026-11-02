"""
Unit tests for meta_label_optimizer.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from meta_label_optimizer import run_meta_labeling_optimization


def test_meta_label_optimizer_sweep_and_artifacts(tmp_path: Path):
    # Create synthetic strategy CSV with confidence scores
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    
    # Simulate a strategy where high confidence correlates with higher positive PnL
    confidences = np.random.uniform(0.5, 0.95, size=n)
    base_noise = np.random.normal(0, 100, size=n)
    pnls = (confidences - 0.6) * 500 + base_noise

    df = pd.DataFrame({
        "timestamp": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": "AAPL",
        "side": ["BUY" if i % 2 == 0 else "SELL" for i in range(n)],
        "pnl": pnls,
        "confidence": confidences,
    })

    csv_path = tmp_path / "test_trades.csv"
    df.to_csv(csv_path, index=False)

    out_dir = tmp_path / "meta_report"
    res = run_meta_labeling_optimization(
        str(csv_path),
        output_dir=str(out_dir),
        min_trades_hurdle=50,
    )

    assert "base_metrics" in res
    assert "optimal_metrics" in res
    assert "optimal_threshold" in res
    assert res["optimal_threshold"] >= 0.55
    assert Path(res["chart_path"]).exists()
    assert Path(res["out_dir"]).exists()


def test_meta_label_optimizer_missing_confidence(tmp_path: Path):
    df = pd.DataFrame({
        "timestamp": ["2026-01-01", "2026-01-02"],
        "ticker": ["AAPL", "AAPL"],
        "pnl": [10.0, -5.0],
    })
    csv_path = tmp_path / "no_conf.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="No confidence or probability column"):
        run_meta_labeling_optimization(str(csv_path), output_dir=str(tmp_path / "out"))
