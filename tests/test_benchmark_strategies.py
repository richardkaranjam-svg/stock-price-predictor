"""
test_benchmark_strategies.py — Unit & Integration tests for benchmark_strategies.py
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from benchmark_strategies import (
    generate_market_data,
    simulate_trend_breakout_strategy,
    simulate_mean_reversion_strategy,
    simulate_multi_alpha_strategy,
    run_all_benchmarks,
)


def test_generate_market_data():
    """Verify synthetic multi-regime market data generator."""
    df = generate_market_data(n_bars=300, seed=123)
    assert len(df) == 300
    for col in ("Open", "High", "Low", "Close", "Volume", "ATR"):
        assert col in df.columns, f"Missing {col}"
    assert (df["High"] >= df["Low"]).all()
    assert (df["ATR"] > 0).all()


def test_simulate_strategies_return_valid_trade_logs():
    """Verify that all three benchmark strategies generate standardized trade records."""
    df = generate_market_data(n_bars=400, seed=42)

    t_breakout = simulate_trend_breakout_strategy(df)
    t_reversion = simulate_mean_reversion_strategy(df)
    t_multi_alpha = simulate_multi_alpha_strategy(df)

    for name, t_df in [
        ("Trend Breakout", t_breakout),
        ("Mean Reversion", t_reversion),
        ("Multi-Alpha", t_multi_alpha),
    ]:
        assert len(t_df) > 0, f"{name} generated 0 trades"
        assert "Entry_Timestamp" in t_df.columns
        assert "Exit_Timestamp" in t_df.columns
        assert "Side" in t_df.columns
        assert "Net_PnL_USD" in t_df.columns
        assert set(t_df["Side"].str.upper().unique()) <= {"LONG", "SHORT"}


def test_run_all_benchmarks_isolated(tmp_path: Path, monkeypatch):
    """Run benchmark generation in isolated directory and check tear sheet generation."""
    # Monkeypatch to small bar count for ultra-fast test execution
    import benchmark_strategies
    monkeypatch.setattr(benchmark_strategies, "generate_market_data", lambda n_bars=1200, seed=42: generate_market_data(n_bars=250, seed=seed))

    out_dir = tmp_path / "benchmarks_test"
    summary_df = run_all_benchmarks(out_base=str(out_dir))

    assert len(summary_df) == 3
    assert set(summary_df["Strategy Name"]) == {
        "Trend Breakout (Donchian)",
        "Mean Reversion (Bollinger)",
        "WorldQuant Multi-Alpha",
    }

    # Verify that factsheets exist on disk
    for factsheet in summary_df["Factsheet"]:
        assert Path(factsheet).is_file(), f"Factsheet not found at {factsheet}"
