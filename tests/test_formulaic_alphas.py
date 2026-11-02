"""
test_formulaic_alphas.py — Unit tests for formulaic_alphas.py
"""

import numpy as np
import pandas as pd
import pytest

from formulaic_alphas import (
    alpha_001,
    alpha_002,
    alpha_003,
    alpha_004,
    alpha_005,
    alpha_006,
    alpha_012,
    alpha_014,
    alpha_041,
    alpha_053,
    alpha_054,
    alpha_101,
    combine_alphas,
    compute_all_alphas,
    evaluate_alpha_ic,
    rank_all_alphas,
    simulate_alpha_trades,
    optimize_alpha_weights,
)


@pytest.fixture
def sample_ohlcv_data():
    dates = pd.date_range("2024-01-01", periods=50, freq="1D")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(50))
    high = close + np.random.uniform(0.5, 2.0, 50)
    low = close - np.random.uniform(0.5, 2.0, 50)
    open_ = low + np.random.uniform(0, 1.0, 50) * (high - low)
    volume = np.random.randint(100_000, 5_000_000, 50)
    vwap = (high + low + close) / 3.0

    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
        "VWAP": vwap,
    }, index=dates)


def test_individual_alphas(sample_ohlcv_data):
    a1 = alpha_001(sample_ohlcv_data)
    assert len(a1) == 50

    a6 = alpha_006(sample_ohlcv_data)
    assert len(a6) == 50

    a12 = alpha_012(sample_ohlcv_data)
    assert len(a12) == 50

    a41 = alpha_041(sample_ohlcv_data)
    assert len(a41) == 50

    a53 = alpha_053(sample_ohlcv_data)
    assert len(a53) == 50

    a101 = alpha_101(sample_ohlcv_data)
    assert len(a101) == 50
    # Alpha 101 should be roughly in [-1, 1]
    assert a101.dropna().abs().max() <= 1.05


def test_compute_all_alphas(sample_ohlcv_data):
    all_alphas = compute_all_alphas(sample_ohlcv_data)
    expected_cols = [
        'alpha_001', 'alpha_002', 'alpha_003', 'alpha_004', 'alpha_005', 'alpha_006',
        'alpha_012', 'alpha_014', 'alpha_041', 'alpha_053', 'alpha_054', 'alpha_101',
    ]
    for col in expected_cols:
        assert col in all_alphas.columns
    assert len(all_alphas) == 50


def test_alpha_ic_and_ranking(sample_ohlcv_data):
    ic_dict = evaluate_alpha_ic(sample_ohlcv_data['Close'].pct_change(), sample_ohlcv_data['Close'])
    assert 1 in ic_dict
    assert isinstance(ic_dict[1], float)

    leaderboard = rank_all_alphas(sample_ohlcv_data)
    assert len(leaderboard) == 12
    assert 'Alpha' in leaderboard.columns
    assert 'Mean IC' in leaderboard.columns


def test_combine_alphas(sample_ohlcv_data):
    all_alphas = compute_all_alphas(sample_ohlcv_data)
    # Equal weight blend
    blend_eq = combine_alphas(all_alphas)
    assert len(blend_eq) == len(sample_ohlcv_data)
    assert isinstance(blend_eq, pd.Series)

    # Weighted blend
    weights = {'alpha_001': 0.5, 'alpha_006': -0.5}
    blend_w = combine_alphas(all_alphas, weights=weights)
    assert len(blend_w) == len(sample_ohlcv_data)
    assert isinstance(blend_w, pd.Series)


def test_quant_core_integration(sample_ohlcv_data):
    from quant_core import (
        FEATURE_COLUMNS,
        EXTENDED_FEATURE_COLUMNS,
        FORMULAIC_ALPHA_COLUMNS,
        compute_features,
    )
    assert len(FEATURE_COLUMNS) == 31
    assert len(FORMULAIC_ALPHA_COLUMNS) == 12
    assert len(EXTENDED_FEATURE_COLUMNS) == 43

    # Default: 31 columns
    f_default = compute_features(sample_ohlcv_data, include_alphas=False)
    assert list(f_default.columns) == FEATURE_COLUMNS

    # With alphas: 43 columns
    f_extended = compute_features(sample_ohlcv_data, include_alphas=True)
    assert list(f_extended.columns) == EXTENDED_FEATURE_COLUMNS
    for col in FORMULAIC_ALPHA_COLUMNS:
        assert col in f_extended.columns



def test_simulate_alpha_trades(sample_ohlcv_data):
    # Expand data to 100 rows so rolling quantile window works
    dates = pd.date_range("2024-01-01", periods=100, freq="1D")
    np.random.seed(42)
    close = pd.Series(100 + np.cumsum(np.random.randn(100)), index=dates)
    alpha = pd.Series(np.random.randn(100), index=dates)

    trades = simulate_alpha_trades(alpha, close, horizon=3)
    assert isinstance(trades, pd.DataFrame)
    if not trades.empty:
        assert 'Entry_Timestamp' in trades.columns
        assert 'Exit_Timestamp' in trades.columns
        assert 'Net_PnL_USD' in trades.columns
        assert 'Side' in trades.columns


def test_optimize_alpha_weights(sample_ohlcv_data):
    all_alphas = compute_all_alphas(sample_ohlcv_data)

    w_ic = optimize_alpha_weights(all_alphas, sample_ohlcv_data['Close'], method='mean_ic')
    assert isinstance(w_ic, dict)
    assert len(w_ic) == 12

    w_vol = optimize_alpha_weights(all_alphas, sample_ohlcv_data['Close'], method='inv_vol')
    assert isinstance(w_vol, dict)
    assert len(w_vol) == 12

    # Weights should sum to ~1.0 in absolute terms
    assert abs(sum(abs(v) for v in w_vol.values()) - 1.0) < 1e-2

