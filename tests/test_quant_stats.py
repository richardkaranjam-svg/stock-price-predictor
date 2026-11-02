"""
test_quant_stats.py — Unit tests for quant_stats.py
"""

import numpy as np
import pytest
from quant_stats import (
    compute_skew_kurtosis,
    compute_probabilistic_sharpe_ratio,
    compute_expected_max_sharpe,
    compute_deflated_sharpe_ratio,
    compute_var_cvar,
)


def test_skew_kurtosis():
    np.random.seed(42)
    norm_returns = np.random.randn(500)
    skew, kurt = compute_skew_kurtosis(norm_returns)
    assert abs(skew) < 0.3
    assert abs(kurt - 3.0) < 0.5


def test_probabilistic_sharpe_ratio():
    np.random.seed(42)
    # Strong positive strategy returns
    pos_returns = np.random.normal(loc=0.01, scale=0.02, size=200)
    psr = compute_probabilistic_sharpe_ratio(pos_returns, benchmark_sr=0.0)
    assert psr > 0.95

    # Negative strategy returns
    neg_returns = np.random.normal(loc=-0.01, scale=0.02, size=200)
    psr_neg = compute_probabilistic_sharpe_ratio(neg_returns, benchmark_sr=0.0)
    assert psr_neg < 0.05


def test_deflated_sharpe_ratio():
    np.random.seed(42)
    returns = np.random.normal(loc=0.002, scale=0.01, size=250)

    # With 1 trial, DSR == PSR
    dsr_1 = compute_deflated_sharpe_ratio(returns, n_trials=1)
    psr = compute_probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)
    assert abs(dsr_1 - psr) < 1e-4

    # With 100 trials tested, DSR should decrease (raising the statistical threshold)
    dsr_100 = compute_deflated_sharpe_ratio(returns, n_trials=100)
    assert dsr_100 < dsr_1


def test_var_and_cvar():
    np.random.seed(42)
    returns = np.random.normal(loc=0.0, scale=0.02, size=500)
    var_95, cvar_95 = compute_var_cvar(returns, confidence=0.95)

    assert var_95 > 0
    assert cvar_95 >= var_95  # Expected shortfall must be at least as severe as VaR
