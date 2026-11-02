"""
quant_stats.py — Institutional Statistical Risk & Overfitting Metrics.

Implements Marcos López de Prado & David Bailey (2014) formulations:
- Probabilistic Sharpe Ratio (PSR)
- Deflated Sharpe Ratio (DSR)
- Parametric & Historical Value-at-Risk (VaR)
- Conditional Value-at-Risk (CVaR / Expected Shortfall)
"""

from __future__ import annotations

import math
from typing import Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


def compute_skew_kurtosis(returns: Union[pd.Series, np.ndarray]) -> Tuple[float, float]:
    """Compute sample skewness and kurtosis (Fisher or Pearson standard)."""
    clean = pd.Series(returns).dropna().values
    if len(clean) < 4:
        return 0.0, 3.0
    skew = float(stats.skew(clean, bias=False))
    kurt = float(stats.kurtosis(clean, fisher=False, bias=False))  # Pearson kurtosis (normal = 3)
    return skew, kurt


def compute_probabilistic_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray],
    benchmark_sr: float = 0.0,
    annualization_factor: float = 1.0,
) -> float:
    """
    Compute Probabilistic Sharpe Ratio (PSR) adjusting for skewness and kurtosis.
    Returns probability in [0, 1] that true SR > benchmark_sr.
    """
    clean = pd.Series(returns).dropna().values
    n = len(clean)
    if n < 5:
        return 0.5

    mean = float(np.mean(clean))
    std = float(np.std(clean, ddof=1))
    if std <= 1e-9:
        return 1.0 if mean > benchmark_sr else 0.0

    sr = (mean / std) * math.sqrt(annualization_factor)
    skew, kurt = compute_skew_kurtosis(clean)

    # Lopez de Prado (2014) standard error of Sharpe ratio
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr ** 2)
    if denom <= 0:
        denom = 1e-6
    sigma_sr = math.sqrt(denom / (n - 1.0))

    z_score = (sr - benchmark_sr) / sigma_sr
    psr = float(stats.norm.cdf(z_score))
    return max(0.0, min(1.0, psr))


def compute_expected_max_sharpe(
    n_trials: int,
    n_obs: int,
    var_sr: float = 1.0,
) -> float:
    """
    Expected maximum Sharpe ratio under the null hypothesis of no skill across N trials.
    Euler-Mascheroni constant gamma = 0.5772156649.
    """
    if n_trials <= 1:
        return 0.0
    gamma = 0.57721566490153286
    term1 = math.sqrt(2.0 * math.log(n_trials))
    term2 = gamma / term1
    em_sr = (term1 + term2) * math.sqrt(var_sr) / math.sqrt(n_obs)
    return float(em_sr)


def compute_deflated_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray],
    n_trials: int = 1,
    var_sr: float = 1.0,
    annualization_factor: float = 1.0,
) -> float:
    """
    Compute Deflated Sharpe Ratio (DSR) controlling for selection bias across N trials.
    If DSR >= 0.95, strategy passes institutional false-positive rejection test.
    """
    clean = pd.Series(returns).dropna().values
    n = len(clean)
    if n < 5:
        return 0.5

    sr_star = compute_expected_max_sharpe(n_trials=n_trials, n_obs=n, var_sr=var_sr)
    return compute_probabilistic_sharpe_ratio(
        returns=clean,
        benchmark_sr=sr_star,
        annualization_factor=annualization_factor,
    )


def compute_var_cvar(
    returns: Union[pd.Series, np.ndarray],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR / Expected Shortfall).
    Returned as positive loss percentages (e.g. 0.02 = 2% loss).
    """
    clean = pd.Series(returns).dropna().values
    if len(clean) < 10:
        return 0.0, 0.0

    cutoff = 1.0 - confidence
    var = -float(np.percentile(clean, cutoff * 100.0))
    tail_losses = clean[clean <= -var]
    cvar = -float(np.mean(tail_losses)) if len(tail_losses) > 0 else var
    return max(0.0, var), max(0.0, cvar)
