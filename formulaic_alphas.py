"""
formulaic_alphas.py — WorldQuant 101 Formulaic Alphas Vectorized Engine.

Implements core high-performing alphas from Kakushadze (2016)
"101 Formulaic Alphas" for US equities intraday / daily bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# ATOMIC QUANT OPERATORS (Kakushadze 2016 Standard)
# ═══════════════════════════════════════════════════════════════════════════

RANK_WINDOW = 50  # Trailing bars used by rank(); keeps every alpha strictly causal.


def _rolling_last_pct_rank(values: np.ndarray, window: int) -> np.ndarray:
    """Percentile rank (0-1] of each value within its trailing ``window`` bars.

    Uses only the current and previous ``window - 1`` bars, so a value never
    changes when later bars are appended (no lookahead). Windows containing a
    NaN yield NaN. Ties receive the average rank, matching ``Series.rank(pct=True)``.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    out = np.full(n, np.nan)
    if window < 1 or n < window:
        return out
    win = np.lib.stride_tricks.sliding_window_view(values, window)
    last = win[:, -1][:, None]
    less = (win < last).sum(axis=1)
    equal = (win == last).sum(axis=1)
    ranks = (less + (equal + 1) / 2.0) / window
    valid = ~np.isnan(win).any(axis=1)
    out[window - 1:] = np.where(valid, ranks, np.nan)
    return out


def rank(series: pd.Series, window: int = RANK_WINDOW) -> pd.Series:
    """Trailing-window percentile rank in [0, 1] (causal; never uses future bars).

    The original paper ranks across a stock universe at one instant. With a
    single ticker there is no cross-section, and ranking over the whole history
    leaks future data into past rows, so we rank against the trailing window.
    """
    return pd.Series(_rolling_last_pct_rank(series.to_numpy(), window), index=series.index)


def ts_rank(series: pd.Series, window: int = 10) -> pd.Series:
    """Time-series percentile rank over rolling window."""
    return pd.Series(_rolling_last_pct_rank(series.to_numpy(), window), index=series.index)


def delta(series: pd.Series, period: int = 1) -> pd.Series:
    """Current value minus value period bars ago."""
    return series.diff(period)


def ts_argmax(series: pd.Series, window: int = 5) -> pd.Series:
    """Day of maximum value over past window bars (1 = most recent)."""
    return series.rolling(window).apply(
        lambda s: (window - np.argmax(s)) if len(s) == window else np.nan,
        raw=True
    )


def correlation(s1: pd.Series, s2: pd.Series, window: int = 10) -> pd.Series:
    """Rolling Pearson correlation between two series."""
    return s1.rolling(window).corr(s2)


# ═══════════════════════════════════════════════════════════════════════════
# FORMULAIC ALPHA FORMULATIONS
# ═══════════════════════════════════════════════════════════════════════════

def alpha_001(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#1: (rank(ts_argmax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
    Asymmetric volatility spike identifier.
    """
    close = df['Close']
    returns = close.pct_change()
    std_20 = returns.rolling(20).std()

    cond_val = pd.Series(np.where(returns < 0, std_20, close), index=df.index)
    signed_power = np.sign(cond_val) * (cond_val ** 2)
    arg_max = ts_argmax(signed_power, 5)
    return rank(arg_max) - 0.5


def alpha_002(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#2: -1 * correlation(rank(delta(log(volume), 2)), rank((close - open) / open), 6)
    Volume acceleration vs price return divergence.
    """
    vol_clean = df['Volume'].replace(0, np.nan)
    vol_log = np.log(vol_clean)
    delta_vol = delta(vol_log, 2)
    open_ret = (df['Close'] - df['Open']) / df['Open'].replace(0, np.nan)
    return -1.0 * correlation(rank(delta_vol), rank(open_ret), 6)


def alpha_003(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#3: -1 * correlation(rank(open), rank(volume), 10)
    Aggressive opening surge exhaustion.
    """
    return -1.0 * correlation(rank(df['Open']), rank(df['Volume']), 10)


def alpha_004(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#4: -1 * ts_rank(rank(low), 9)
    9-day low price range momentum reversal.
    """
    return -1.0 * ts_rank(rank(df['Low']), 9)


def alpha_005(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#5: rank(open - sum(vwap, 10)/10) * (-1 * abs(rank(close - vwap)))
    VWAP distance stretching factor.
    """
    vwap = df.get('VWAP', (df['High'] + df['Low'] + df['Close']) / 3.0)
    vwap_ma = vwap.rolling(10).mean()
    term1 = rank(df['Open'] - vwap_ma)
    term2 = -1.0 * np.abs(rank(df['Close'] - vwap))
    return term1 * term2


def alpha_006(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    Alpha#6: -1 * correlation(open, volume, 10)
    Opening surge exhaustion: if open is surging with volume, expect mean-reversion.
    """
    corr = correlation(df['Open'], df['Volume'], window=window)
    return -1.0 * corr


def alpha_012(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#12: sign(delta(volume, 1)) * (-1 * delta(close, 1))
    Volume divergence momentum. The price change is expressed as a return
    (divided by the prior close) so the feature is stationary across years.
    """
    vol_delta = delta(df['Volume'], 1)
    close_ret = df['Close'].pct_change(1)
    return np.sign(vol_delta) * (-1.0 * close_ret)


def alpha_014(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#14: (-1 * rank(delta(returns, 3))) * correlation(open, volume, 10)
    Return acceleration vs volume-open correlation.
    """
    ret = df['Close'].pct_change()
    ret_accel = delta(ret, 3)
    corr = correlation(df['Open'], df['Volume'], 10)
    return (-1.0 * rank(ret_accel)) * corr


def alpha_041(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#41: (((high * low)^0.5) - vwap)
    Geometric bar center vs. Volume-Weighted clearing price, divided by close
    so the value is a dimensionless fraction rather than a dollar amount.
    """
    geom_mean = np.sqrt(df['High'] * df['Low'])
    vwap = df.get('VWAP', (df['High'] + df['Low'] + df['Close']) / 3.0)
    return (geom_mean - vwap) / df['Close'].replace(0, np.nan)


def alpha_053(df: pd.DataFrame, window: int = 9) -> pd.Series:
    """
    Alpha#53: -1 * delta((((close - low) - (high - close)) / (close - low + 1e-5)), 9)
    Close-position exhaustion momentum.
    """
    c = df['Close']
    h = df['High']
    l = df['Low']
    denom = (c - l).replace(0, 1e-5)
    val = ((c - l) - (h - c)) / denom
    return -1.0 * delta(val, window)


def alpha_054(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#54: (-1 * (low - close) * (open^5)) / ((low - high) * (close^5))
    Non-linear bar geometry tail absorption.
    """
    l = df['Low']
    c = df['Close']
    o = df['Open']
    h = df['High']
    # (open^5) / (close^5) == (open / close)^5: dividing first avoids float
    # overflow without needing a whole-history mean (which leaked future data).
    price_ratio_5 = (o / c.replace(0, np.nan)) ** 5
    num = -1.0 * (l - c) * price_ratio_5
    denom = (l - h).replace(0, -1e-5)
    return num / denom


def alpha_101(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#101: ((close - open) / ((high - low) + 0.001))
    Bar efficiency / conviction ratio (candle body vs total range).
    """
    denom = (df['High'] - df['Low']) + 0.001
    return (df['Close'] - df['Open']) / denom


def compute_all_alphas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all formulaic alphas and return as a feature DataFrame."""
    alphas = pd.DataFrame(index=df.index)
    alphas['alpha_001'] = alpha_001(df)
    alphas['alpha_002'] = alpha_002(df)
    alphas['alpha_003'] = alpha_003(df)
    alphas['alpha_004'] = alpha_004(df)
    alphas['alpha_005'] = alpha_005(df)
    alphas['alpha_006'] = alpha_006(df)
    alphas['alpha_012'] = alpha_012(df)
    alphas['alpha_014'] = alpha_014(df)
    alphas['alpha_041'] = alpha_041(df)
    alphas['alpha_053'] = alpha_053(df)
    alphas['alpha_054'] = alpha_054(df)
    alphas['alpha_101'] = alpha_101(df)
    # Rolling correlations on near-constant windows (and 0/0 divisions on flat or
    # zero-volume bars) can yield +/-inf. Treat them as missing so that
    # quant_core.compute_features() drops those rows with the other warm-up
    # NaNs; DataFrame.dropna() alone would let inf through into training.
    return alphas.replace([np.inf, -np.inf], np.nan)


def combine_alphas(
    alphas_df: pd.DataFrame,
    weights: Optional[dict[str, float]] = None,
) -> pd.Series:
    """
    Combine multiple formulaic alphas into a single composite alpha signal.
    Normalizes each alpha by rolling z-score before blending.
    """
    normed = pd.DataFrame(index=alphas_df.index)
    for col in alphas_df.columns:
        s = alphas_df[col]
        roll_mean = s.rolling(50, min_periods=10).mean()
        roll_std = s.rolling(50, min_periods=10).std().replace(0, 1.0)
        normed[col] = (s - roll_mean) / roll_std

    if weights is not None:
        total_w = sum(abs(v) for v in weights.values()) or 1.0
        composite = pd.Series(0.0, index=alphas_df.index)
        for col, w in weights.items():
            if col in normed.columns:
                composite += normed[col].fillna(0) * (w / total_w)
        return composite
    else:
        # Equal-weighted blend
        return normed.mean(axis=1)


def optimize_alpha_weights(
    alphas_df: pd.DataFrame,
    close: pd.Series,
    horizon: int = 5,
    method: str = "mean_ic",
) -> dict[str, float]:
    """
    Calculate mathematically optimal weights for combining multiple alphas.
    Supported methods:
      - 'mean_ic': Weight proportional to Rank Information Coefficient (IC).
      - 'inv_vol': Inverse volatility weighting (Equal Risk Contribution proxy).
      - 'equal': Equal weighting (1 / N).
    """
    valid_cols = [c for c in alphas_df.columns if alphas_df[c].notna().sum() > 10]
    if not valid_cols:
        return {c: 1.0 / len(alphas_df.columns) for c in alphas_df.columns}

    weights = {}
    if method == "mean_ic":
        for col in valid_cols:
            ic_dict = evaluate_alpha_ic(alphas_df[col], close, horizons=[horizon])
            ic = ic_dict.get(horizon, 0.0)
            weights[col] = float(ic) if not np.isnan(ic) else 0.0
    elif method == "inv_vol":
        for col in valid_cols:
            std = float(alphas_df[col].std())
            weights[col] = (1.0 / std) if std > 1e-6 and not np.isnan(std) else 0.0
    else:  # equal
        for col in valid_cols:
            weights[col] = 1.0

    total_abs = sum(abs(v) for v in weights.values())
    if total_abs > 1e-6:
        return {k: round(v / total_abs, 4) for k, v in weights.items()}
    else:
        # Fallback to equal weights
        return {k: round(1.0 / len(valid_cols), 4) for k in valid_cols}


# ═══════════════════════════════════════════════════════════════════════════
# ALPHA EVALUATION & RANKING (Information Coefficient & Quantile Spreads)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_alpha_ic(
    alpha: pd.Series,
    close: pd.Series,
    horizons: list[int] = [1, 3, 5, 10],
) -> dict[int, float]:
    """Compute Spearman Rank Information Coefficient (IC) across forward horizons."""
    ic_results = {}
    for h in horizons:
        fwd_ret = close.pct_change(h).shift(-h)
        valid = pd.concat([alpha, fwd_ret], axis=1).dropna()
        if len(valid) > 10:
            ic = valid.iloc[:, 0].corr(valid.iloc[:, 1], method='spearman')
            ic_results[h] = round(float(ic), 4)
        else:
            ic_results[h] = 0.0
    return ic_results


def rank_all_alphas(
    df: pd.DataFrame,
    horizons: list[int] = [1, 3, 5, 10],
) -> pd.DataFrame:
    """Rank all formulaic alphas by Spearman Rank Information Coefficient (IC)."""
    alphas_df = compute_all_alphas(df)
    close = df['Close'].astype(float)

    records = []
    for col in alphas_df.columns:
        ic_dict = evaluate_alpha_ic(alphas_df[col], close, horizons=horizons)
        mean_ic = np.nanmean(list(ic_dict.values())) if ic_dict else 0.0
        rec = {
            'Alpha': col,
            'Mean IC': round(float(mean_ic), 4),
            'Abs IC': round(abs(float(mean_ic)), 4),
        }
        for h in horizons:
            rec[f'IC (h={h})'] = ic_dict.get(h, 0.0)
        records.append(rec)

    leaderboard = pd.DataFrame(records).sort_values(by='Abs IC', ascending=False).reset_index(drop=True)
    return leaderboard


def simulate_alpha_trades(
    alpha: pd.Series,
    close: pd.Series,
    top_quantile: float = 0.8,
    bottom_quantile: float = 0.2,
    horizon: int = 5,
    shares: int = 100,
) -> pd.DataFrame:
    """
    Simulate non-overlapping quantile breakout trades for a formulaic alpha.
    Produces a trade log directly auditable by audit_strategy.py.
    """
    df = pd.DataFrame({'alpha': alpha, 'close': close}).dropna()
    if len(df) < 50:
        return pd.DataFrame()

    q_high = df['alpha'].rolling(50, min_periods=20).quantile(top_quantile)
    q_low = df['alpha'].rolling(50, min_periods=20).quantile(bottom_quantile)

    trades = []
    i = 50
    while i < len(df) - horizon:
        entry_time = df.index[i]
        entry_price = float(df['close'].iloc[i])
        val = df['alpha'].iloc[i]

        if val > q_high.iloc[i]:
            # LONG trade
            exit_time = df.index[i + horizon]
            exit_price = float(df['close'].iloc[i + horizon])
            pnl = (exit_price - entry_price) * shares
            r = (exit_price - entry_price) / (entry_price * 0.01)  # 1% risk proxy
            trades.append({
                'Entry_Timestamp': str(entry_time),
                'Exit_Timestamp': str(exit_time),
                'Side': 'LONG',
                'Net_PnL_USD': round(pnl, 2),
                'R_Multiple': round(r, 2),
                'Regime_Period': str(entry_time)[:7],
            })
            i += horizon
        elif val < q_low.iloc[i]:
            # SHORT trade
            exit_time = df.index[i + horizon]
            exit_price = float(df['close'].iloc[i + horizon])
            pnl = (entry_price - exit_price) * shares
            r = (entry_price - exit_price) / (entry_price * 0.01)
            trades.append({
                'Entry_Timestamp': str(entry_time),
                'Exit_Timestamp': str(exit_time),
                'Side': 'SHORT',
                'Net_PnL_USD': round(pnl, 2),
                'R_Multiple': round(r, 2),
                'Regime_Period': str(entry_time)[:7],
            })
            i += horizon
        else:
            i += 1

    return pd.DataFrame(trades)


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Evaluate and rank WorldQuant 101 Formulaic Alphas on OHLCV data")
    parser.add_argument("--csv", type=str, default=None, help="Path to OHLCV CSV file")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic bars for offline evaluation")
    parser.add_argument("--export-trades", type=str, default=None, help="Alpha name to export trade log for (e.g. alpha_101)")
    parser.add_argument("--out-csv", type=str, default="reports/formulaic_trades.csv", help="Output path for trade log CSV")
    parser.add_argument("--optimize-portfolio", action="store_true", help="Calculate optimal weights and simulate multi-alpha portfolio")
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_file():
            raise FileNotFoundError(f"Data file not found at {args.csv}")
        df_in = pd.read_csv(csv_path)
    else:
        print("  Notice: No --csv supplied. Generating 250 synthetic OHLCV bars for offline evaluation...")
        np.random.seed(42)
        dates = pd.date_range("2026-01-01", periods=250, freq="B")
        ret = np.random.normal(0.0004, 0.015, size=250)
        close = 150.0 * np.exp(np.cumsum(ret))
        high = close * (1 + np.abs(np.random.normal(0, 0.008, size=250)))
        low = close * (1 - np.abs(np.random.normal(0, 0.008, size=250)))
        open_ = (high + low) / 2.0
        volume = np.random.lognormal(mean=14, sigma=0.5, size=250)
        df_in = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)
    if 'Date' in df_in.columns:
        df_in.index = pd.to_datetime(df_in['Date'])
    elif 'timestamp' in df_in.columns:
        df_in.index = pd.to_datetime(df_in['timestamp'])

    # Standardize column casing
    col_map = {c.lower(): c for c in df_in.columns}
    for req in ['open', 'high', 'low', 'close', 'volume']:
        if req in col_map and col_map[req] != req.capitalize():
            df_in[req.capitalize()] = df_in[col_map[req]]

    print("\n" + "=" * 65)
    print("      WORLDQUANT 101 FORMULAIC ALPHAS: RANK IC LEADERBOARD     ")
    print("=" * 65)
    lb = rank_all_alphas(df_in)
    print(lb.to_string(index=False))
    print("=" * 65 + "\n")

    if args.export_trades:
        target_alpha = args.export_trades
        alphas_all = compute_all_alphas(df_in)
        if target_alpha not in alphas_all.columns:
            raise ValueError(f"Unknown alpha '{target_alpha}'. Available: {alphas_all.columns.tolist()}")

        sim_trades = simulate_alpha_trades(alphas_all[target_alpha], df_in['Close'])
        out_p = Path(args.out_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        sim_trades.to_csv(out_p, index=False)
        print(f"Exported {len(sim_trades)} simulated trades for {target_alpha} to: {out_p}")



    if args.optimize_portfolio:
        alphas_all = compute_all_alphas(df_in)
        opt_weights = optimize_alpha_weights(alphas_all, df_in['Close'], horizon=5, method="mean_ic")
        print("\n" + "=" * 65)
        print("      OPTIMAL MULTI-ALPHA FACTOR WEIGHTS (Rank IC Maximized)     ")
        print("=" * 65)
        for a_name, w in sorted(opt_weights.items(), key=lambda x: abs(x[1]), reverse=True):
            if abs(w) > 0.001:
                print(f"  {a_name:12} : {w:+.4f} ({'+' if w >= 0 else ''}{w*100:.1f}%)")
        print("=" * 65)

        composite_series = sum(alphas_all[a] * w for a, w in opt_weights.items())
        port_trades = simulate_alpha_trades(composite_series, df_in['Close'], horizon=5)
        port_csv = Path("reports/optimal_multi_alpha_trades.csv")
        port_csv.parent.mkdir(parents=True, exist_ok=True)
        port_trades.to_csv(port_csv, index=False)
        print(f"\n  Exported {len(port_trades)} simulated multi-alpha trades to: {port_csv}")
