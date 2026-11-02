"""
quant_core.py — Shared Feature Engine, Uniqueness Math & Portable Inference.

V3+ institutional-grade module imported by:
  - colab_cloud_trainer.py  (cloud training)
  - data_loader.py          (local live feature computation)
  - predict.py              (local inference)
  - auto_trader.py          (local execution)
  - ensemble.py             (backward-compatible shim)

Resolves Issues: #3, #7, #10, #11, #12, #16, #17
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: List[str] = [
    # --- 23 Core Technical & Session Features ---
    'ret_1', 'ret_3', 'ret_6',
    'rsi_14', 'rsi_slope_3',
    'macd_line', 'macd_signal', 'macd_hist', 'macd_hist_slope',
    'vwap_dev_atr', 'vwap_slope',
    'atr_pct', 'atr_ratio_20',
    'bb_pct_b', 'bb_width_norm',
    'vol_surge_20', 'vol_trend_5',
    'ema_9_21_spread', 'ema_21_50_spread',
    'high_low_range_atr', 'close_pos_in_bar',
    'minutes_to_close', 'session_phase',
    # --- 8 Institutional Upgrades ---
    'frac_diff_close', 'overnight_gap_atr',
    'spy_ret_1', 'spy_ret_3',
    'rel_strength_spy_1', 'idiosyncratic_alpha_3',
    'spy_vwap_dev_atr', 'spy_vol_regime_ratio',
]

FORMULAIC_ALPHA_COLUMNS: List[str] = [
    'alpha_001', 'alpha_002', 'alpha_003', 'alpha_004', 'alpha_005', 'alpha_006',
    'alpha_012', 'alpha_014', 'alpha_041', 'alpha_053', 'alpha_054', 'alpha_101',
]

EXTENDED_FEATURE_COLUMNS: List[str] = FEATURE_COLUMNS + FORMULAIC_ALPHA_COLUMNS


_MARKET_OPEN_HOUR = 9
_MARKET_OPEN_MIN = 30
_MARKET_CLOSE_HOUR = 16
_MARKET_CLOSE_MIN = 0
_BAR_MINUTES = 20
_RTH_MINUTES = 390  # Regular trading hours: 09:30–16:00 ET


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: EXACT 09:30-ANCHORED 20-MINUTE RESAMPLER  (Issue #7)
# ═══════════════════════════════════════════════════════════════════════════

def resample_bars_20min(
    df: pd.DataFrame,
    tz: str = 'America/New_York',
) -> pd.DataFrame:
    """Resample minute-level bars into exact 09:30-anchored 20-minute bars.

    Solves the 390-minute resampling bug (Issue #7) by using
    ``resample('20min', offset='10min')`` which anchors buckets to
    :10, :30, :50 past the hour → producing bars starting at
    09:30, 09:50, 10:10, ..., 15:30 ET.

    The 10-minute stub 15:50–16:00 ET (which falls after the 15:45 ET
    flatten cutoff) is dropped.

    Also computes **Intraday Cumulative VWAP** resetting at 09:30 ET daily.

    Parameters
    ----------
    df : pd.DataFrame
        Minute-level OHLCV with a DatetimeIndex (tz-aware or naive UTC).
    tz : str
        IANA timezone for US Eastern (handles EDT/EST automatically).

    Returns
    -------
    pd.DataFrame
        20-minute bars with columns [Open, High, Low, Close, Volume, VWAP]
        and a timezone-naive DatetimeIndex.
    """
    if df is None or df.empty:
        return df

    import pytz
    et = pytz.timezone(tz)

    # --- Ensure ET-aware index ---
    idx = df.index
    if idx.tzinfo is not None:
        et_idx = idx.tz_convert(et)
    else:
        et_idx = idx.tz_localize('UTC').tz_convert(et)

    work = df.copy()
    work.index = et_idx

    # --- Filter to Regular Trading Hours: 09:30 <= time < 15:50 ET ---
    # We keep up to 15:49 to form the 15:30–15:50 bucket, then drop
    # anything in the 15:50–16:00 stub.
    t = work.index
    minutes_since_midnight = t.hour * 60 + t.minute
    rth_mask = (
        (t.weekday < 5) &
        (minutes_since_midnight >= _MARKET_OPEN_HOUR * 60 + _MARKET_OPEN_MIN) &
        (minutes_since_midnight < 15 * 60 + 50)  # < 15:50
    )
    work = work[rth_mask]
    if work.empty:
        return work

    # --- Compute typical price for VWAP before resampling ---
    typical_price = (work['High'] + work['Low'] + work['Close']) / 3.0
    work['_tp_x_vol'] = typical_price * work['Volume']

    # --- Group by session date to prevent cross-day bar merging ---
    session_date = work.index.normalize()

    # Resample within each session date using 20-min buckets offset by 10min
    def _resample_day(day_df: pd.DataFrame) -> pd.DataFrame:
        """Resample a single trading day's minute bars to 20-min bars."""
        resampled = day_df.resample(
            '20min', offset='10min', label='left', closed='left'
        ).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            '_tp_x_vol': 'sum',
        })
        # Drop any bar with NaN open (empty bucket)
        resampled = resampled.dropna(subset=['Open'])

        # Drop 15:50 stub if it snuck in
        bar_minutes = resampled.index.hour * 60 + resampled.index.minute
        resampled = resampled[bar_minutes < 15 * 60 + 50]

        return resampled

    chunks = []
    for date_val, day_group in work.groupby(session_date):
        r = _resample_day(day_group)
        if not r.empty:
            chunks.append(r)

    if not chunks:
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP'])

    result = pd.concat(chunks).sort_index()

    # --- Intraday Cumulative VWAP (resets each day) ---
    result_session = result.index.normalize()
    cum_tp_vol = result['_tp_x_vol'].groupby(result_session).cumsum()
    cum_vol = result['Volume'].groupby(result_session).cumsum()
    result['VWAP'] = cum_tp_vol / cum_vol.replace(0, np.nan)
    result = result.drop(columns=['_tp_x_vol'])

    # --- Strip timezone for local code compatibility ---
    result.index = result.index.tz_localize(None)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: OVERNIGHT-GAP-FREE FRACTIONAL DIFFERENTIATION  (Issue #12)
# ═══════════════════════════════════════════════════════════════════════════

def _get_ffd_weights(d: float, thres: float = 1e-3) -> np.ndarray:
    """Compute FFD kernel weights until they drop below threshold.

    Returns a 1-D array of weights [w_0, w_1, w_2, ...] for the
    Fixed-Window Fractional Differentiation filter.
    """
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < thres:
            break
        weights.append(w)
        k += 1
    return np.array(weights[::-1], dtype=np.float64)  # oldest-first


def build_gap_adjusted_series(
    close: pd.Series,
    session_date: pd.Series,
    prev_session_close: pd.Series,
) -> pd.Series:
    """Construct the gap-adjusted continuous intraday log-price series.

    Within each session, we accumulate *intraday* log-returns so that
    overnight jumps never enter the multi-day FFD memory kernel.
    The first bar of each day is anchored to the previous session's
    last close.

    Parameters
    ----------
    close : pd.Series
        Bar close prices with the same index as session_date.
    session_date : pd.Series
        Normalized date for each bar (same length as close).
    prev_session_close : pd.Series
        Previous session's last close mapped to each bar.

    Returns
    -------
    pd.Series
        Gap-adjusted continuous log-price series.
    """
    log_close = np.log(close.values.astype(np.float64))
    dates = session_date.values
    prev_close = prev_session_close.values.astype(np.float64)

    result = np.empty_like(log_close)
    prev_date = None
    cum = 0.0

    for i in range(len(log_close)):
        current_date = dates[i]
        if current_date != prev_date:
            # New session: anchor to previous session's close
            if np.isfinite(prev_close[i]) and prev_close[i] > 0:
                cum = np.log(prev_close[i])
            else:
                cum = log_close[i]
            prev_date = current_date

        # Intraday return from first bar of day
        if i > 0 and dates[i] == dates[i - 1]:
            cum += log_close[i] - log_close[i - 1]

        result[i] = cum

    return pd.Series(result, index=close.index, name='gap_adj_log_price')


def frac_diff_ffd(
    series: pd.Series,
    d: float = 0.40,
    thres: float = 1e-3,
) -> pd.Series:
    """Apply Fixed-Window Fractional Differentiation to a series.

    Parameters
    ----------
    series : pd.Series
        The gap-adjusted continuous log-price series.
    d : float
        Fractional differentiation order (0 < d < 1). 0.40 balances
        stationarity and memory retention.
    thres : float
        Minimum absolute weight to include in the kernel.

    Returns
    -------
    pd.Series
        Fractionally differentiated series (NaN where kernel exceeds
        available history).
    """
    weights = _get_ffd_weights(d, thres)
    width = len(weights)
    values = series.values.astype(np.float64)
    n = len(values)

    result = np.full(n, np.nan)
    for i in range(width - 1, n):
        result[i] = np.dot(weights, values[i - width + 1: i + 1])

    return pd.Series(result, index=series.index, name='frac_diff_close')


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: 31 STATIONARY, FEED-INVARIANT FEATURES  (Issue #11)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using exponential moving average of gains/losses."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _day_boundary_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    session_date: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute ATR that is aware of day boundaries.

    Uses the True Range definition but prevents the previous close
    component from spanning overnight gaps.  Within each day the
    previous close is just the prior bar's close; at the first bar
    of a new day we use (High - Low) only.
    """
    prev_close = close.shift(1)
    same_day = session_date == session_date.shift(1)

    # True Range with day-boundary awareness
    tr1 = high - low
    tr2 = (high - prev_close).abs().where(same_day, tr1)
    tr3 = (low - prev_close).abs().where(same_day, tr1)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(period, min_periods=period).mean()


def compute_features(
    bars_20m: pd.DataFrame,
    spy_20m: Optional[pd.DataFrame] = None,
    tz: str = 'America/New_York',
    include_alphas: bool = False,
) -> pd.DataFrame:
    """Compute the full 31-feature matrix for V3+.

    All price-distance features are normalized by 14-bar Day-Boundary-
    Aware ATR. All volume features are dimensionless rolling ratios.
    This makes features invariant across SIP and IEX feeds (Tweak 2).

    Parameters
    ----------
    bars_20m : pd.DataFrame
        20-minute OHLCV bars with VWAP column (from resample_bars_20min).
    spy_20m : pd.DataFrame or None
        Matching 20-minute SPY bars. If None, institutional features
        are filled with 0.0 (graceful degradation).
    tz : str
        IANA timezone for session-aware computations.

    Returns
    -------
    pd.DataFrame
        DataFrame with exactly ``FEATURE_COLUMNS`` columns.
    """
    if bars_20m is None or bars_20m.empty:
        raise ValueError("bars_20m must be a non-empty DataFrame with OHLCV columns.")

    df = bars_20m.copy()
    feat = pd.DataFrame(index=df.index)

    # --- Session date for day-boundary awareness ---
    try:
        import pytz
        et = pytz.timezone(tz)
        if df.index.tzinfo is not None:
            et_idx = df.index.tz_convert(et)
        else:
            # Assume naive index is already in ET
            et_idx = df.index
        session_date = pd.Series(et_idx.normalize(), index=df.index)
    except Exception:
        session_date = pd.Series(df.index.normalize(), index=df.index)
        et_idx = df.index

    close = df['Close'].astype(np.float64)
    high = df['High'].astype(np.float64)
    low = df['Low'].astype(np.float64)
    volume = df['Volume'].astype(np.float64)
    open_price = df['Open'].astype(np.float64) if 'Open' in df.columns else close

    # --- ATR (day-boundary-aware, 14-bar) ---
    atr = _day_boundary_atr(high, low, close, session_date, period=14)
    atr_safe = atr.replace(0, np.nan)

    # ── 1–3. Returns ──
    feat['ret_1'] = close.pct_change(1)
    feat['ret_3'] = close.pct_change(3)
    feat['ret_6'] = close.pct_change(6)

    # ── 4–5. RSI ──
    rsi = _compute_rsi(close, 14)
    feat['rsi_14'] = rsi
    feat['rsi_slope_3'] = rsi.diff(3)

    # ── 6–9. MACD (12/26/9) ──
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    feat['macd_line'] = macd_line / atr_safe
    feat['macd_signal'] = macd_signal / atr_safe
    feat['macd_hist'] = macd_hist / atr_safe
    feat['macd_hist_slope'] = macd_hist.diff(3) / atr_safe

    # ── 10–11. VWAP deviation & slope ──
    if 'VWAP' in df.columns:
        vwap = df['VWAP'].astype(np.float64)
    else:
        # Fallback: 20-bar rolling VWAP
        tp = (high + low + close) / 3.0
        vwap = (tp * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    vwap_dev = close - vwap
    feat['vwap_dev_atr'] = vwap_dev / atr_safe
    feat['vwap_slope'] = (vwap_dev / atr_safe).diff(3)

    # ── 12–13. ATR features ──
    feat['atr_pct'] = atr / close.replace(0, np.nan)
    atr_20 = atr.rolling(20, min_periods=14).mean()
    feat['atr_ratio_20'] = atr / atr_20.replace(0, np.nan)

    # ── 14–15. Bollinger Bands ──
    ma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    bb_upper = ma_20 + 2 * std_20
    bb_lower = ma_20 - 2 * std_20
    bb_width = bb_upper - bb_lower
    feat['bb_pct_b'] = (close - bb_lower) / bb_width.replace(0, np.nan)
    feat['bb_width_norm'] = bb_width / atr_safe

    # ── 16–17. Volume features (dimensionless ratios) ──
    vol_ma_20 = volume.rolling(20).mean()
    feat['vol_surge_20'] = volume / vol_ma_20.replace(0, np.nan)
    ema5_vol = volume.ewm(span=5, adjust=False).mean()
    ema20_vol = volume.ewm(span=20, adjust=False).mean()
    feat['vol_trend_5'] = ema5_vol / ema20_vol.replace(0, np.nan)

    # ── 18–19. EMA spreads (ATR-normalized) ──
    ema_9 = close.ewm(span=9, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    feat['ema_9_21_spread'] = (ema_9 - ema_21) / atr_safe
    feat['ema_21_50_spread'] = (ema_21 - ema_50) / atr_safe

    # ── 20. High-Low range / ATR ──
    feat['high_low_range_atr'] = (high - low) / atr_safe

    # ── 21. Close position within bar [0, 1] ──
    bar_range = (high - low).replace(0, np.nan)
    feat['close_pos_in_bar'] = (close - low) / bar_range

    # ── 22. Minutes to close ──
    try:
        if hasattr(et_idx, 'hour'):
            mins_since_open = (et_idx.hour - 9) * 60 + (et_idx.minute - 30)
        else:
            mins_since_open = (df.index.hour - 9) * 60 + (df.index.minute - 30)
        feat['minutes_to_close'] = np.clip(
            _RTH_MINUTES - np.clip(mins_since_open, 0, _RTH_MINUTES), 0, _RTH_MINUTES
        )
    except Exception:
        feat['minutes_to_close'] = 195.0  # midday default

    # ── 23. Session phase (0=open ≤90m, 1=mid, 2=close ≥300m) ──
    mins_from_open = _RTH_MINUTES - feat['minutes_to_close']
    feat['session_phase'] = np.where(
        mins_from_open <= 90, 0,
        np.where(mins_from_open >= 300, 2, 1)
    ).astype(np.float64)

    # ── 24. Fractionally differentiated close (overnight-gap-free) ──
    # Build previous session close map
    last_close_per_day = close.groupby(session_date).last()
    prev_sess_close_map = last_close_per_day.shift(1)
    prev_session_close = session_date.map(prev_sess_close_map)

    gap_adj = build_gap_adjusted_series(close, session_date, prev_session_close)
    feat['frac_diff_close'] = frac_diff_ffd(gap_adj, d=0.40, thres=1e-3)

    # ── 25. Overnight gap / ATR ──
    day_open_price = open_price.groupby(session_date).transform('first')
    prev_day_close = session_date.map(prev_sess_close_map)
    overnight_gap = day_open_price - prev_day_close
    feat['overnight_gap_atr'] = overnight_gap / atr_safe

    # ── 26–31. SPY / cross-sectional features ──
    if spy_20m is not None and not spy_20m.empty:
        spy = spy_20m.reindex(df.index, method='ffill')
        spy_close = spy['Close'].astype(np.float64)
        spy_volume = spy['Volume'].astype(np.float64) if 'Volume' in spy.columns else volume

        feat['spy_ret_1'] = spy_close.pct_change(1)
        feat['spy_ret_3'] = spy_close.pct_change(3)

        # Relative strength: AAPL ret - SPY ret
        feat['rel_strength_spy_1'] = feat['ret_1'] - feat['spy_ret_1']

        # Idiosyncratic alpha via 50-bar rolling beta
        aapl_ret = close.pct_change(1)
        spy_ret = spy_close.pct_change(1)
        cov_50 = aapl_ret.rolling(50).cov(spy_ret)
        var_50 = spy_ret.rolling(50).var()
        beta = cov_50 / var_50.replace(0, np.nan)
        aapl_ret_3 = close.pct_change(3)
        spy_ret_3 = spy_close.pct_change(3)
        feat['idiosyncratic_alpha_3'] = aapl_ret_3 - beta * spy_ret_3

        # SPY VWAP deviation / ATR
        if 'VWAP' in spy.columns:
            spy_vwap = spy['VWAP'].astype(np.float64)
        else:
            spy_tp = (spy['High'] + spy['Low'] + spy_close) / 3.0 if 'High' in spy.columns else spy_close
            spy_vwap = (spy_tp * spy_volume).rolling(20).sum() / spy_volume.rolling(20).sum().replace(0, np.nan)
        feat['spy_vwap_dev_atr'] = (spy_close - spy_vwap) / atr_safe

        # SPY volume regime ratio
        spy_vol_ema5 = spy_volume.ewm(span=5, adjust=False).mean()
        spy_vol_ema20 = spy_volume.ewm(span=20, adjust=False).mean()
        feat['spy_vol_regime_ratio'] = spy_vol_ema5 / spy_vol_ema20.replace(0, np.nan)
    else:
        # Graceful degradation: fill SPY features with 0.0
        for col in ['spy_ret_1', 'spy_ret_3', 'rel_strength_spy_1',
                     'idiosyncratic_alpha_3', 'spy_vwap_dev_atr', 'spy_vol_regime_ratio']:
            feat[col] = 0.0

    if include_alphas:
        from formulaic_alphas import compute_all_alphas
        alphas = compute_all_alphas(df)
        feat = pd.concat([feat, alphas], axis=1)
        target_cols = EXTENDED_FEATURE_COLUMNS
    else:
        target_cols = FEATURE_COLUMNS

    # --- Ensure exact column order ---
    feat = feat.reindex(columns=target_cols)

    # --- Drop rows with NaN (warm-up period) ---
    feat = feat.dropna()

    return feat


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: O(N) UNIQUENESS & PER-FOLD WEIGHTING  (Issue #10, #17)
# ═══════════════════════════════════════════════════════════════════════════

def compute_vectorized_uniqueness(t1_indices: np.ndarray) -> np.ndarray:
    """Compute exact sample uniqueness in O(N) using a NumPy difference array.

    For each sample *i* with label horizon ending at ``t1_indices[i]``,
    uniqueness is the average of 1/c_t for t in [i, t1_i], where c_t is
    the concurrency (number of active labels) at time t.

    Parameters
    ----------
    t1_indices : np.ndarray
        1-D integer array of length N where ``t1_indices[i]`` is the
        (inclusive) end index of the label horizon for sample i.
        Must satisfy ``t1_indices[i] >= i`` for all i.

    Returns
    -------
    np.ndarray
        1-D float array of length N with uniqueness values in (0, 1].
    """
    t1 = np.asarray(t1_indices, dtype=np.int64)
    n = len(t1)
    if n == 0:
        return np.array([], dtype=np.float64)
    if np.any(t1 < np.arange(n, dtype=np.int64)):
        # span = t1 - i + 1 would be <= 0 → division by zero / negative uniqueness
        bad = int(np.argmax(t1 < np.arange(n, dtype=np.int64)))
        raise ValueError(f"t1_indices[i] must be >= i (first violation at i={bad}, t1={int(t1[bad])})")

    max_idx = int(t1.max())
    T = max_idx + 2  # +2 for the difference-array sentinel

    # --- Step 1: Build concurrency c_t via difference array ---
    diff = np.zeros(T, dtype=np.int64)
    starts = np.arange(n, dtype=np.int64)
    np.add.at(diff, starts, 1)
    np.add.at(diff, t1 + 1, -1)
    concurrency = np.cumsum(diff)  # c_t for each t

    # --- Step 2: Prefix sum of 1/c_t for fast range queries ---
    inv_c = np.zeros(T, dtype=np.float64)
    nonzero = concurrency[:T] > 0
    inv_c[nonzero] = 1.0 / concurrency[:T][nonzero]
    prefix_inv_c = np.zeros(T + 1, dtype=np.float64)
    prefix_inv_c[1:] = np.cumsum(inv_c)

    # --- Step 3: Compute uniqueness per sample ---
    # u_i = sum(1/c_t for t in [i, t1_i]) / (t1_i - i + 1)
    span = t1 - starts + 1
    range_sum = prefix_inv_c[t1 + 1] - prefix_inv_c[starts]
    uniqueness = range_sum / span.astype(np.float64)

    return uniqueness


def compute_fold_sample_weights(
    u_fold: np.ndarray,
    oldest_weight_floor: float = 0.50,
    min_clip: float = 0.25,
    max_clip: float = 2.50,
) -> Tuple[np.ndarray, float, float]:
    """Compute per-fold information-clock recency + uniqueness weights.

    Time decay is computed strictly inside the passed fold as a function
    of cumulative uniqueness (information clock), not wall-clock bar index.

    Parameters
    ----------
    u_fold : np.ndarray
        1-D uniqueness array for the fold's training samples.
    oldest_weight_floor : float
        Minimum recency weight for the oldest sample (default 0.50).
    min_clip, max_clip : float
        Clip bounds for the combined weight.

    Returns
    -------
    tuple of (final_weights, mean_uniqueness, n_eff_ratio)
        - final_weights: 1-D array, mean = 1.0
        - mean_uniqueness: float, mean of u_fold
        - n_eff_ratio: Kish N_eff / N (should be >= 0.70)
    """
    u = np.asarray(u_fold, dtype=np.float64)
    n = len(u)
    if n == 0:
        return np.array([], dtype=np.float64), 0.0, 0.0

    mean_u = float(np.mean(u))

    # --- Information-clock recency decay ---
    # Cumulative uniqueness as the time axis
    cum_u = np.cumsum(u)
    total_u = cum_u[-1]
    if total_u > 0:
        # Linear decay: oldest → floor, newest → 1.0
        normalized_time = cum_u / total_u  # [0 → 1]
        decay = oldest_weight_floor + (1.0 - oldest_weight_floor) * normalized_time
    else:
        decay = np.ones(n)

    # --- Square-root damped uniqueness × decay ---
    w = np.sqrt(u) * decay

    # --- Clip ---
    w = np.clip(w, min_clip, max_clip)

    # --- Normalize to mean 1.0 ---
    w_mean = np.mean(w)
    if w_mean > 0:
        w = w / w_mean
    else:
        w = np.ones(n)

    # --- Kish Effective Sample Size ---
    # Identical weights → ratio exactly 1.0.  Guard the denominator against
    # all-zero / non-finite weights (e.g. NaN uniqueness) → no ZeroDivision.
    sum_w2 = float(np.sum(w ** 2))
    if not np.isfinite(sum_w2) or sum_w2 <= 0.0:
        w = np.ones(n)
        sum_w2 = float(n)
    n_eff = (np.sum(w) ** 2) / sum_w2
    n_eff_ratio = float(n_eff / n)

    return w, mean_u, n_eff_ratio


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: PORTABLE INFERENCE & DictModelAdapter  (Issue #3, #16)
# ═══════════════════════════════════════════════════════════════════════════

def predict_calibrated_probabilities(
    artifact_dict: Dict[str, Any],
    X: np.ndarray,
) -> np.ndarray:
    """Generate calibrated 3-class probabilities from a V3+ plain-dict artifact.

    Reconstructs LightGBM from its native text string and blends with
    the pickled RandomForest using 60/40 weighting, then applies
    Platt/Softmax probability calibration.

    The reconstructed LightGBM Booster is cached inside the dict at
    ``artifact_dict['_lgb_booster_cache']`` so it is not re-parsed on
    every tick.

    Parameters
    ----------
    artifact_dict : dict
        Plain Python dict loaded from ``stock_price_model.pkl`` containing:
        ``lgb_model_str``, ``rf_model``, ``calib_coef``, ``calib_intercept``,
        ``feature_columns``, ``config``, ``env_versions``.
    X : np.ndarray or pd.DataFrame
        Feature matrix with shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Calibrated probability matrix of shape (n_samples, 3) for
        classes [-1, 0, +1] (DOWN, CHOP, UP).
    """
    import lightgbm as lgb

    # --- Reconstruct LightGBM booster (cached) ---
    if '_lgb_booster_cache' not in artifact_dict:
        booster = lgb.Booster(model_str=artifact_dict['lgb_model_str'])
        artifact_dict['_lgb_booster_cache'] = booster
    else:
        booster = artifact_dict['_lgb_booster_cache']

    # --- LightGBM raw predictions ---
    if isinstance(X, pd.DataFrame):
        # Enforce the TRAINING column order (never trust caller order).
        cols = list(artifact_dict.get('feature_columns') or FEATURE_COLUMNS)
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise KeyError(f"Inference features missing columns required by artifact: {missing}")
        X_arr = X[cols].values.astype(np.float64)
    else:
        X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)
    n_expected = booster.num_feature()
    if X_arr.shape[1] != n_expected:
        raise ValueError(
            f"Feature dimension mismatch: got {X_arr.shape[1]} columns, "
            f"LightGBM booster expects {n_expected}."
        )

    p_lgb_raw = booster.predict(X_arr)
    if p_lgb_raw.ndim == 1:
        # Binary output — shouldn't happen for 3-class but handle gracefully
        p_lgb = np.column_stack([1 - p_lgb_raw, p_lgb_raw])
    else:
        p_lgb = p_lgb_raw

    # --- RandomForest predictions ---
    rf_model = artifact_dict['rf_model']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)  # version mismatch
        p_rf = rf_model.predict_proba(X_arr)

    # --- Ensure both have 3 columns ---
    n_classes = 3
    if p_lgb.shape[1] < n_classes:
        pad = np.zeros((p_lgb.shape[0], n_classes - p_lgb.shape[1]))
        p_lgb = np.hstack([p_lgb, pad])
    if p_rf.shape[1] < n_classes:
        # Class-aware placement via rf.classes_ (encoded 0=DOWN,1=CHOP,2=UP);
        # right-padding would shift columns when a LOW class is absent.
        rf_classes = np.asarray(getattr(rf_model, 'classes_', np.arange(p_rf.shape[1]))).astype(int)
        aligned = np.zeros((p_rf.shape[0], n_classes))
        for col, cls in enumerate(rf_classes[:p_rf.shape[1]]):
            if 0 <= cls < n_classes:
                aligned[:, cls] = p_rf[:, col]
        p_rf = aligned

    # --- Blend: 60% LGB + 40% RF ---
    p_blend = 0.60 * p_lgb + 0.40 * p_rf

    # --- Platt / Softmax calibration ---
    calib_coef = np.asarray(artifact_dict.get('calib_coef', np.ones(n_classes)))
    calib_intercept = np.asarray(artifact_dict.get('calib_intercept', np.zeros(n_classes)))

    # Apply calibration: logit-space scaling
    # Clip to prevent log(0)
    p_clipped = np.clip(p_blend, 1e-15, 1 - 1e-15)
    logits = np.log(p_clipped / (1 - p_clipped))
    calibrated_logits = logits * calib_coef + calib_intercept

    # Softmax to get final calibrated probabilities
    exp_logits = np.exp(calibrated_logits - calibrated_logits.max(axis=1, keepdims=True))
    p_calibrated = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    return p_calibrated


class DictModelAdapter:
    """Wrapper that gives a plain-dict V3+ artifact a scikit-learn interface.

    Provides ``.predict_proba(X)``, ``.predict(X)``, and ``.classes_``
    so any code expecting a trained sklearn model works seamlessly.

    Resolves Gap 2 (predict.py ``AttributeError: 'dict' has no attribute
    'predict_proba'``) and Tweak 5 (cross-environment pickle safety).

    Parameters
    ----------
    artifact_dict : dict
        Plain Python dict loaded from ``stock_price_model.pkl``.
    """

    def __init__(self, artifact_dict: Dict[str, Any]):
        self._artifact = artifact_dict
        self.classes_ = np.array([-1, 0, 1])
        self.model_version = artifact_dict.get('model_version', 'V3+')
        self.feature_columns = artifact_dict.get('feature_columns', FEATURE_COLUMNS)

    def predict_proba(self, X) -> np.ndarray:
        """Return calibrated probabilities for classes [-1, 0, +1]."""
        return predict_calibrated_probabilities(self._artifact, X)

    def predict(self, X) -> np.ndarray:
        """Return predicted class labels [-1, 0, +1]."""
        probas = self.predict_proba(X)
        return self.classes_[probas.argmax(axis=1)]

    def get_config(self) -> Dict[str, Any]:
        """Return the artifact's config sub-dict."""
        return self._artifact.get('config', {})

    def get_env_versions(self) -> Dict[str, str]:
        """Return the environment versions used during training."""
        return self._artifact.get('env_versions', {})

    @property
    def artifact(self) -> Dict[str, Any]:
        """Direct access to the underlying plain dict."""
        return self._artifact


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: CLOSED-BAR FILTER  (Issue #18)
# ═══════════════════════════════════════════════════════════════════════════

def filter_closed_bars(
    bars_20m: pd.DataFrame,
    now_et: Optional[Any] = None,
    tz: str = 'America/New_York',
) -> pd.DataFrame:
    """Remove the currently-forming (unclosed) 20-minute bar.

    A bar is considered closed when
    ``bar_open_timestamp + 20 minutes <= current_wall_clock_ET``.

    Parameters
    ----------
    bars_20m : pd.DataFrame
        20-minute bars with a DatetimeIndex.
    now_et : datetime-like or None
        Current wall-clock time in ET. If None, uses ``datetime.now()``.
    tz : str
        IANA timezone.

    Returns
    -------
    pd.DataFrame
        Only fully closed bars.
    """
    if bars_20m is None or bars_20m.empty:
        return bars_20m

    if now_et is None:
        try:
            import pytz
            now_et = pd.Timestamp.now(tz=pytz.timezone(tz))
        except ImportError:
            from datetime import timezone, timedelta
            now_et = pd.Timestamp.now(tz=timezone(timedelta(hours=-5)))

    # Convert bar open timestamps to aware if needed
    bar_opens = bars_20m.index
    if bar_opens.tzinfo is None:
        try:
            import pytz
            bar_opens = bar_opens.tz_localize(pytz.timezone(tz))
        except Exception:
            pass

    bar_close_times = bar_opens + pd.Timedelta(minutes=_BAR_MINUTES)
    mask = bar_close_times <= now_et
    return bars_20m[mask]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: HELPER — GET CURRENT ET TIME
# ═══════════════════════════════════════════════════════════════════════════

def get_et_now() -> 'pd.Timestamp':
    """Return the current wall-clock time in US Eastern."""
    try:
        import pytz
        return pd.Timestamp.now(tz=pytz.timezone('America/New_York'))
    except ImportError:
        from datetime import timezone, timedelta
        return pd.Timestamp.now(tz=timezone(timedelta(hours=-5)))

