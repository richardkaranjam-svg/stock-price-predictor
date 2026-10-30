"""
evaluate_v3_model.py — Out-of-Sample V3+ Walk-Forward Signal Evaluation Engine.

Generates out-of-sample signals from the V3+ pipeline (PurgedWalkForwardCV,
quant_core.compute_features, triple_barrier_label) ONLY on test folds, converts
predictions to the standardized signal format, and runs backtest_report.run_backtest.
NEVER scores on training rows.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import pytz

import colab_cloud_trainer as cct
import quant_core
from backtest_report import run_backtest, save_report, verdict


def _safe_align_proba(model: Any, proba: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Class-aware probability alignment (handles folds with missing classes)."""
    proba = np.asarray(proba, dtype=np.float64)
    classes = np.asarray(getattr(model, 'classes_', np.arange(proba.shape[1]))).astype(int)
    out = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
    for col, cls in enumerate(classes[:proba.shape[1]]):
        if 0 <= cls < n_classes:
            out[:, cls] = proba[:, col]
    row_sum = out.sum(axis=1, keepdims=True)
    return np.divide(out, row_sum, out=np.full_like(out, 1.0 / n_classes), where=row_sum > 0)


def generate_oos_signals(
    bars_20m: pd.DataFrame,
    spy_20m: Optional[pd.DataFrame] = None,
    ticker: str = 'AAPL',
    n_splits: int = cct.N_CV_FOLDS,
    embargo_bars: int = cct.EMBARGO_BARS,
    min_confidence: float = cct.MIN_CONFIDENCE,
    tp_mult: float = cct.TP_ATR_MULT,
    sl_mult: float = cct.SL_ATR_MULT,
    slippage_bps: float = cct.SLIPPAGE_BPS,
    lgb_params: Optional[Dict[str, Any]] = None,
    rf_params: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Train on training folds and generate signals STRICTLY on test folds.

    Never scores on training rows. Calibration strictly purges against sample space
    and falls back to identity if calibration samples are insufficient.
    Returns (signals_df, bars_dict).
    """
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    ticker = str(ticker).strip().upper()

    # Ensure tz-aware UTC DatetimeIndex
    df_bars = bars_20m.copy()
    if df_bars.index.tzinfo is None:
        df_bars.index = pd.to_datetime(df_bars.index).tz_localize('UTC')
    else:
        df_bars.index = pd.to_datetime(df_bars.index).tz_convert('UTC')
    df_bars = df_bars.sort_index()

    # Pre-align SPY if provided
    df_spy = None
    if spy_20m is not None and not spy_20m.empty:
        df_spy = spy_20m.copy()
        if df_spy.index.tzinfo is None:
            df_spy.index = pd.to_datetime(df_spy.index).tz_localize('UTC')
        else:
            df_spy.index = pd.to_datetime(df_spy.index).tz_convert('UTC')
        df_spy = df_spy.sort_index()
        # Reindex SPY to exact bars index and forward-fill within session
        df_spy = df_spy.reindex(df_bars.index)
        df_spy = df_spy.groupby(df_spy.index.normalize()).ffill()

    # 1. Feature computation (31 features)
    features = quant_core.compute_features(df_bars, df_spy)
    if features.empty:
        raise ValueError("No valid features computed. Check input data length and warm-up requirements.")

    # 2. Session date & Day-boundary ATR calculation
    et = pytz.timezone('America/New_York')
    session_date = pd.Series(df_bars.index.tz_convert(et).normalize(), index=df_bars.index)
    atr = quant_core._day_boundary_atr(
        df_bars['High'], df_bars['Low'], df_bars['Close'], session_date, period=14
    )

    # 3. Triple-barrier labeling
    labels, t1_indices = cct.triple_barrier_label(
        df_bars, atr, session_date,
        tp_mult=tp_mult, sl_mult=sl_mult, max_horizon=cct.MAX_HORIZON_BARS,
    )

    # Align features with valid ATR bars
    valid_mask = ~np.isnan(atr.values) & (atr.values > 0)
    common_idx = features.index.intersection(df_bars.index[valid_mask])
    features = features.loc[common_idx]

    pos_map = {ts: i for i, ts in enumerate(df_bars.index)}
    pos_indices = np.array([pos_map[ts] for ts in common_idx])
    labels = labels[pos_indices]
    t1_indices = t1_indices[pos_indices]

    if len(features) == 0:
        raise ValueError("No valid samples remaining after aligning features and ATR.")

    # Reindex t1_indices relative to filtered sample space (sample indices 0..N-1)
    sample_indices = np.arange(len(features))
    t1_reindexed = np.searchsorted(pos_indices, t1_indices, side='right') - 1
    t1_reindexed = np.maximum(t1_reindexed, sample_indices).astype(np.int64)

    # 4. Uniqueness
    uniqueness = quant_core.compute_vectorized_uniqueness(t1_reindexed)

    # 5. Purged Walk-Forward Cross Validation
    X = features.values.astype(np.float64)
    y = (np.asarray(labels) + 1).astype(np.int64)  # 0=DOWN, 1=CHOP, 2=UP
    n_samples = len(X)

    # Adjust n_splits and embargo_bars dynamically if dataset is smaller
    effective_embargo = min(embargo_bars, max(2, n_samples // 20))
    test_size = n_samples // (n_splits + 1)
    if test_size <= effective_embargo:
        n_splits = max(2, (n_samples // (effective_embargo + 2)) - 1)
        test_size = n_samples // (n_splits + 1)
        effective_embargo = max(1, test_size // 3)

    cv = cct.PurgedWalkForwardCV(n_splits=n_splits, embargo_bars=effective_embargo)

    # Model parameters
    lgb_base_params = cct.LGB_PARAMS.copy()
    if lgb_params:
        lgb_base_params.update(lgb_params)

    rf_base_params = cct.RF_PARAMS.copy()
    if rf_params:
        rf_base_params.update(rf_params)

    signals_list: List[Dict[str, Any]] = []

    for fold_num, (train_idx, test_idx) in enumerate(cv.split(n_samples, t1_reindexed)):
        if len(test_idx) == 0 or len(train_idx) < 10:
            continue

        # 85/15 fit/calibration split within training fold
        split_point = int(len(train_idx) * 0.85)
        fit_idx = train_idx[:split_point]
        cal_idx = train_idx[split_point:]

        # Purge calibration boundary using t1_reindexed (sample space) consistently
        if len(cal_idx) > 0:
            cal_purge_mask = t1_reindexed[fit_idx] < cal_idx[0]
            fit_idx = fit_idx[cal_purge_mask]

        # NEVER fall back to calibrating on training rows - skip calibration (identity) instead
        skip_calibration = (len(fit_idx) < 10 or len(cal_idx) < 10)
        if skip_calibration:
            fit_idx = train_idx
            cal_idx = np.array([], dtype=int)

        X_fit, y_fit = X[fit_idx], y[fit_idx]
        X_test = X[test_idx]

        # Weights
        fold_w, mean_u, _ = quant_core.compute_fold_sample_weights(uniqueness[fit_idx])
        rf_max_samples = float(np.clip(mean_u, 0.12, 0.35))
        lgb_subsample = float(np.clip(mean_u * 1.5, 0.25, 0.60))

        # LightGBM
        f_lgb_params = lgb_base_params.copy()
        f_lgb_params['subsample'] = lgb_subsample
        f_lgb_params['subsample_freq'] = 1
        f_lgb = lgb.LGBMClassifier(**f_lgb_params)
        f_lgb.fit(X_fit, y_fit, sample_weight=fold_w)

        # Random Forest
        f_rf_params = rf_base_params.copy()
        f_rf_params['max_samples'] = rf_max_samples
        f_rf = RandomForestClassifier(**f_rf_params)
        f_rf.fit(X_fit, y_fit, sample_weight=fold_w)

        # Blend & Calibration
        calib_coef = np.ones(3)
        calib_intercept = np.zeros(3)

        if not skip_calibration and len(cal_idx) >= 10:
            X_cal, y_cal = X[cal_idx], y[cal_idx]
            p_lgb_cal = _safe_align_proba(f_lgb, f_lgb.predict_proba(X_cal))
            p_rf_cal = _safe_align_proba(f_rf, f_rf.predict_proba(X_cal))
            p_blend_cal = 0.60 * p_lgb_cal + 0.40 * p_rf_cal
            p_clipped = np.clip(p_blend_cal, 1e-15, 1 - 1e-15)
            logits_cal = np.log(p_clipped / (1 - p_clipped))

            for c in range(3):
                y_binary = (y_cal == c).astype(np.float64)
                if y_binary.sum() >= 3 and (1 - y_binary).sum() >= 3:
                    try:
                        lr = LogisticRegression(max_iter=1000, solver='lbfgs')
                        lr.fit(logits_cal[:, c:c+1], y_binary)
                        calib_coef[c] = float(lr.coef_[0, 0])
                        calib_intercept[c] = float(lr.intercept_[0])
                    except Exception:
                        pass

        # PREDICT STRICTLY ON TEST FOLD (NEVER ON TRAINING ROWS)
        p_lgb_test = _safe_align_proba(f_lgb, f_lgb.predict_proba(X_test))
        p_rf_test = _safe_align_proba(f_rf, f_rf.predict_proba(X_test))
        p_blend_test = 0.60 * p_lgb_test + 0.40 * p_rf_test
        p_clipped_test = np.clip(p_blend_test, 1e-15, 1 - 1e-15)
        logits_test = np.log(p_clipped_test / (1 - p_clipped_test))

        cal_logits_test = logits_test * calib_coef + calib_intercept
        exp_l = np.exp(cal_logits_test - cal_logits_test.max(axis=1, keepdims=True))
        p_cal_test = exp_l / exp_l.sum(axis=1, keepdims=True)

        pred_classes = p_cal_test.argmax(axis=1)

        # Convert predictions to signals for each test row
        for row_i, sample_k in enumerate(test_idx):
            ts = features.index[sample_k]
            pred_cls = pred_classes[row_i]  # 0=DOWN, 1=CHOP, 2=UP
            p_win = float(p_cal_test[row_i, 2] if pred_cls == 2 else (p_cal_test[row_i, 0] if pred_cls == 0 else 0.0))

            ev = p_win * tp_mult - (1.0 - p_win) * sl_mult - 2.0 * (slippage_bps / 10000.0)

            if p_win >= min_confidence and ev > 0:
                side = 'long' if pred_cls == 2 else ('short' if pred_cls == 0 else 'flat')
            else:
                side = 'flat'

            signals_list.append({
                'timestamp': ts,
                'ticker': ticker,
                'side': side,
                'confidence': round(p_win, 4),
            })

    signals_df = pd.DataFrame(signals_list)
    if not signals_df.empty:
        signals_df = signals_df.sort_values('timestamp').reset_index(drop=True)
    else:
        signals_df = pd.DataFrame(columns=['timestamp', 'ticker', 'side', 'confidence'])

    # Prepare bars dict with ATR column for backtesting triple barriers
    bars_with_atr = df_bars.copy()
    bars_with_atr['ATR'] = atr.reindex(df_bars.index).bfill().ffill()
    bars_dict = {ticker: bars_with_atr}

    return signals_df, bars_dict


def evaluate_v3(
    bars_20m: pd.DataFrame,
    spy_20m: Optional[pd.DataFrame] = None,
    ticker: str = 'AAPL',
    out_dir: Optional[Union[str, Path]] = None,
    n_splits: int = cct.N_CV_FOLDS,
    cost_per_share: float = 0.01,
    slippage_bps: float = 2.0,
    risk_per_trade: float = 0.005,
    horizon_bars: int = 12,
    initial_capital: float = 100_000.0,
    min_confidence: float = cct.MIN_CONFIDENCE,
    allow_overlap: bool = False,
    max_concurrent_positions: int = 5,
    max_leverage: float = 1.0,
    min_bars: int = 0,
) -> Dict[str, Any]:
    """End-to-end evaluation: generate OOS signals and run backtest."""
    signals_df, bars_dict = generate_oos_signals(
        bars_20m=bars_20m,
        spy_20m=spy_20m,
        ticker=ticker,
        n_splits=n_splits,
        min_confidence=min_confidence,
    )

    result = run_backtest(
        signals=signals_df,
        bars=bars_dict,
        cost_per_share=cost_per_share,
        slippage_bps=slippage_bps,
        risk_per_trade=risk_per_trade,
        horizon_bars=horizon_bars,
        initial_capital=initial_capital,
        allow_overlap=allow_overlap,
        max_concurrent_positions=max_concurrent_positions,
        max_leverage=max_leverage,
    )

    if min_bars > 0 and len(bars_20m) < min_bars:
        msg = f"insufficient history: fewer than {min_bars} bars are available ({len(bars_20m)} < {min_bars})"
        warnings.warn(msg)
        result['verdict'] = 'REFUSED'
        result['reasons'] = [f"insufficient history (< {min_bars} bars)"]
        if out_dir:
            save_report(
                metrics=result['metrics'],
                trades=result['trades'],
                out_dir=out_dir,
                initial_capital=initial_capital,
                verdict_override='REFUSED',
                reasons_override=result['reasons'],
            )
        return result

    if out_dir:
        save_report(
            metrics=result['metrics'],
            trades=result['trades'],
            out_dir=out_dir,
            initial_capital=initial_capital,
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V3+ walk-forward model out-of-sample.")
    parser.add_argument('--ticker', type=str, default='AAPL', help="Ticker symbol.")
    parser.add_argument('--cache-dir', type=str, default='.', help="Directory containing cached 20m bars.")
    parser.add_argument('--bars', type=str, default=None, help="Explicit path to ticker bars file.")
    parser.add_argument('--spy', type=str, default=None, help="Explicit path to SPY bars file.")
    parser.add_argument('--out', type=str, default='reports/v3_eval', help="Output directory for reports.")
    parser.add_argument('--folds', type=int, default=cct.N_CV_FOLDS, help="Number of walk-forward folds.")
    parser.add_argument('--cost-per-share', type=float, default=0.01, help="Commission per share.")
    parser.add_argument('--slippage-bps', type=float, default=2.0, help="Slippage in basis points.")
    parser.add_argument('--risk-per-trade', type=float, default=0.005, help="Risk fraction per trade.")
    parser.add_argument('--horizon-bars', type=int, default=12, help="Horizon bars.")
    parser.add_argument('--min-conf', type=float, default=cct.MIN_CONFIDENCE, help="Min confidence threshold.")
    parser.add_argument('--allow-overlap', action='store_true', default=False, help="Allow overlapping positions.")
    parser.add_argument('--max-concurrent-positions', type=int, default=5, help="Max concurrent positions.")
    parser.add_argument('--max-leverage', type=float, default=1.0, help="Max leverage cap.")

    args = parser.parse_args()
    cache_path = Path(args.cache_dir)
    ticker_clean = args.ticker.upper().strip()

    # Exact cache file selection: .cache_{TICKER}_v3_20m.pkl and .cache_SPY_v3_20m.pkl
    if args.bars:
        ticker_file = Path(args.bars)
    else:
        ticker_file = cache_path / f".cache_{ticker_clean}_v3_20m.pkl"

    if not ticker_file.exists():
        print(f"Error: Required ticker file not found: {ticker_file}", file=sys.stderr)
        sys.exit(1)

    if args.spy:
        spy_file = Path(args.spy)
    else:
        spy_file = cache_path / ".cache_SPY_v3_20m.pkl"

    if not spy_file.exists():
        print(f"Error: Required SPY file not found: {spy_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {ticker_clean} bars from {ticker_file}...")
    bars_df = joblib.load(ticker_file) if ticker_file.suffix == '.pkl' else pd.read_csv(ticker_file, index_col=0, parse_dates=True)

    print(f"Loading SPY bars from {spy_file}...")
    spy_df = joblib.load(spy_file) if spy_file.suffix == '.pkl' else pd.read_csv(spy_file, index_col=0, parse_dates=True)

    # Check for minimum 2,000 bars
    refuse_verdict = False
    if len(bars_df) < 2000:
        warning_msg = f"insufficient history: fewer than 2,000 bars are available ({len(bars_df)} < 2000)"
        warnings.warn(warning_msg)
        print(f"WARNING: {warning_msg}", file=sys.stderr)
        refuse_verdict = True

    print(f"\nRunning V3+ Out-of-Sample Walk-Forward Evaluation ({args.folds} folds)...")
    res = evaluate_v3(
        bars_20m=bars_df,
        spy_20m=spy_df,
        ticker=ticker_clean,
        out_dir=args.out,
        n_splits=args.folds,
        cost_per_share=args.cost_per_share,
        slippage_bps=args.slippage_bps,
        risk_per_trade=args.risk_per_trade,
        horizon_bars=args.horizon_bars,
        min_confidence=args.min_conf,
        allow_overlap=args.allow_overlap,
        max_concurrent_positions=args.max_concurrent_positions,
        max_leverage=args.max_leverage,
        min_bars=0,
    )

    if refuse_verdict:
        res['verdict'] = 'REFUSED'
        res['reasons'] = [f"insufficient history (< 2,000 bars: {len(bars_df)})"]
        if args.out:
            save_report(
                metrics=res['metrics'],
                trades=res['trades'],
                out_dir=args.out,
                verdict_override='REFUSED',
                reasons_override=res['reasons'],
            )

    v = res['verdict']
    m = res['metrics']
    print(f"\n{'=' * 50}")
    print(f"  V3+ OOS VERDICT: {v}")
    if res['reasons']:
        print(f"  Reasons: {', '.join(res['reasons'])}")
    print(f"{'=' * 50}")
    print(f"  Trades:          {m['n_trades']}")
    print(f"  Win Rate:        {m['win_rate']:.1%}")
    print(f"  Profit Factor:   {m['profit_factor']:.2f}")
    print(f"  Expectancy:      ${m['expectancy']:.2f} ({m['expectancy_r']:.3f} R)")
    print(f"  Net PnL:         ${m['net_pnl']:,.2f}")
    print(f"  Max Drawdown:    {m['max_drawdown_pct']:.2f}%")
    print(f"  Daily Sharpe:    {m['sharpe']:.2f}")
    print(f"\n  Report saved to: {args.out}/")


if __name__ == '__main__':
    main()
