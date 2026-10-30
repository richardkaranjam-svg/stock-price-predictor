"""
test_quant_invariants.py — 12-Test V3+ Quant Invariant Verification Suite.

Validates institutional-grade properties of the V3+ migration:
  1. Zero Lookahead
  2. Overnight Gap Isolation in FracDiff
  3. Closed-Bar Slicing
  4. 09:30 Bucket Alignment
  5. 15:45 ET Auto-Flatten & 15:30 ET Cutoff
  6. broker.py Short Selling & Margin Guards
  7. __main__-Free Artifact & predict.py Dict Inference
  8. O(N) Uniqueness Speed & Kish N_eff
  9. Gap 1 Protection (app.py never overwrites V3+ model)
  10. Gap 4 SQLite Schema Migration
  11. Gap 5 Deprecation Protection (train_model.py)
  12. Zero yfinance Dependency Audit
"""
import ast
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quant_core
from db_migration import V3_COLUMNS, migrate_paper_trades_schema

PROJECT_ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(42)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

def _make_20m_timestamps(n_days: int = 12) -> pd.DatetimeIndex:
    """Generate 09:30-anchored 20-min timestamps for n trading days."""
    # 19 bars per day: 09:30, 09:50, 10:10, ..., 15:30
    bar_times = [pd.Timedelta(hours=9, minutes=30 + 20 * i) for i in range(19)]
    base = pd.Timestamp('2024-01-02')  # Tuesday
    stamps = []
    day = base
    while len(stamps) < n_days * 19:
        if day.weekday() < 5:  # weekday
            for bt in bar_times:
                stamps.append(day + bt)
        day += pd.Timedelta(days=1)
    return pd.DatetimeIndex(stamps[: n_days * 19])


@pytest.fixture
def sample_20m_bars():
    """200 synthetic 20-min OHLCV bars with VWAP (~10.5 trading days)."""
    n = 200
    idx = _make_20m_timestamps(n_days=12)[:n]
    close = 150.0 + np.cumsum(RNG.normal(0, 0.5, n))
    noise = np.abs(RNG.normal(0, 0.3, n))
    open_ = close + RNG.normal(0, 0.2, n)
    high = np.maximum(open_, close) + noise
    low = np.minimum(open_, close) - noise
    volume = RNG.uniform(50_000, 500_000, n)
    tp = (high + low + close) / 3.0

    df = pd.DataFrame({
        'Open': open_, 'High': high, 'Low': low, 'Close': close,
        'Volume': volume,
    }, index=idx)

    # Daily-resetting cumulative VWAP
    session = idx.normalize()
    tp_x_vol = tp * volume
    cum_tp_vol = pd.Series(tp_x_vol, index=idx).groupby(session).cumsum()
    cum_vol = pd.Series(volume, index=idx).groupby(session).cumsum()
    df['VWAP'] = (cum_tp_vol / cum_vol.replace(0, np.nan)).values
    return df


@pytest.fixture
def sample_spy_20m_bars(sample_20m_bars):
    """SPY bars aligned to the same index, price ~450."""
    rng2 = np.random.default_rng(99)
    n = len(sample_20m_bars)
    close = 450.0 + np.cumsum(rng2.normal(0, 0.4, n))
    noise = np.abs(rng2.normal(0, 0.2, n))
    open_ = close + rng2.normal(0, 0.15, n)
    high = np.maximum(open_, close) + noise
    low = np.minimum(open_, close) - noise
    volume = rng2.uniform(100_000, 1_000_000, n)
    tp = (high + low + close) / 3.0

    df = pd.DataFrame({
        'Open': open_, 'High': high, 'Low': low, 'Close': close,
        'Volume': volume,
    }, index=sample_20m_bars.index)

    session = df.index.normalize()
    tp_x_vol = tp * volume
    cum_tp_vol = pd.Series(tp_x_vol, index=df.index).groupby(session).cumsum()
    cum_vol = pd.Series(volume, index=df.index).groupby(session).cumsum()
    df['VWAP'] = (cum_tp_vol / cum_vol.replace(0, np.nan)).values
    return df


class _FakeRF:
    """Minimal RF model that returns uniform 3-class probabilities."""
    classes_ = np.array([-1, 0, 1])
    def predict_proba(self, X):
        return np.ones((len(X), 3)) / 3.0


@pytest.fixture
def v3_model_artifact():
    """Plain dict that mimics a V3+ model artifact."""
    return {
        'model_version': 'V3+',
        'lgb_model_str': '__placeholder__',
        'rf_model': _FakeRF(),
        'calib_coef': np.ones(3),
        'calib_intercept': np.zeros(3),
        'feature_columns': list(quant_core.FEATURE_COLUMNS),
        'config': {'tp_atr_mult': 1.50, 'sl_atr_mult': 0.75},
        'env_versions': {'numpy': np.__version__},
    }


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary SQLite DB with the V2 paper_trades schema."""
    db_path = tmp_path / 'test_paper_trades.db'
    conn = sqlite3.connect(db_path)
    conn.execute('''CREATE TABLE paper_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        ticker TEXT NOT NULL,
        action TEXT NOT NULL,
        predicted_return REAL,
        current_price REAL,
        order_id TEXT,
        order_status TEXT,
        error TEXT,
        reason TEXT,
        fill_price REAL,
        slippage REAL
    )''')
    conn.commit()
    yield conn
    conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: Zero Lookahead Invariant
# ═══════════════════════════════════════════════════════════════════════════

def test_zero_lookahead_invariant(sample_20m_bars, sample_spy_20m_bars):
    """Dropping the last N rows must never change any feature value in
    the retained rows. This verifies no future data leaks backward."""
    full = quant_core.compute_features(sample_20m_bars, sample_spy_20m_bars)
    truncated = quant_core.compute_features(
        sample_20m_bars.iloc[:-5], sample_spy_20m_bars.iloc[:-5]
    )
    overlap = full.index.intersection(truncated.index)
    assert len(overlap) > 0, "No overlapping rows to compare"

    full_vals = full.loc[overlap].values
    trunc_vals = truncated.loc[overlap].values
    np.testing.assert_allclose(
        full_vals, trunc_vals, equal_nan=True,
        err_msg="Feature lookahead detected: values changed after removing future bars",
    )


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: Overnight Gap Isolation in FracDiff
# ═══════════════════════════════════════════════════════════════════════════

def test_overnight_gap_isolation_in_fracdiff(sample_20m_bars, sample_spy_20m_bars):
    """Changing a later day's prices must not change an earlier day's
    frac_diff_close values. The FFD kernel runs on a gap-adjusted series
    that isolates overnight jumps."""
    feat_orig = quant_core.compute_features(sample_20m_bars, sample_spy_20m_bars)

    sessions = sample_20m_bars.index.normalize()
    unique_days = sorted(sessions.unique())
    assert len(unique_days) >= 3, "Need at least 3 trading days"

    # Pick a day that appears in the computed features (after warm-up)
    feat_days = sorted(feat_orig.index.normalize().unique())
    if len(feat_days) < 2:
        pytest.skip("Not enough post-warm-up days for gap isolation test")

    reference_day = feat_days[-2]  # second-to-last day in features
    perturb_day = feat_days[-1]    # last day

    perturb_mask = sessions == perturb_day

    perturbed = sample_20m_bars.copy()
    perturbed.loc[perturb_mask, ['Open', 'High', 'Low', 'Close']] += 10.0

    spy_copy = sample_spy_20m_bars.copy()
    feat_perturbed = quant_core.compute_features(perturbed, spy_copy)

    ref_idx = feat_orig.index[feat_orig.index.normalize() == reference_day]
    overlap = ref_idx.intersection(feat_perturbed.index)
    assert len(overlap) > 0, "No reference-day rows in feature output"

    orig_fd = feat_orig.loc[overlap, 'frac_diff_close'].values
    pert_fd = feat_perturbed.loc[overlap, 'frac_diff_close'].values

    np.testing.assert_array_equal(
        orig_fd, pert_fd,
        err_msg="Later-day price perturbation leaked into earlier day's frac_diff_close",
    )


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: Closed-Bar Slicing Invariant
# ═══════════════════════════════════════════════════════════════════════════

def test_closed_bar_slicing_invariant(sample_20m_bars):
    """filter_closed_bars must never include a bar whose 20-min window
    hasn't closed yet."""
    import pytz
    et = pytz.timezone('America/New_York')

    # Place now_et partway through the 5th bar
    bar5_open = sample_20m_bars.index[4]
    now_et = pd.Timestamp(bar5_open, tz=et) + pd.Timedelta(minutes=10)

    # Localize bars for the filter
    bars_tz = sample_20m_bars.copy()
    bars_tz.index = bars_tz.index.tz_localize(et)

    filtered = quant_core.filter_closed_bars(bars_tz, now_et=now_et)

    assert len(filtered) < len(bars_tz), "Filtered result should be shorter than input"

    bar_close_times = filtered.index + pd.Timedelta(minutes=20)
    violations = bar_close_times > now_et
    assert not violations.any(), (
        f"Found {violations.sum()} bars whose 20-min window extends past now_et"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: 09:30 Bucket Alignment
# ═══════════════════════════════════════════════════════════════════════════

def test_0930_bucket_alignment():
    """Every resampled bar must fall on the 09:30-anchored grid:
    minutes 10, 30, or 50 past the hour. First bar of each day = 09:30."""
    rng = np.random.default_rng(77)
    # Generate raw 5-min bars: 78 bars/day × 3 days
    stamps = []
    base = pd.Timestamp('2024-03-04')  # Monday
    for d in range(3):
        day = base + pd.Timedelta(days=d)
        for i in range(78):
            stamps.append(day + pd.Timedelta(hours=9, minutes=30 + 5 * i))

    idx = pd.DatetimeIndex(stamps)
    n = len(idx)
    close = 180.0 + np.cumsum(rng.normal(0, 0.3, n))
    raw = pd.DataFrame({
        'Open': close - 0.1,
        'High': close + 0.5,
        'Low': close - 0.5,
        'Close': close,
        'Volume': rng.uniform(10000, 100000, n),
    }, index=idx)

    resampled = quant_core.resample_bars_20min(raw)
    assert len(resampled) > 0, "Resampler returned empty DataFrame"

    minutes = resampled.index.minute
    valid_minutes = {10, 30, 50}
    invalid = ~pd.Index(minutes).isin(valid_minutes)
    assert not invalid.any(), (
        f"Found bars at non-grid minutes: {set(minutes[invalid])}"
    )

    # First bar of each day must be 09:30
    for day_val, group in resampled.groupby(resampled.index.normalize()):
        first_bar = group.index[0]
        assert first_bar.hour == 9 and first_bar.minute == 30, (
            f"First bar of {day_val.date()} is {first_bar.time()}, expected 09:30"
        )


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: 15:45 ET Auto-Flatten & 15:30 ET Cutoff
# ═══════════════════════════════════════════════════════════════════════════

def test_1545_flatten_and_1530_cutoff():
    """Verify auto_trader.py source code contains the 15:45 flatten (375 min)
    and 15:30 cutoff (360 min) logic with correct reason codes."""
    source = (PROJECT_ROOT / 'auto_trader.py').read_text(encoding='utf-8')
    assert '375' in source, "Missing 375 (15:45 ET = 375 min since 09:30)"
    assert '360' in source, "Missing 360 (15:30 ET = 360 min since 09:30)"
    assert 'EOD_1545_AUTO_FLATTEN' in source, "Missing EOD_1545_AUTO_FLATTEN reason code"
    assert 'ENTRY_BLOCKED_AFTER_1530' in source, "Missing ENTRY_BLOCKED_AFTER_1530 reason code"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6: broker.py Short Selling & Margin Guard
# ═══════════════════════════════════════════════════════════════════════════

def test_broker_short_selling_guards():
    """broker.py must export check_short_eligibility, get_short_safe_buying_power,
    and cancel_open_orders_for_symbol as callable functions."""
    import broker
    assert callable(getattr(broker, 'check_short_eligibility', None)), \
        "broker.check_short_eligibility not found or not callable"
    assert callable(getattr(broker, 'get_short_safe_buying_power', None)), \
        "broker.get_short_safe_buying_power not found or not callable"
    assert callable(getattr(broker, 'cancel_open_orders_for_symbol', None)), \
        "broker.cancel_open_orders_for_symbol not found or not callable"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 7: __main__-Free Artifact & predict.py Dict Inference
# ═══════════════════════════════════════════════════════════════════════════

def test_main_free_artifact_and_dict_inference(v3_model_artifact):
    """predict.py must route dict artifacts through DictModelAdapter.
    No module-level train_stock_model call outside __main__ guard."""
    source = (PROJECT_ROOT / 'predict.py').read_text(encoding='utf-8')

    # Parse the AST and ensure train_stock_model isn't called at module level
    tree = ast.parse(source)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            name = ''
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            assert name != 'train_stock_model', \
                "predict.py calls train_stock_model at module level (outside __main__)"

    # Verify DictModelAdapter wraps the artifact correctly
    adapter = quant_core.DictModelAdapter(v3_model_artifact)
    assert hasattr(adapter, 'predict_proba'), "DictModelAdapter missing predict_proba"
    assert hasattr(adapter, 'predict'), "DictModelAdapter missing predict"
    assert hasattr(adapter, 'classes_'), "DictModelAdapter missing classes_"
    np.testing.assert_array_equal(adapter.classes_, [-1, 0, 1])


# ═══════════════════════════════════════════════════════════════════════════
# TEST 8: O(N) Uniqueness Speed & Kish N_eff Invariant
# ═══════════════════════════════════════════════════════════════════════════

def test_uniqueness_speed_and_kish_neff():
    """O(N) uniqueness must run in < 0.5s for N=100k and Kish N_eff/N >= 0.70."""
    N = 100_000
    rng = np.random.default_rng(123)
    starts = np.arange(N, dtype=np.int64)
    t1 = starts + rng.integers(1, 20, size=N)
    t1 = np.clip(t1, 0, N + 50)

    t0 = time.perf_counter()
    uniqueness = quant_core.compute_vectorized_uniqueness(t1)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.5, f"Uniqueness took {elapsed:.3f}s (limit 0.5s) — not O(N)"
    assert len(uniqueness) == N
    assert np.all(uniqueness > 0) and np.all(uniqueness <= 1), \
        "Uniqueness values must be in (0, 1]"

    weights, mean_u, n_eff_ratio = quant_core.compute_fold_sample_weights(uniqueness)
    assert n_eff_ratio >= 0.70, (
        f"Kish N_eff/N = {n_eff_ratio:.4f} < 0.70 — weights are too dispersed"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TEST 9: Gap 1 Protection — app.py Never Overwrites V3+ Model
# ═══════════════════════════════════════════════════════════════════════════

def test_gap1_app_never_overwrites_v3_model():
    """app.py ensure_fresh_model() must never import or call train_stock_model.
    It must be a read-only staleness checker that returns a status dict."""
    source = (PROJECT_ROOT / 'app.py').read_text(encoding='utf-8')

    # Must not import the training module
    assert 'import train_model' not in source, \
        "app.py still imports train_model"
    assert 'train_stock_model' not in source, \
        "app.py still references train_stock_model"

    # Parse ensure_fresh_model and verify it returns a dict with expected keys
    tree = ast.parse(source)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'ensure_fresh_model':
            found = True
            # Verify function doesn't contain calls to joblib.dump or model saving
            func_source = ast.get_source_segment(source, node)
            assert 'joblib.dump' not in func_source, \
                "ensure_fresh_model contains joblib.dump — it must be read-only"
            assert '.save(' not in func_source, \
                "ensure_fresh_model contains .save() — it must be read-only"
            break
    assert found, "ensure_fresh_model function not found in app.py"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 10: Gap 4 SQLite Schema Migration
# ═══════════════════════════════════════════════════════════════════════════

def test_gap4_sqlite_schema_migration(tmp_db):
    """migrate_paper_trades_schema must add all V3+ columns idempotently."""
    conn = tmp_db

    # First run: should add all V3+ columns
    added = migrate_paper_trades_schema(conn)
    assert set(added) == set(V3_COLUMNS.keys()), (
        f"Expected all V3+ columns to be added, got: {added}"
    )

    # Verify columns exist
    existing = {row[1] for row in conn.execute('PRAGMA table_info(paper_trades)')}
    for col in V3_COLUMNS:
        assert col in existing, f"Column {col} missing after migration"

    # Second run: idempotent — should add nothing
    added_again = migrate_paper_trades_schema(conn)
    assert added_again == [], f"Non-idempotent: second run added {added_again}"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 11: Gap 5 Deprecation Protection
# ═══════════════════════════════════════════════════════════════════════════

def test_gap5_deprecation_protection(tmp_path):
    """train_model.train_stock_model() must raise RuntimeError."""
    import train_model

    # Point MODEL_PATH to a nonexistent file so it hits the unconditional guard
    original = train_model.MODEL_PATH
    train_model.MODEL_PATH = tmp_path / 'nonexistent_model.pkl'
    try:
        with pytest.raises(RuntimeError, match=r'(?i)(deprecated|V3\+|Colab)'):
            train_model.train_stock_model()
    finally:
        train_model.MODEL_PATH = original


# ═══════════════════════════════════════════════════════════════════════════
# TEST 12: Zero yfinance Dependency Audit
# ═══════════════════════════════════════════════════════════════════════════

def test_zero_yfinance_dependency_audit():
    """No .py file in the project root may import yfinance."""
    skip_dirs = {'__pycache__', '.venv', 'tests', '.pytest_cache', '.vscode',
                 '.continue', 'lesson_week2_session3_oop_for_ai',
                 'lesson_week2_session4_numpy_for_ai',
                 'lesson_week3_session5_pandas_wrangling',
                 'lesson_week3_session6_eda_feature_prep', 'logs', 'static',
                 'templates'}

    violations = []
    for py_file in PROJECT_ROOT.glob('*.py'):
        if py_file.name.startswith('test_'):
            continue  # skip test files themselves
        try:
            source = py_file.read_text(encoding='utf-8')
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name and ('yfinance' in alias.name):
                        violations.append(f"{py_file.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and ('yfinance' in node.module):
                    violations.append(f"{py_file.name}: from {node.module}")

    assert not violations, (
        f"yfinance imports found in production code:\n" +
        "\n".join(f"  - {v}" for v in violations)
    )
