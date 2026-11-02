"""
tests/test_version_and_signals.py — Unit tests for:
1. Model version check and ALLOW_VERSION_MISMATCH bypass.
2. Signal logging in predict_v3 and auto_trader.PaperTrader.
3. Outcome filling after horizon_bars have elapsed.

Offline only: uses temp DB and synthetic data, zero network.
"""
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import auto_trader
import db_migration
import predict
import quant_core
from db_migration import (
    SIGNALS_COLUMNS,
    fill_signal_outcomes,
    log_signal,
    migrate_signals_schema,
)
from predict import (
    check_library_versions,
    compute_features_hash,
    predict_v3,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. VERSION CHECK TESTS
# ═══════════════════════════════════════════════════════════════════════════

def test_version_check_matching_versions():
    """Matching major and minor versions pass without error."""
    import sklearn
    import lightgbm as lgb
    env_versions = {
        'scikit-learn': sklearn.__version__,
        'lightgbm': lgb.__version__,
        'numpy': np.__version__,
    }
    # Should not raise
    check_library_versions(env_versions)


def test_version_check_patch_difference_passes():
    """Different patch version (same major and minor) passes."""
    import sklearn
    import lightgbm as lgb
    from packaging.version import Version
    v_sk = Version(sklearn.__version__)
    v_lgb = Version(lgb.__version__)

    env_versions = {
        'scikit-learn': f"{v_sk.major}.{v_sk.minor}.999",
        'lightgbm': f"{v_lgb.major}.{v_lgb.minor}.999",
    }
    # Should not raise
    check_library_versions(env_versions)


def test_version_check_sklearn_minor_mismatch_fails():
    """Minor version mismatch in scikit-learn raises RuntimeError."""
    import lightgbm as lgb
    env_versions = {
        'scikit-learn': '1.6.1',  # Local is 1.9.0
        'lightgbm': lgb.__version__,
    }
    with pytest.raises(RuntimeError, match=r"Version mismatch for scikit-learn"):
        check_library_versions(env_versions)


def test_version_check_lightgbm_minor_mismatch_fails():
    """Minor version mismatch in lightgbm raises RuntimeError."""
    import sklearn
    env_versions = {
        'scikit-learn': sklearn.__version__,
        'lightgbm': '4.6.0',  # Local is 4.7.0
    }
    with pytest.raises(RuntimeError, match=r"Version mismatch for lightgbm"):
        check_library_versions(env_versions)


def test_version_check_major_mismatch_fails():
    """Major version mismatch raises RuntimeError."""
    import lightgbm as lgb
    env_versions = {
        'scikit-learn': '2.0.0',
        'lightgbm': lgb.__version__,
    }
    with pytest.raises(RuntimeError, match=r"Version mismatch for scikit-learn"):
        check_library_versions(env_versions)


def test_version_check_allow_mismatch_env_var(monkeypatch):
    """Setting ALLOW_VERSION_MISMATCH=1 bypasses version check failure."""
    monkeypatch.setenv('ALLOW_VERSION_MISMATCH', '1')
    env_versions = {
        'scikit-learn': '1.6.1',
        'lightgbm': '4.6.0',
    }
    # Should not raise because bypass is active
    check_library_versions(env_versions)


def test_predict_load_model_triggers_version_check(tmp_path, monkeypatch):
    """_load_model raises RuntimeError on version mismatch unless allowed."""
    fake_artifact = {
        'model_version': 'V3+',
        'lgb_model_str': '',
        'rf_model': None,
        'calib_coef': [1.0, 1.0, 1.0],
        'calib_intercept': [0.0, 0.0, 0.0],
        'feature_columns': [],
        'env_versions': {
            'scikit-learn': '1.6.1',
            'lightgbm': '4.6.0',
        },
    }
    art_path = tmp_path / 'stock_price_model.pkl'
    joblib.dump(fake_artifact, art_path)

    monkeypatch.setattr(predict, 'MODEL_PATH', art_path)
    monkeypatch.setattr(predict, '_model_cache', {})
    monkeypatch.delenv('ALLOW_VERSION_MISMATCH', raising=False)

    with pytest.raises(RuntimeError, match=r"Version mismatch for scikit-learn"):
        predict._load_model()

    # With ALLOW_VERSION_MISMATCH=1, it succeeds
    monkeypatch.setenv('ALLOW_VERSION_MISMATCH', '1')
    predict._model_cache.clear()
    adapter = predict._load_model()
    assert isinstance(adapter, quant_core.DictModelAdapter)


# ═══════════════════════════════════════════════════════════════════════════
# 2. SIGNAL LOGGING & MIGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

def test_migrate_signals_schema(tmp_path):
    """migrate_signals_schema creates the signals table with exact columns idempotently."""
    db_path = tmp_path / 'test.db'
    with sqlite3.connect(db_path) as conn:
        migrate_signals_schema(conn)

        columns = {row[1]: row[2] for row in conn.execute('PRAGMA table_info(signals)')}
        expected_cols = {
            'id': 'INTEGER',
            'timestamp_utc': 'TEXT',
            'ticker': 'TEXT',
            'model_version': 'TEXT',
            'side': 'TEXT',
            'confidence': 'REAL',
            'price': 'REAL',
            'features_hash': 'TEXT',
            'outcome': 'REAL',
        }
        for col, col_type in expected_cols.items():
            assert col in columns, f"Missing column {col} in signals table"
            assert columns[col] == col_type, f"Column {col} has wrong type {columns[col]}"

        # Second call must be idempotent
        added = migrate_signals_schema(conn)
        assert added == []


def test_log_signal_appends_row(tmp_path):
    """log_signal appends a row with exact values to signals table."""
    db_path = tmp_path / 'test.db'
    ts = datetime.now(timezone.utc).isoformat()
    row_id = log_signal(
        conn_or_path=db_path,
        timestamp_utc=ts,
        ticker='AAPL',
        model_version='V3+',
        side='long',
        confidence=0.85,
        price=175.50,
        features_hash='abc123def456',
        outcome=None,
    )
    assert row_id == 1

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute('SELECT * FROM signals').fetchall()
        assert len(rows) == 1
        r = rows[0]
        # id, timestamp_utc, ticker, model_version, side, confidence, price, features_hash, outcome
        assert r[0] == 1
        assert r[1] == ts
        assert r[2] == 'AAPL'
        assert r[3] == 'V3+'
        assert r[4] == 'long'
        assert r[5] == pytest.approx(0.85)
        assert r[6] == pytest.approx(175.50)
        assert r[7] == 'abc123def456'
        assert r[8] is None


def test_features_hash_determinism():
    """compute_features_hash is deterministic across column permutations and changes on value diff."""
    df1 = pd.DataFrame({'b': [2.0], 'a': [1.0]})
    df2 = pd.DataFrame({'a': [1.0], 'b': [2.0]})
    df3 = pd.DataFrame({'a': [1.0], 'b': [2.1]})

    hash1 = compute_features_hash(df1)
    hash2 = compute_features_hash(df2)
    hash3 = compute_features_hash(df3)

    assert hash1 == hash2, "compute_features_hash must be independent of column order"
    assert hash1 != hash3, "compute_features_hash must change when feature values change"
    assert len(hash1) == 64


def test_predict_v3_logs_to_signals(tmp_path, monkeypatch):
    """predict_v3 automatically logs every prediction to the signals table."""
    db_path = tmp_path / 'predict_test.db'
    monkeypatch.setattr(predict, 'DB_PATH', db_path)

    # Fake market data
    from tests.test_pipeline import sample_market_data
    bars = sample_market_data(rows=200)

    class FakeAdapter(quant_core.DictModelAdapter):
        def __init__(self):
            self.classes_ = np.array([-1, 0, 1])
        def predict_proba(self, values):
            return np.array([[0.10, 0.15, 0.75]])

    monkeypatch.setattr(predict, '_load_model', lambda: FakeAdapter())
    monkeypatch.setattr(predict, 'get_latest_stock_data', lambda t: bars)
    monkeypatch.setattr(predict, 'download_spy_intraday', lambda: bars)
    monkeypatch.setattr(predict, 'get_news_summary', lambda t: {'sentiment': 0.0})
    monkeypatch.setattr(predict, '_load_metadata', lambda: {'model_version': 'V3+', 'tp_atr_mult': 1.5, 'sl_atr_mult': 0.75})

    result = predict_v3('AAPL')
    assert result['predicted_class'] == 1
    assert result['side'] == 'long'
    assert 'features_hash' in result
    assert len(result['features_hash']) == 64

    # Check signal table
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            'SELECT ticker, model_version, side, confidence, price, features_hash, outcome FROM signals'
        ).fetchall()
        assert len(rows) == 1
        ticker, mv, side, conf, price, f_hash, outcome = rows[0]
        assert ticker == 'AAPL'
        assert mv == 'V3+'
        assert side == 'long'
        assert conf == pytest.approx(0.75)
        assert price == pytest.approx(result['current_price'])
        assert f_hash == result['features_hash']
        assert outcome is None


def test_paper_trader_decision_logging(tmp_path, monkeypatch):
    """PaperTrader._record logs all decisions to the signals table."""
    db_path = tmp_path / 'trader_test.db'
    monkeypatch.setattr(auto_trader, 'DB_PATH', db_path)
    # Don't touch the real stock_price_model.pkl during construction
    monkeypatch.setattr(auto_trader, 'MODEL_PATH', tmp_path / 'missing_model.pkl')

    trader = auto_trader.PaperTrader()
    # Log a buy decision
    trader._record(
        'AAPL', 'buy', current_price=150.0, confidence=0.80,
        features_hash='decision_hash_1',
    )
    # Log a hold decision
    trader._record(
        'AAPL', 'hold', current_price=151.0, confidence=0.45,
        features_hash='decision_hash_2',
    )
    # Log a sell decision
    trader._record(
        'AAPL', 'sell', current_price=152.0, confidence=0.88,
        features_hash='decision_hash_3',
    )

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            'SELECT ticker, side, price, confidence, features_hash, outcome FROM signals ORDER BY id ASC'
        ).fetchall()
        assert len(rows) == 3
        # 1. Buy decision
        assert rows[0][0] == 'AAPL'
        assert rows[0][1] == 'long'
        assert rows[0][2] == pytest.approx(150.0)
        assert rows[0][3] == pytest.approx(0.80)
        assert rows[0][4] == 'decision_hash_1'
        assert rows[0][5] is None

        # 2. Hold decision
        assert rows[1][1] == 'flat'
        assert rows[1][2] == pytest.approx(151.0)
        assert rows[1][3] == pytest.approx(0.45)
        assert rows[1][4] == 'decision_hash_2'

        # 3. Sell decision
        assert rows[2][1] == 'short'
        assert rows[2][2] == pytest.approx(152.0)
        assert rows[2][3] == pytest.approx(0.88)
        assert rows[2][4] == 'decision_hash_3'

        sources = {r[0] for r in conn.execute('SELECT source FROM signals')}
        assert sources == {'trader'}


def test_paper_trader_classifies_exits_and_non_decisions(tmp_path, monkeypatch):
    """Exits are 'exit', non-decisions are 'none'; raw action is preserved."""
    db_path = tmp_path / 'trader_side_test.db'
    monkeypatch.setattr(auto_trader, 'DB_PATH', db_path)
    monkeypatch.setattr(auto_trader, 'MODEL_PATH', tmp_path / 'missing_model.pkl')
    trader = auto_trader.PaperTrader()

    trader._record('AAPL', 'sell', current_price=150.0, reason='DYNAMIC_ATR_TAKE_PROFIT')
    trader._record('AAPL', 'buy', current_price=150.0, reason='DYNAMIC_ATR_STOP_LOSS')
    trader._record('AAPL', 'sell', current_price=150.0, reason='close_before_short')
    trader._record('AAPL', 'eod_flatten_sell', current_price=150.0, reason='EOD_1545_AUTO_FLATTEN')
    trader._record('AAPL', 'market_closed', reason='US market is closed')
    trader._record('AAPL', 'daily_kill_switch', reason='daily loss limit')
    trader._record('AAPL', 'ratchet', current_price=150.0, reason='RATCHET_1R_BREAKEVEN')
    trader._record('AAPL', 'buy', current_price=150.0, error='broker down', reason='go long')
    trader._record('AAPL', 'sell', current_price=150.0, reason='go short (conf=60%, EV=0.1)')

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute('SELECT action, side, source FROM signals ORDER BY id').fetchall()

    assert [r[1] for r in rows] == [
        'exit', 'exit', 'exit', 'exit', 'none', 'none', 'none', 'none', 'short',
    ]
    assert rows[3][0] == 'eod_flatten_sell'
    assert all(r[2] == 'trader' for r in rows)


def test_predict_v3_tags_source_predict(tmp_path, monkeypatch):
    """predict_v3 rows are tagged source='predict' so they can be separated from trader rows."""
    db_path = tmp_path / 'predict_source.db'
    monkeypatch.setattr(predict, 'DB_PATH', db_path)
    from tests.test_pipeline import sample_market_data
    bars = sample_market_data(rows=200)

    class FakeAdapter(quant_core.DictModelAdapter):
        def __init__(self):
            self.classes_ = np.array([-1, 0, 1])
        def predict_proba(self, values):
            return np.array([[0.70, 0.20, 0.10]])

    monkeypatch.setattr(predict, '_load_model', lambda: FakeAdapter())
    monkeypatch.setattr(predict, 'get_latest_stock_data', lambda t: bars)
    monkeypatch.setattr(predict, 'download_spy_intraday', lambda: bars)
    monkeypatch.setattr(predict, 'get_news_summary', lambda t: {'sentiment': 0.0})
    monkeypatch.setattr(predict, '_load_metadata', lambda: {'model_version': 'V3+'})

    predict_v3('AAPL')
    with sqlite3.connect(db_path) as conn:
        assert conn.execute('SELECT source, side, action FROM signals').fetchall() == [
            ('predict', 'short', None)
        ]


def test_migrate_signals_schema_upgrades_old_table(tmp_path):
    """A signals table created without source/action gets those columns added."""
    db_path = tmp_path / 'old.db'
    with sqlite3.connect(db_path) as conn:
        conn.execute('''CREATE TABLE signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp_utc TEXT NOT NULL,
            ticker TEXT NOT NULL, model_version TEXT, side TEXT, confidence REAL,
            price REAL, features_hash TEXT, outcome REAL)''')
        added = migrate_signals_schema(conn)
        assert set(added) == {'source', 'action'}
        assert migrate_signals_schema(conn) == []


def test_paper_trader_startup_version_check_logs_but_does_not_raise(tmp_path, monkeypatch, caplog):
    """Startup check uses the MAJOR.MINOR rule, logs an error, and never crashes the trader."""
    art_path = tmp_path / 'stock_price_model.pkl'
    joblib.dump({'env_versions': {'scikit-learn': '0.1.0', 'lightgbm': '0.1.0'}}, art_path)
    monkeypatch.setattr(auto_trader, 'DB_PATH', tmp_path / 'trader.db')
    monkeypatch.setattr(auto_trader, 'MODEL_PATH', art_path)
    monkeypatch.delenv('ALLOW_VERSION_MISMATCH', raising=False)

    with caplog.at_level('ERROR', logger='auto_trader'):
        auto_trader.PaperTrader()
    assert any('Version mismatch' in rec.getMessage() for rec in caplog.records)


def test_config_market_status_is_defined_once_and_timezone_fallback_works(monkeypatch):
    """Regression: duplicate get_market_status removed; get_et_now fallback imports timezone."""
    import inspect
    import config
    source = inspect.getsource(config)
    assert source.count('def get_market_status') == 1
    assert 'ΓÇ' not in source

    import builtins
    real_import = builtins.__import__

    def no_tz_libs(name, *args, **kwargs):
        if name in ('zoneinfo', 'pytz'):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_tz_libs)
    now = config.get_et_now()
    assert now.utcoffset() == timedelta(hours=-5)


# ═══════════════════════════════════════════════════════════════════════════
# 3. OUTCOME FILLING TESTS
# ═══════════════════════════════════════════════════════════════════════════

def test_fill_signal_outcomes_after_horizon_bars(tmp_path):
    """fill_signal_outcomes fills outcome only when >= horizon_bars have elapsed."""
    db_path = tmp_path / 'outcome_test.db'
    base_time = datetime(2026, 3, 16, 14, 0, tzinfo=timezone.utc)

    # Log two signals at base_time: one long, one short
    log_signal(
        conn_or_path=db_path,
        timestamp_utc=base_time.isoformat(),
        ticker='AAPL',
        side='long',
        confidence=0.70,
        price=100.0,
        features_hash='hash_long',
    )
    log_signal(
        conn_or_path=db_path,
        timestamp_utc=base_time.isoformat(),
        ticker='AAPL',
        side='short',
        confidence=0.70,
        price=100.0,
        features_hash='hash_short',
    )

    # Build 25 synthetic 20-min bars starting from base_time
    bar_times = [base_time + timedelta(minutes=20 * i) for i in range(25)]
    # Set prices: starts at 100, bar 12 is 105.0
    prices = [100.0 + i * 0.5 for i in range(25)]
    prices[12] = 105.0  # Exactly 105.0 at horizon_bars=12

    bars = pd.DataFrame({'Close': prices, 'Open': prices, 'High': prices, 'Low': prices}, index=pd.DatetimeIndex(bar_times))

    # Test 1: Fewer than horizon_bars available (only 8 bars provided)
    short_bars = bars.iloc[:8]
    updated = fill_signal_outcomes(conn_or_path=db_path, bars=short_bars, horizon_bars=12)
    assert updated == 0

    with sqlite3.connect(db_path) as conn:
        pending = conn.execute('SELECT outcome FROM signals WHERE outcome IS NULL').fetchall()
        assert len(pending) == 2

    # Test 2: Full 25 bars provided (horizon_bars=12 is reachable)
    updated = fill_signal_outcomes(conn_or_path=db_path, bars=bars, horizon_bars=12)
    assert updated == 2

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute('SELECT side, outcome FROM signals ORDER BY id ASC').fetchall()
        assert len(rows) == 2
        # Long outcome: (105.0 - 100.0) / 100.0 = +0.05 (+5%)
        assert rows[0][0] == 'long'
        assert rows[0][1] == pytest.approx(0.05)

        # Short outcome: (100.0 - 105.0) / 100.0 = -0.05 (-5%)
        assert rows[1][0] == 'short'
        assert rows[1][1] == pytest.approx(-0.05)

    # Calling fill_signal_outcomes again updates 0 rows (all already filled)
    updated_again = fill_signal_outcomes(conn_or_path=db_path, bars=bars, horizon_bars=12)
    assert updated_again == 0


def test_fill_signal_outcomes_skips_exits_non_decisions_and_missing_price(tmp_path):
    """Only long/short/flat rows with a price are scored; flat gets the raw asset return."""
    db_path = tmp_path / 'skip_test.db'
    base_time = datetime(2026, 3, 16, 14, 0, tzinfo=timezone.utc)
    for side, price in [('exit', 100.0), ('none', 100.0), ('long', None), ('flat', 100.0)]:
        log_signal(conn_or_path=db_path, timestamp_utc=base_time.isoformat(),
                   ticker='AAPL', side=side, price=price)

    bar_times = [base_time + timedelta(minutes=20 * i) for i in range(5)]
    bars = {'AAPL': pd.DataFrame({'Close': [100.0, 101.0, 102.0, 103.0, 104.0]},
                                 index=pd.DatetimeIndex(bar_times))}
    assert fill_signal_outcomes(conn_or_path=db_path, bars=bars, horizon_bars=2) == 1

    with sqlite3.connect(db_path) as conn:
        rows = dict(conn.execute('SELECT side, outcome FROM signals').fetchall())
    assert rows['flat'] == pytest.approx(0.02)
    assert rows['exit'] is None and rows['none'] is None and rows['long'] is None


def test_paper_signals_api_endpoint():
    """Test /paper/signals endpoint in app.py using Flask test client."""
    from app import app
    with app.test_client() as client:
        res = client.get('/paper/signals')
        assert res.status_code == 200
        data = res.get_json()
        assert 'signals' in data
        assert 'count' in data
        assert isinstance(data['signals'], list)


def test_webhook_trade_filter_api_endpoint(monkeypatch, tmp_path):
    """Test /api/v3/filter webhook endpoint in app.py with mock predict_v3."""
    from app import app
    import app as app_module

    fake_result = {
        'predicted_class': 1,
        'confidence': 0.78,
        'current_price': 150.0,
        'predicted_price': 152.0,
        'predicted_return': 0.013,
        'take_profit_price': 153.0,
        'stop_loss_price': 148.5,
        'features_hash': 'f' * 64,
        'model_version': 'v3-test',
    }
    monkeypatch.setattr(app_module, 'predict_v3', lambda ticker: fake_result)

    with app.test_client() as client:
        # Case 1: Direction matches and confidence >= min_confidence -> EXECUTE
        res = client.post('/api/v3/filter', json={
            'ticker': 'AAPL',
            'action': 'BUY',
            'min_confidence': 0.65,
            'source': 'tradingview_alert'
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data['decision'] == 'EXECUTE'
        assert data['model_confidence'] == 0.78
        assert data['signal_side'] == 'LONG'

        # Case 2: Direction conflict -> REJECT
        res_conflict = client.post('/api/v3/filter', json={
            'ticker': 'AAPL',
            'action': 'SHORT',
            'min_confidence': 0.65,
        })
        assert res_conflict.status_code == 200
        assert res_conflict.get_json()['decision'] == 'REJECT'

        # Case 3: Confidence below hurdle -> REJECT
        res_low_conf = client.post('/api/v3/filter', json={
            'ticker': 'AAPL',
            'action': 'BUY',
            'min_confidence': 0.85,
        })
        assert res_low_conf.status_code == 200
        assert res_low_conf.get_json()['decision'] == 'REJECT'


