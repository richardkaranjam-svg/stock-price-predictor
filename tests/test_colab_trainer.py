"""
test_colab_trainer.py — Offline regression tests for colab_cloud_trainer.py.

Covers the Colab failure modes found in the audit:
  - Alpaca error classification (str(APIError) has NO status code; 403 = bad keys)
  - Retry/backoff only for 429 / 5xx / network errors
  - Credential sanitisation & os.environ write-back
  - SIP→IEX fallback, IEX history fast-forward, no feed splicing, no rewind
  - Lost-chunk accounting & abort threshold
  - Purged CV guards, class-aware probability alignment, Kish N_eff guard
  - End-to-end: train_v3_model → joblib artifact → DictModelAdapter → predict_v3
"""
import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import requests
from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed

import colab_cloud_trainer as cct
import quant_core

UTC = timezone.utc
SIP_FIRST = datetime(2016, 1, 4, 14, 30, tzinfo=UTC)
IEX_FIRST = datetime(2020, 8, 3, 13, 30, tzinfo=UTC)
START_RECENT = datetime(2025, 1, 1, tzinfo=UTC)
SUB_403_BODY = '{"message": "subscription does not permit querying recent SIP data"}'


def _api_error(status: int, body: str) -> APIError:
    """Build an APIError exactly the way alpaca-py's RESTClient does."""
    resp = requests.models.Response()
    resp.status_code = status
    return APIError(body, requests.HTTPError(response=resp))


def _bar(ts, price=100.0):
    return SimpleNamespace(timestamp=ts, open=price, high=price + 0.5,
                           low=price - 0.5, close=price, volume=1000.0)


class FakeClient:
    """Fake StockHistoricalDataClient.

    first_bar: feed value → earliest available UTC datetime.
    errors:    callable(request) -> Exception | None (raised instead of data).
    """

    def __init__(self, first_bar, errors=None):
        self.first_bar = first_bar
        self.errors = errors or (lambda req: None)
        self.calls = []

    @staticmethod
    def _aware(dt):
        # alpaca-py's request validator converts tz-aware datetimes to NAIVE UTC.
        if dt is None:
            return None
        return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)

    def get_stock_bars(self, raw_req):
        req = SimpleNamespace(symbol_or_symbols=raw_req.symbol_or_symbols, feed=raw_req.feed,
                              limit=raw_req.limit, start=self._aware(raw_req.start),
                              end=self._aware(raw_req.end))
        self.calls.append(req)
        err = self.errors(req)
        if err is not None:
            raise err
        sym = req.symbol_or_symbols
        first = self.first_bar.get(req.feed.value)
        end = req.end or datetime.now(UTC)
        start = max(req.start, first) if first else None
        if start is None or start >= end:
            return SimpleNamespace(data={})
        if req.limit == 1:  # daily first-bar discovery
            return SimpleNamespace(data={sym: [_bar(start)]})
        # One bar per chunk: enough to verify ordering / feed logic.
        return SimpleNamespace(data={sym: [_bar(start + pd.Timedelta(hours=1))]})


def _bar_requests(client):
    return [c for c in client.calls if c.limit != 1]


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    """No real sleeping and no forced feed during tests."""
    monkeypatch.setattr(cct, 'RETRY_BASE_DELAY_S', 0.0)
    monkeypatch.setattr(cct.time, 'sleep', lambda s: None)
    monkeypatch.delenv('ALPACA_DATA_FEED', raising=False)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Error classification & retry
# ═══════════════════════════════════════════════════════════════════════════

def test_apierror_str_has_no_status_code():
    """Root cause: substring checks on str(e) can never see '401'/'403'."""
    e = _api_error(401, '{"message": "unauthorized."}')
    assert '401' not in str(e)
    assert cct._api_status_code(e) == 401


@pytest.mark.parametrize('status, body, expected', [
    (401, '{"message": "unauthorized."}', 'auth'),
    (403, '{"message": "forbidden."}', 'auth'),  # Alpaca: bad keys → 403
    (403, SUB_403_BODY, 'subscription'),
    (429, '{"message": "too many requests."}', 'rate_limit'),
    (500, 'internal error', 'transient'),
    (504, 'gateway timeout', 'transient'),
    (422, '{"message": "invalid start"}', 'other'),
])
def test_classify_alpaca_error(status, body, expected):
    assert cct._classify_alpaca_error(_api_error(status, body)) == expected


def test_classify_network_errors_are_transient():
    assert cct._classify_alpaca_error(requests.exceptions.ConnectionError('reset')) == 'transient'
    assert cct._classify_alpaca_error(TimeoutError()) == 'transient'


def test_retry_recovers_from_transient_and_429():
    seq = [_api_error(429, 'slow down'), _api_error(503, 'down'), 'ok']

    def fn():
        item = seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    assert cct._call_with_retry(fn, 'test') == 'ok'


def test_retry_never_retries_auth():
    calls = []

    def fn():
        calls.append(1)
        raise _api_error(403, '{"message": "forbidden."}')

    with pytest.raises(APIError):
        cct._call_with_retry(fn, 'test')
    assert len(calls) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 2. Credentials
# ═══════════════════════════════════════════════════════════════════════════

def test_credentials_sanitised_and_written_back(monkeypatch):
    monkeypatch.setenv('APCA_API_KEY_ID', ' "PKTEST123456"\u200b\n')
    monkeypatch.setenv('APCA_API_SECRET_KEY', "\ufeff'secretvalue' ")
    key, secret = cct._resolve_alpaca_credentials()
    assert (key, secret) == ('PKTEST123456', 'secretvalue')
    assert os.environ['APCA_API_KEY_ID'] == 'PKTEST123456'  # subprocess inheritance
    assert os.environ['APCA_API_SECRET_KEY'] == 'secretvalue'


def test_colab_secrets_override_stale_environ_keys(monkeypatch, capsys):
    """Regression: stale os.environ keys (old cell / earlier write-back) must
    NOT shadow freshly regenerated Colab Secrets."""
    import sys
    monkeypatch.setenv('APCA_API_KEY_ID', 'PKOLDSTALE0001')
    monkeypatch.setenv('APCA_API_SECRET_KEY', 'oldsecret')
    fake_secrets = {'APCA_API_KEY_ID': 'PKNEWFRESH0002', 'APCA_API_SECRET_KEY': 'newsecret'}
    userdata = SimpleNamespace(get=lambda name: fake_secrets[name])
    google = SimpleNamespace(colab=SimpleNamespace(userdata=userdata))
    monkeypatch.setitem(sys.modules, 'google', google)
    monkeypatch.setitem(sys.modules, 'google.colab', google.colab)
    monkeypatch.setitem(sys.modules, 'google.colab.userdata', userdata)

    key, secret = cct._resolve_alpaca_credentials()
    assert (key, secret) == ('PKNEWFRESH0002', 'newsecret')
    assert os.environ['APCA_API_KEY_ID'] == 'PKNEWFRESH0002'  # stale value replaced
    assert 'stale' in capsys.readouterr().out


def test_missing_credentials_raise_actionable_error(monkeypatch):
    monkeypatch.delenv('APCA_API_KEY_ID', raising=False)
    monkeypatch.delenv('APCA_API_SECRET_KEY', raising=False)
    with pytest.raises(RuntimeError, match='Colab Secrets'):
        cct._resolve_alpaca_credentials()


# ═══════════════════════════════════════════════════════════════════════════
# 3. Probe & streaming
# ═══════════════════════════════════════════════════════════════════════════

def test_probe_bad_keys_403_fails_fast():
    client = FakeClient({}, errors=lambda r: _api_error(403, '{"message": "forbidden."}'))
    with pytest.raises(RuntimeError, match='AUTHENTICATION FAILED'):
        cct._probe_alpaca_credentials(client, 'PKBADKEY0000', 'secret000000')
    assert len(client.calls) == 1  # no retries, no SIP probe, no monthly loop


def test_probe_sip_subscription_403_falls_back_to_iex():
    sub = _api_error(403, SUB_403_BODY)
    client = FakeClient({'iex': IEX_FIRST},
                        errors=lambda r: sub if r.feed == DataFeed.SIP else None)
    assert cct._probe_alpaca_credentials(client, 'PKKEY0000000', 'secret000000') == 'iex'


def test_probe_sip_ok_returns_sip():
    client = FakeClient({'sip': SIP_FIRST, 'iex': IEX_FIRST})
    assert cct._probe_alpaca_credentials(client, 'PKKEY0000000', 'secret000000') == 'sip'


def test_probe_forced_feed_env(monkeypatch):
    monkeypatch.setenv('ALPACA_DATA_FEED', 'iex')
    client = FakeClient({'sip': SIP_FIRST, 'iex': IEX_FIRST})
    assert cct._probe_alpaca_credentials(client, 'PKKEY0000000', 'secret000000') == 'iex'


def test_iex_stream_fast_forwards_past_empty_history():
    """START_YEAR=2016 on IEX must skip the empty pre-history, not raise."""
    client = FakeClient({'iex': IEX_FIRST})
    df = cct.stream_alpaca_bars('AAPL', 2016, client=client, feed='iex')
    reqs = _bar_requests(client)
    assert reqs[0].start >= datetime(2020, 8, 3, tzinfo=UTC)
    assert df.index[0] >= pd.Timestamp('2020-08-03', tz='UTC')
    assert df.attrs['feed'] == 'iex'
    assert str(df.index.tz) == 'UTC'
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()


def test_midstream_sip_403_restarts_on_iex_without_splice_or_rewind():
    sub = _api_error(403, SUB_403_BODY)
    cutoff = datetime(2024, 6, 1, tzinfo=UTC)

    def errors(req):
        if req.feed == DataFeed.SIP and req.limit != 1 and req.start >= cutoff:
            return sub
        return None

    client = FakeClient({'sip': SIP_FIRST, 'iex': IEX_FIRST}, errors=errors)
    df = cct.stream_alpaca_bars('AAPL', 2024, client=client, feed='sip')
    assert df.attrs['feed'] == 'iex'

    switch = next(i for i, c in enumerate(client.calls)
                  if c.feed == DataFeed.SIP and c.limit != 1 and c.start >= cutoff)
    after = [c for c in client.calls[switch + 1:] if c.limit != 1]
    assert after and all(c.feed == DataFeed.IEX for c in after)
    starts = [c.start for c in after]
    assert starts == sorted(starts)              # never rewinds
    assert all(c.start < c.end for c in after)   # no inverted windows
    # SIP chunks discarded → whole series is IEX starting at the restart point.
    assert df.index[0] < pd.Timestamp(cutoff)
    assert len(df) == len(after)


def test_midstream_auth_failure_raises():
    def errors(req):
        if req.limit != 1 and req.start >= datetime(2025, 3, 1, tzinfo=UTC):
            return _api_error(401, '{"message": "unauthorized."}')
        return None

    client = FakeClient({'sip': SIP_FIRST}, errors=errors)
    with pytest.raises(RuntimeError, match='rejected credentials'):
        cct.stream_alpaca_bars('AAPL', start_dt=START_RECENT, client=client, feed='sip')


def test_all_chunks_lost_aborts():
    client = FakeClient({'sip': SIP_FIRST},
                        errors=lambda r: _api_error(503, 'unavailable') if r.limit != 1 else None)
    with pytest.raises(RuntimeError, match='No data received'):
        cct.stream_alpaca_bars('AAPL', start_dt=START_RECENT, client=client, feed='sip')


def test_lost_chunks_above_threshold_abort():
    def errors(req):
        if req.limit != 1 and req.start.month in (2, 3, 4) and req.start.year == 2025:
            return _api_error(503, 'unavailable')
        return None

    client = FakeClient({'sip': SIP_FIRST}, errors=errors)
    with pytest.raises(RuntimeError, match='chunks lost'):
        cct.stream_alpaca_bars('AAPL', start_dt=START_RECENT, client=client, feed='sip')


def test_single_lost_chunk_is_retried_then_reported(monkeypatch):
    monkeypatch.setattr(cct, 'MAX_FAILED_CHUNK_FRACTION', 0.50)
    hits = {'n': 0}
    target_start = {'v': None}

    def errors(req):
        if req.limit == 1:
            return None
        if target_start['v'] is None and req.start >= datetime(2025, 2, 1, tzinfo=UTC):
            target_start['v'] = req.start
        if req.start == target_start['v']:
            hits['n'] += 1
            return _api_error(503, 'unavailable')
        return None

    client = FakeClient({'sip': SIP_FIRST}, errors=errors)
    with pytest.warns(UserWarning, match='chunks lost'):
        df = cct.stream_alpaca_bars('AAPL', start_dt=START_RECENT, client=client, feed='sip')
    assert hits['n'] == cct.MAX_CHUNK_ATTEMPTS
    assert len(df) > 0


def test_unknown_symbol_raises_clear_error():
    client = FakeClient({})
    with pytest.raises(RuntimeError, match='NO bars'):
        cct.stream_alpaca_bars('ZZZZ', start_dt=START_RECENT, client=client, feed='sip')



# ═══════════════════════════════════════════════════════════════════════════
# 4. CV, alignment & weighting math
# ═══════════════════════════════════════════════════════════════════════════

def test_purged_cv_never_yields_empty_splits_and_has_no_leakage():
    n = 600
    t1 = np.minimum(np.arange(n) + 12, n - 1)
    cv = cct.PurgedWalkForwardCV(n_splits=5, embargo_bars=15)
    folds = list(cv.split(n, t1))
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        assert len(train_idx) > 0 and len(test_idx) > 0
        assert t1[train_idx].max() < test_idx[0] - 15
    assert folds[-1][1][-1] == n - 1  # remainder absorbed by last fold


def test_purged_cv_rejects_tiny_dataset():
    with pytest.raises(ValueError, match='too small'):
        list(cct.PurgedWalkForwardCV(5, 15).split(50, np.arange(50)))


def test_align_proba_missing_low_class():
    """Model trained without DOWN → classes_=[1,2] must map to CHOP/UP, not DOWN/CHOP."""
    model = SimpleNamespace(classes_=np.array([1, 2]))
    out = cct._align_proba_to_classes(model, np.array([[0.3, 0.7]]))
    np.testing.assert_allclose(out, [[0.0, 0.3, 0.7]])


def test_align_proba_lightgbm_label_encoded_three_columns():
    """LightGBM num_class=3 with classes_=[1,2] returns 3 cols (CHOP, UP, unused)."""
    model = SimpleNamespace(classes_=np.array([1, 2]))
    out = cct._align_proba_to_classes(model, np.array([[0.2, 0.8, 0.0]]))
    np.testing.assert_allclose(out, [[0.0, 0.2, 0.8]])


def test_kish_ratio_identical_and_degenerate_weights():
    # Identical uniqueness still gets the 0.5→1.0 information-clock recency
    # decay, so weights are NOT identical; ratio must stay finite and >= 0.70.
    w, _, ratio = quant_core.compute_fold_sample_weights(np.full(100, 0.4))
    assert 0.70 <= ratio <= 1.0
    expected = (w.sum() ** 2) / (w ** 2).sum() / len(w)
    assert ratio == pytest.approx(expected)
    w, _, ratio = quant_core.compute_fold_sample_weights(np.zeros(10))
    assert np.isfinite(ratio) and np.isfinite(w).all()
    _, _, ratio = quant_core.compute_fold_sample_weights(np.array([1.0]))
    assert ratio == pytest.approx(1.0)
    _, _, ratio = quant_core.compute_fold_sample_weights(np.array([]))
    assert ratio == 0.0


def test_uniqueness_rejects_invalid_t1():
    with pytest.raises(ValueError, match='>= i'):
        quant_core.compute_vectorized_uniqueness(np.array([0, 0, 1]))


def test_triple_barrier_handles_nan_and_zero_atr():
    idx = pd.date_range('2024-01-02 09:30', periods=19, freq='20min')
    bars = pd.DataFrame({'Open': 100.0, 'High': 101.0, 'Low': 99.0, 'Close': 100.0}, index=idx)
    atr = pd.Series(np.nan, index=idx)
    atr.iloc[5:] = 0.0
    labels, t1 = cct.triple_barrier_label(bars, atr, pd.Series(idx.normalize(), index=idx))
    assert (labels == 0).all() and (t1 == np.arange(19)).all()


def test_train_v3_model_rejects_nan_features():
    X = pd.DataFrame(np.zeros((10, 31)), columns=quant_core.FEATURE_COLUMNS)
    X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match='NaN'):
        cct.train_v3_model(X, np.zeros(10, int), np.arange(10), np.ones(10))




# ═══════════════════════════════════════════════════════════════════════════
# 5. End-to-end artifact contract → inference
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def trained_artifact(tmp_path_factory):
    """Train a REAL (tiny) V3+ artifact via train_v3_model on synthetic data."""
    rng = np.random.default_rng(7)
    n = 900
    X = pd.DataFrame(rng.normal(size=(n, len(quant_core.FEATURE_COLUMNS))),
                     columns=quant_core.FEATURE_COLUMNS)
    labels = rng.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3])
    t1 = np.minimum(np.arange(n) + 6, n - 1)
    u = quant_core.compute_vectorized_uniqueness(t1)

    mp = pytest.MonkeyPatch()
    mp.setitem(cct.LGB_PARAMS, 'n_estimators', 20)
    mp.setitem(cct.LGB_PARAMS, 'min_child_samples', 10)
    mp.setitem(cct.RF_PARAMS, 'n_estimators', 10)
    mp.setitem(cct.RF_PARAMS, 'n_jobs', 1)
    try:
        artifact, metrics = cct.train_v3_model(X, labels, t1, u)
    finally:
        mp.undo()

    path = tmp_path_factory.mktemp('art') / 'stock_price_model.pkl'
    joblib.dump(artifact, path, compress=3)
    return path, X, metrics


def test_artifact_has_exact_key_contract(trained_artifact):
    path, _, metrics = trained_artifact
    raw = joblib.load(path)
    assert isinstance(raw, dict)
    for key in ('lgb_model_str', 'rf_model', 'calib_coef', 'calib_intercept',
                'feature_columns', 'config', 'env_versions', 'model_version'):
        assert key in raw, key
    assert isinstance(raw['lgb_model_str'], str)  # native text, not a pickled Booster
    assert raw['feature_columns'] == quant_core.FEATURE_COLUMNS
    assert {'numpy', 'scikit-learn', 'lightgbm'} <= set(raw['env_versions'])
    assert {'tp_atr_mult', 'sl_atr_mult'} <= set(raw['config'])
    assert len(raw['calib_coef']) == 3 and len(raw['calib_intercept']) == 3
    assert list(raw['rf_model'].classes_) == [0, 1, 2]
    assert 'slippage_bps' in metrics
    json.dumps(metrics, default=str)  # metadata-serialisable


def test_booster_rebuilds_from_model_string(trained_artifact):
    import lightgbm as lgb
    path, _, _ = trained_artifact
    booster = lgb.Booster(model_str=joblib.load(path)['lgb_model_str'])
    assert booster.num_feature() == 31
    assert booster.num_model_per_iteration() == 3


def test_dict_adapter_predicts_three_classes(trained_artifact):
    path, X, _ = trained_artifact
    adapter = quant_core.DictModelAdapter(joblib.load(path))
    proba = adapter.predict_proba(X.iloc[:5])
    assert proba.shape == (5, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert set(adapter.predict(X.iloc[:5])) <= {-1, 0, 1}
    # Shuffled column order must give identical output (column-order enforcement)
    shuffled = X.iloc[:5][list(reversed(quant_core.FEATURE_COLUMNS))]
    np.testing.assert_allclose(adapter.predict_proba(shuffled), proba)
    with pytest.raises(ValueError, match='dimension mismatch'):
        adapter.predict_proba(X.iloc[:5].values[:, :30])


def test_predict_v3_runs_on_real_artifact(trained_artifact, monkeypatch, tmp_path):
    import predict as predict_module
    from tests.test_pipeline import sample_market_data

    path, _, metrics = trained_artifact
    meta = tmp_path / 'model_metadata.json'
    meta.write_text(json.dumps({
        'model_version': 'V3+', 'tp_atr_mult': 1.5, 'sl_atr_mult': 0.75,
        'horizon_bars': 12, 'metrics': metrics,
    }, default=str), encoding='utf-8')
    data = sample_market_data(rows=200)
    monkeypatch.setattr(predict_module, 'MODEL_PATH', path)
    monkeypatch.setattr(predict_module, 'METADATA_PATH', meta)
    monkeypatch.setattr(predict_module, '_model_cache', {})
    monkeypatch.setattr(predict_module, 'get_latest_stock_data', lambda t: data)
    monkeypatch.setattr(predict_module, 'download_spy_intraday', lambda: sample_market_data(200))
    monkeypatch.setattr(predict_module, 'get_news_summary', lambda t: {'sentiment': 0.0})

    result = predict_module.predict_v3('AAPL')
    assert result['predicted_class'] in (-1, 0, 1)
    assert set(result['probabilities']) == {'P_DOWN', 'P_CHOP', 'P_UP'}
    assert sum(result['probabilities'].values()) == pytest.approx(1.0)
    assert result['model_version'] == 'V3+'
    assert result['atr_at_entry'] > 0


def test_train_v4_alpha_model_and_predict(monkeypatch):
    """Verify that train_v3_model trains on 43 extended alpha features and produces a V4 artifact."""
    rng = np.random.default_rng(42)
    n = 600
    cols = list(quant_core.EXTENDED_FEATURE_COLUMNS)
    assert len(cols) == 43
    X = pd.DataFrame(rng.normal(size=(n, len(cols))), columns=cols)
    labels = rng.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3])
    t1 = np.minimum(np.arange(n) + 6, n - 1)
    u = quant_core.compute_vectorized_uniqueness(t1)

    monkeypatch.setitem(cct.LGB_PARAMS, 'n_estimators', 10)
    monkeypatch.setitem(cct.LGB_PARAMS, 'min_child_samples', 5)
    monkeypatch.setitem(cct.RF_PARAMS, 'n_estimators', 5)
    monkeypatch.setitem(cct.RF_PARAMS, 'n_jobs', 1)

    artifact, metrics = cct.train_v3_model(X, labels, t1, u)
    assert artifact['model_version'] == 'V4-Alpha'
    assert artifact['feature_columns'] == quant_core.EXTENDED_FEATURE_COLUMNS

    adapter = quant_core.DictModelAdapter(artifact)
    proba = adapter.predict_proba(X.iloc[:5])
    assert proba.shape == (5, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert set(adapter.predict(X.iloc[:5])) <= {-1, 0, 1}
