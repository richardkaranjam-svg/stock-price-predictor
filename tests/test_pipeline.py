"""
test_pipeline.py — V3+ Integration Tests for Core Pipeline.

Updated from V2 to work with the V3+ API surface:
  - data_loader.create_features (uses quant_core)
  - predict.predict_v3 (dict-artifact inference)
  - auto_trader.PaperTrader (V3+ execution engine)
  - broker (with short-selling guards)
"""
import numpy as np
import pandas as pd
import pytest

import app
import auto_trader
import broker
import quant_core
from data_loader import create_features


def sample_market_data(rows=200):
    """Generate synthetic 20-min OHLCV bars with VWAP for V3+ features.

    Produces enough bars (200 ~ 10.5 trading days) for the 31-feature
    warm-up period (EMA-50, ATR-14, rolling-20, etc.).
    """
    rng = np.random.default_rng(42)
    bar_times = [pd.Timedelta(hours=9, minutes=30 + 20 * i) for i in range(19)]
    base = pd.Timestamp('2024-01-02')
    stamps = []
    day = base
    while len(stamps) < rows:
        if day.weekday() < 5:
            for bt in bar_times:
                stamps.append(day + bt)
                if len(stamps) >= rows:
                    break
        day += pd.Timedelta(days=1)

    index = pd.DatetimeIndex(stamps[:rows])
    close = 150.0 + np.cumsum(rng.normal(0, 0.5, rows))
    noise = np.abs(rng.normal(0, 0.3, rows))
    open_ = close + rng.normal(0, 0.2, rows)
    high = np.maximum(open_, close) + noise
    low = np.minimum(open_, close) - noise
    volume = rng.uniform(50_000, 500_000, rows)
    tp = (high + low + close) / 3.0

    df = pd.DataFrame({
        'Open': open_, 'High': high, 'Low': low, 'Close': close,
        'Volume': volume,
    }, index=index)

    session = index.normalize()
    tp_x_vol = tp * volume
    cum_tp_vol = pd.Series(tp_x_vol, index=index).groupby(session).cumsum()
    cum_vol = pd.Series(volume, index=index).groupby(session).cumsum()
    df['VWAP'] = (cum_tp_vol / cum_vol.replace(0, np.nan)).values
    return df


def test_features_have_v3_columns():
    """V3+ features must produce exactly 31 columns from FEATURE_COLUMNS."""
    data = sample_market_data(rows=200)
    features = create_features(data)
    assert list(features.columns) == quant_core.FEATURE_COLUMNS
    assert len(features.columns) == 31
    assert not features.isna().any().any()


def test_features_require_enough_history():
    """create_features must reject datasets with fewer than 65 bars."""
    with pytest.raises(ValueError, match='65 trading bars'):
        create_features(sample_market_data(rows=49))


def test_prediction_endpoint_returns_v3_result(monkeypatch, tmp_path):
    """The /predict endpoint must return V3+ fields (predicted_class, etc.)."""
    data = sample_market_data(rows=200)
    spy_data = sample_market_data(rows=200)

    class FakeModel:
        classes_ = np.array([-1, 0, 1])
        def predict_proba(self, values):
            return np.array([[0.1, 0.1, 0.8]])

    import predict as predict_module

    class FakeAdapter:
        classes_ = np.array([-1, 0, 1])
        def predict_proba(self, values):
            return np.array([[0.1, 0.1, 0.8]])

    monkeypatch.setattr(app, 'MODEL_PATH', tmp_path / 'model.pkl')
    app.MODEL_PATH.write_bytes(b'model')
    monkeypatch.setattr(app, 'METADATA_PATH', tmp_path / 'meta.json')
    (tmp_path / 'meta.json').write_text('{"model_version":"V3+","tp_atr_mult":1.5,"sl_atr_mult":0.75}')
    monkeypatch.setattr(app, 'is_market_open', lambda: True)
    monkeypatch.setattr(predict_module, 'get_latest_stock_data', lambda ticker: data)
    monkeypatch.setattr(predict_module, 'download_spy_intraday', lambda: spy_data)
    monkeypatch.setattr(predict_module, '_load_model', lambda: FakeAdapter())

    response = app.app.test_client().get('/predict?ticker=AAPL')
    body = response.get_json()
    assert response.status_code == 200
    assert body['ticker'] == 'AAPL'
    assert 'predicted_class' in body
    assert 'confidence' in body
    assert body['predicted_class'] in (-1, 0, 1)


def test_paper_order_returns_json_for_broker_failures(monkeypatch):
    """Broker failures must return structured JSON errors, not 500s."""
    monkeypatch.setattr(app, 'submit_paper_market_order',
                        lambda *args: (_ for _ in ()).throw(Exception('token rejected')))

    response = app.app.test_client().post('/paper/order', json={
        'ticker': 'AAPL',
        'side': 'buy',
        'quantity': 1,
    })

    assert response.status_code == 502
    assert response.get_json()['error'] == 'Paper broker request failed: token rejected'


def test_autonomous_trader_defaults_to_1200_second_cycles(monkeypatch):
    """PaperTrader must default to 20-minute (1200s) trading cycles."""
    class FakeThread:
        def __init__(self, target, daemon):
            self.target = target
        def start(self):
            pass
        def is_alive(self):
            return True

    monkeypatch.setattr(auto_trader.threading, 'Thread', FakeThread)
    trader = auto_trader.PaperTrader()

    assert trader.start()
    assert trader.status()['interval_seconds'] == 1200


def test_slippage_buffer_and_limit_order_support(monkeypatch):
    """PaperTrader must have a slippage buffer and broker must support limit orders."""
    trader = auto_trader.PaperTrader()
    assert trader.slippage_pct == 0.0015

    # Build minimal fakes for the alpaca trading enums/requests that
    # broker.submit_paper_limit_order imports lazily.
    FakeOrderSide = type('OrderSide', (), {'BUY': 'buy', 'SELL': 'sell'})
    FakeTimeInForce = type('TimeInForce', (), {'DAY': 'day'})

    class FakeLimitOrderRequest:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    fake_enums = type('Module', (), {
        'OrderSide': FakeOrderSide,
        'TimeInForce': FakeTimeInForce,
    })()
    fake_requests = type('Module', (), {
        'LimitOrderRequest': FakeLimitOrderRequest,
    })()

    import sys
    monkeypatch.setitem(sys.modules, 'alpaca.trading.enums', fake_enums)
    monkeypatch.setitem(sys.modules, 'alpaca.trading.requests', fake_requests)

    fake_client = type('Client', (), {
        'submit_order': lambda self, order_data=None: type('Order', (), {
            'id': 'limit-1', 'symbol': 'AAPL', 'side': type('Side', (), {'value': 'buy'})(),
            'qty': 1.0, 'status': type('Status', (), {'value': 'accepted'})(),
        })(),
    })()
    monkeypatch.setattr(broker, 'get_alpaca_client', lambda: fake_client)
    order = broker.submit_paper_limit_order('aapl', 'buy', 1, 101.0)
    assert order['id'] == 'limit-1'
    assert order['limit_price'] == '101.0'
