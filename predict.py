"""
predict.py — V3+ Prediction Pipeline.

Loads stock_price_model.pkl (plain dict or legacy EnsembleModel) and
routes inference through quant_core.predict_calibrated_probabilities().

Returns the full V3+ prediction payload: predicted_class, confidence,
probabilities, ATR barriers, EV value, and timestamps.

Resolves Gap 2 (Issue #3) and Tweak 5 (Issue #16).
"""
import json
import time
import threading
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd

import quant_core
from data_loader import download_intraday_data, download_spy_intraday, create_features, get_news_summary

MODEL_PATH = Path(__file__).resolve().parent / 'stock_price_model.pkl'
METADATA_PATH = Path(__file__).resolve().parent / 'model_metadata.json'
DATA_CACHE_TTL_SECONDS = 1200
_data_cache: dict = {}
_data_cache_lock = threading.Lock()
_model_cache: dict = {}
_model_cache_lock = threading.Lock()


def _load_model() -> Any:
    """Load model artifact with caching and V2/V3+ detection.

    Returns either a DictModelAdapter (V3+) or legacy EnsembleModel (V2).
    """
    with _model_cache_lock:
        mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0
        cached = _model_cache.get('model')
        if cached and cached[0] == mtime:
            return cached[1]

    raw = joblib.load(MODEL_PATH)

    if isinstance(raw, dict):
        # V3+ plain-dict artifact → wrap in DictModelAdapter
        model = quant_core.DictModelAdapter(raw)
    else:
        # V2 legacy EnsembleModel — use as-is
        model = raw

    with _model_cache_lock:
        _model_cache['model'] = (mtime, model)
    return model


def _load_metadata() -> dict:
    """Load model_metadata.json."""
    if METADATA_PATH.exists():
        try:
            return json.loads(METADATA_PATH.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def get_latest_stock_data(ticker: str, days: int = 30):
    """Get recent 20-min bar data for a ticker via Alpaca.

    Uses a 20-minute TTL cache to avoid redundant API calls.
    """
    cache_key = (ticker.upper(), days)
    now = time.monotonic()
    with _data_cache_lock:
        cached = _data_cache.get(cache_key)
        if cached and now - cached[0] < DATA_CACHE_TTL_SECONDS:
            return cached[1].copy()

    data = download_intraday_data(ticker, interval='20m')
    with _data_cache_lock:
        _data_cache[cache_key] = (time.monotonic(), data.copy())
    return data


def predict_v3(ticker: str = 'AAPL') -> Dict[str, Any]:
    """Generate a full V3+ prediction payload.

    Routes inference through quant_core for both V3+ dict artifacts and
    legacy V2 EnsembleModel objects.

    Parameters
    ----------
    ticker : str
        Stock symbol (default 'AAPL').

    Returns
    -------
    dict
        Full prediction payload containing:
        - predicted_class: int (-1, 0, +1)
        - confidence: float
        - probabilities: dict (P_DOWN, P_CHOP, P_UP)
        - current_price: float
        - predicted_price: float
        - predicted_return: float
        - atr_at_entry: float
        - ev_atr_value: float
        - take_profit_price: float
        - stop_loss_price: float
        - bar_open_et: str
        - bar_close_et: str
        - news: dict
    """
    model = _load_model()
    metadata = _load_metadata()

    # --- Get data and compute features ---
    data = get_latest_stock_data(ticker)
    spy_data = download_spy_intraday()

    features = create_features(data, spy_data)

    # Apply closed-bar filter
    now_et = quant_core.get_et_now()
    data_closed = quant_core.filter_closed_bars(data, now_et=now_et)

    if data_closed.empty:
        raise ValueError("No closed bars available for prediction.")

    # Recompute features on closed bars only
    features_closed = create_features(data_closed, spy_data)

    if features_closed.empty:
        raise ValueError("Not enough closed-bar data to create prediction features.")

    latest_features = features_closed.iloc[-1:]

    # --- Get news ---
    news = get_news_summary(ticker)

    # --- Inference ---
    is_v3 = isinstance(model, quant_core.DictModelAdapter)

    if is_v3:
        probas = model.predict_proba(latest_features)[0]
        classes = model.classes_  # [-1, 0, 1]
        predicted_class = int(classes[probas.argmax()])
        confidence = float(probas.max())
        p_down, p_chop, p_up = float(probas[0]), float(probas[1]), float(probas[2])
    else:
        # V2 legacy path
        probas = model.predict_proba(latest_features)[0]
        classes = model.classes_
        predicted_class_mapped = classes[probas.argmax()]
        predicted_class = int(predicted_class_mapped - 1)  # V2: 0,1,2 → -1,0,+1
        confidence = float(probas.max())
        p_down = float(probas[0]) if len(probas) > 0 else 0.0
        p_chop = float(probas[1]) if len(probas) > 1 else 0.0
        p_up = float(probas[2]) if len(probas) > 2 else 0.0

    current_price = float(data_closed['Close'].iloc[-1])

    # --- ATR barriers ---
    tp_mult = metadata.get('tp_atr_mult', 1.50)
    sl_mult = metadata.get('sl_atr_mult', 0.75)

    # Compute ATR at entry
    import pytz
    et = pytz.timezone('America/New_York')
    if data_closed.index.tzinfo is None:
        session_date = data_closed.index.normalize()
    else:
        session_date = data_closed.index.tz_convert(et).normalize()
    session_date_series = pd.Series(session_date, index=data_closed.index)

    atr = quant_core._day_boundary_atr(
        data_closed['High'], data_closed['Low'], data_closed['Close'],
        session_date_series, period=14
    )
    atr_at_entry = float(atr.iloc[-1]) if not atr.empty and not np.isnan(atr.iloc[-1]) else 0.0

    if predicted_class == 1:
        take_profit_price = current_price + tp_mult * atr_at_entry
        stop_loss_price = current_price - sl_mult * atr_at_entry
    elif predicted_class == -1:
        take_profit_price = current_price - tp_mult * atr_at_entry
        stop_loss_price = current_price + sl_mult * atr_at_entry
    else:
        take_profit_price = current_price
        stop_loss_price = current_price

    # --- Expected Value calculation ---
    p_win = p_up if predicted_class == 1 else (p_down if predicted_class == -1 else 0.0)
    slippage_bps = metadata.get('metrics', {}).get('slippage_bps', 3)
    if isinstance(slippage_bps, dict):
        slippage_bps = 3
    ev_atr_value = (
        p_win * tp_mult * atr_at_entry
        - (1 - p_win) * sl_mult * atr_at_entry
        - 2 * slippage_bps / 10000 * current_price
    ) if predicted_class != 0 and atr_at_entry > 0 else 0.0

    # --- Bar timestamps ---
    last_bar_ts = data_closed.index[-1]
    bar_open_et = str(last_bar_ts)
    bar_close_et = str(last_bar_ts + np.timedelta64(20, 'm'))

    # --- Pseudo price for UI compatibility ---
    predicted_return = 0.002 if predicted_class == 1 else (-0.002 if predicted_class == -1 else 0.0)
    predicted_price = current_price * (1 + predicted_return)

    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': {'P_DOWN': p_down, 'P_CHOP': p_chop, 'P_UP': p_up},
        'current_price': current_price,
        'predicted_price': predicted_price,
        'predicted_return': predicted_return,
        'atr_at_entry': atr_at_entry,
        'ev_atr_value': ev_atr_value,
        'take_profit_price': take_profit_price,
        'stop_loss_price': stop_loss_price,
        'bar_open_et': bar_open_et,
        'bar_close_et': bar_close_et,
        'news': news,
        'model_version': metadata.get('model_version', 'V2'),
    }


def predict_next_price(ticker: str = 'AAPL'):
    """Legacy-compatible prediction entry point.

    Returns (predicted_class, confidence) tuple for backward compatibility.
    """
    result = predict_v3(ticker)
    return result['predicted_class'], result['confidence']


if __name__ == "__main__":
    result = predict_v3('AAPL')
    print(f"\n{'='*50}")
    print(f"V3+ Prediction for AAPL")
    print(f"{'='*50}")
    print(f"Current Price: ${result['current_price']:.2f}")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"ATR at Entry: ${result['atr_at_entry']:.2f}")
    print(f"TP: ${result['take_profit_price']:.2f}  SL: ${result['stop_loss_price']:.2f}")
    print(f"EV (ATR): {result['ev_atr_value']:.4f}")
    print(f"{'='*50}\n")
