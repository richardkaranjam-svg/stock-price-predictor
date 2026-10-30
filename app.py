from flask import Flask, jsonify, render_template, request
import joblib
import json
from datetime import datetime, timezone, timedelta

from pathlib import Path

from config import is_market_open
from predict import get_latest_stock_data, predict_v3
from broker import get_paper_account, submit_paper_market_order
from auto_trader import paper_trader
from news_client import usage_status

app = Flask(__name__)
MODEL_PATH = Path(__file__).resolve().parent / 'stock_price_model.pkl'
METADATA_PATH = Path(__file__).resolve().parent / 'model_metadata.json'


def ensure_fresh_model(ticker):
    """V3+ read-only staleness checker.

    NEVER triggers a local retrain or overwrites stock_price_model.pkl.
    Checks file age and sets a status flag if retraining is recommended.
    Returns a status dict for the API response.
    """
    status = {
        'model_exists': MODEL_PATH.exists(),
        'metadata_exists': METADATA_PATH.exists(),
        'retrain_recommended_in_cloud': False,
        'model_age_days': None,
        'model_version': None,
    }

    if not MODEL_PATH.exists():
        status['retrain_recommended_in_cloud'] = True
        return status

    if METADATA_PATH.exists():
        try:
            metadata = json.loads(METADATA_PATH.read_text(encoding='utf-8'))
            status['model_version'] = metadata.get('model_version', 'V2')
            trained_at_str = metadata.get('trained_at')
            if trained_at_str:
                trained_at = datetime.fromisoformat(trained_at_str)
                age = datetime.now(timezone.utc) - trained_at
                status['model_age_days'] = age.days
                if age > timedelta(days=30):
                    status['retrain_recommended_in_cloud'] = True
        except (json.JSONDecodeError, ValueError, KeyError):
            status['retrain_recommended_in_cloud'] = True

    return status


def get_ticker():
    payload = request.get_json(silent=True) or {}
    raw_ticker = request.args.get('ticker') or payload.get('ticker') or 'AAPL'
    return str(raw_ticker).strip().upper()


@app.get('/health')
def health():
    return jsonify({'status': 'ok', 'model_exists': MODEL_PATH.exists()})


@app.post('/shutdown')
def shutdown():
    import os
    import threading
    import time
    def _kill():
        time.sleep(0.5)
        os._exit(0)
    threading.Thread(target=_kill, daemon=True).start()
    return jsonify({'status': 'shutting_down', 'message': 'Server is stopping...'})


@app.get('/news/usage')
def news_usage():
    return jsonify(usage_status())


@app.get('/model/metadata')
def model_metadata():
    """Serve model_metadata.json for dynamic UI binding."""
    if METADATA_PATH.exists():
        try:
            metadata = json.loads(METADATA_PATH.read_text(encoding='utf-8'))
            model_status = ensure_fresh_model(get_ticker())
            metadata['model_status'] = model_status
            return jsonify(metadata)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'model_metadata.json not found'}), 404


@app.get('/')
def home():
    return render_template('index.html')


@app.get('/paper/account')
def paper_account():
    try:
        return jsonify(get_paper_account())
    except Exception as error:
        return jsonify({'error': f'Alpaca paper account unavailable: {error}'}), 503


@app.post('/paper/order')
def paper_order():
    payload = request.get_json(silent=True) or {}
    try:
        ticker = (payload.get('ticker') or '').strip().upper()
        if not ticker or len(ticker) > 32 or any(character.isspace() for character in ticker):
            raise ValueError('Enter a valid ticker symbol.')
        return jsonify(submit_paper_market_order(ticker, payload.get('side'), payload.get('quantity')))
    except (RuntimeError, ValueError) as error:
        return jsonify({'error': str(error)}), 400
    except Exception as error:
        return jsonify({'error': f'Paper broker request failed: {error}'}), 502


@app.get('/paper/auto/status')
def paper_auto_status():
    return jsonify(paper_trader.status())


@app.post('/paper/auto/start')
def paper_auto_start():
    payload = request.get_json(silent=True) or {}
    try:
        ticker = (payload.get('ticker') or 'AAPL').strip().upper()
        if not ticker or len(ticker) > 32 or any(character.isspace() for character in ticker):
            raise ValueError('Enter a valid ticker symbol.')
        started = paper_trader.start(
            ticker=ticker,
            interval_seconds=payload.get('interval_seconds', 1200),
            threshold=payload.get('threshold', 0.005),
            quantity=payload.get('quantity', 1),
            slippage_pct=payload.get('slippage_pct', 0.0015),
        )
        return jsonify({'started': started, **paper_trader.status()})
    except (TypeError, ValueError) as error:
        return jsonify({'error': str(error)}), 400


@app.post('/paper/auto/stop')
def paper_auto_stop():
    paper_trader.stop()
    return jsonify(paper_trader.status())


@app.get('/paper/history')
def paper_history():
    return jsonify({'trades': paper_trader.history()})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        ticker = get_ticker()

        # Market-hours guard: refuse predictions when the market is closed.
        # The frontend renders a "Market closed" card when market_closed=true.
        if not is_market_open():
            return jsonify({
                'market_closed': True,
                'error': 'US equity market is currently closed. Forecasts are unavailable outside 09:30–16:00 ET Mon–Fri.',
            }), 403

        model_status = ensure_fresh_model(ticker)
        if not model_status['model_exists']:
            return jsonify({'error': 'No model found. Train in Colab first.'}), 404

        # Use V3+ prediction pipeline
        result = predict_v3(ticker)

        # Load metadata for response
        metadata = json.loads(METADATA_PATH.read_text(encoding='utf-8')) if METADATA_PATH.exists() else {}

        return jsonify({
            'ticker': ticker,
            'current_price': result['current_price'],
            'predicted_price': result['predicted_price'],
            'predicted_return': result['predicted_return'],
            'predicted_class': result['predicted_class'],
            'confidence': result['confidence'],
            'atr_at_entry': result.get('atr_at_entry'),
            'ev_atr_value': result.get('ev_atr_value'),
            'take_profit_price': result.get('take_profit_price'),
            'stop_loss_price': result.get('stop_loss_price'),
            'metrics': metadata.get('metrics', {}),
            'model_version': metadata.get('model_version', 'V2'),
            'training_bars': metadata.get('training_bars'),
            'horizon_bars': metadata.get('horizon_bars', 20),
            'tp_atr_mult': metadata.get('tp_atr_mult'),
            'sl_atr_mult': metadata.get('sl_atr_mult'),
            'news': result.get('news', {}),
            'news_used_by_model': bool(metadata.get('news_training_source')),
            'model_status': model_status,
            'market_closed': False,
        })
    except Exception as error:
        return jsonify({'error': str(error)}), 400


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
