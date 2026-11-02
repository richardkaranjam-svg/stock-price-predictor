from flask import Flask, jsonify, render_template, request, send_file
import joblib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from config import is_market_open, get_market_status
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


@app.get('/market/status')
def market_status():
    return jsonify(get_market_status())


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


@app.get('/paper/signals')
def paper_signals():
    """Retrieve logged prediction and trader decision signals from SQLite."""
    try:
        limit = min(int(request.args.get('limit', 50)), 500)
        import sqlite3
        db_path = Path(__file__).resolve().parent / 'paper_trades.db'
        if not db_path.exists():
            return jsonify({'signals': [], 'count': 0})
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            'SELECT id, timestamp_utc, ticker, model_version, side, confidence, price, features_hash, outcome, source, action '
            'FROM signals ORDER BY id DESC LIMIT ?', (limit,)
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return jsonify({'signals': rows, 'count': len(rows)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.get('/report')
def view_client_report():
    """Serve the client strategy factsheet HTML."""
    report_file = Path(__file__).resolve().parent / 'reports' / 'client_strategy' / 'factsheet.html'
    if not report_file.exists():
        for alt in [
            Path(__file__).resolve().parent / 'reports' / 'benchmarks' / 'trend_breakout_donchian' / 'factsheet.html',
            Path(__file__).resolve().parent / 'reports' / 'alpha_101' / 'factsheet.html',
        ]:
            if alt.exists():
                report_file = alt
                break
    if not report_file.exists():
        return "No factsheet generated yet. Run audit_strategy.py first.", 404
    return send_file(str(report_file))


@app.get('/meta-report')
def view_meta_report():
    """Serve the ML Meta-Labeled optimized strategy factsheet HTML."""
    report_file = Path(__file__).resolve().parent / 'reports' / 'meta_labeled' / 'factsheet.html'
    if not report_file.exists():
        return "No meta-labeled factsheet generated yet. Run meta_label_optimizer.py first.", 404
    return send_file(str(report_file))


@app.get('/leaderboard')
def view_leaderboard():
    """Serve the Master Strategy Leaderboard HTML."""
    report_file = Path(__file__).resolve().parent / 'reports' / 'league_table.html'
    if not report_file.exists():
        return "No leaderboard generated yet. Run generate_league_table.py first.", 404
    return send_file(str(report_file))


@app.post('/api/v3/filter')
def webhook_trade_filter():
    """Real-time ML Trade Filter & Bouncer API.

    Ingests an external trade signal (from TradingView, NinjaTrader, Python bot)
    and evaluates it against predict_v3 confidence and market microstructure.
    """
    payload = request.get_json(silent=True) or {}
    ticker = (payload.get('ticker') or 'AAPL').strip().upper()
    signal_side = (payload.get('side') or payload.get('action') or 'LONG').strip().upper()
    min_confidence = float(payload.get('min_confidence', 0.65))
    source = str(payload.get('source') or 'webhook_filter')

    desired_class = 1 if ('LONG' in signal_side or 'BUY' in signal_side) else 0
    side_str = 'LONG' if desired_class == 1 else 'SHORT'

    try:
        res = predict_v3(ticker)
        model_class = res['predicted_class']
        model_conf = float(res['confidence'])
        price = float(res['current_price'])
        f_hash = res.get('features_hash', '')

        direction_matches = (model_class == desired_class)
        conf_passes = (model_conf >= min_confidence)

        if direction_matches and conf_passes:
            decision = 'EXECUTE'
            reason = f'Model confirms {side_str} with {model_conf*100:.1f}% confidence.'
        elif not direction_matches:
            decision = 'REJECT'
            reason = f'Model direction conflict (Model predicts {"LONG" if model_class==1 else "SHORT"} vs Signal {side_str}).'
        else:
            decision = 'REJECT'
            reason = f'Confidence {model_conf*100:.1f}% below required hurdle ({min_confidence*100:.1f}%).'

        from db_migration import log_signal
        db_path = Path(__file__).resolve().parent / 'paper_trades.db'
        log_signal(
            conn_or_path=db_path,
            ticker=ticker,
            side=side_str,
            confidence=model_conf,
            price=price,
            features_hash=f_hash,
            source=source,
            action=decision,
            model_version=res.get('model_version', 'v3-gbdt-ensemble'),
        )

        return jsonify({
            'ticker': ticker,
            'signal_side': side_str,
            'decision': decision,
            'reason': reason,
            'model_confidence': model_conf,
            'min_confidence_required': min_confidence,
            'current_price': price,
            'take_profit_price': res.get('take_profit_price'),
            'stop_loss_price': res.get('stop_loss_price'),
            'features_hash': f_hash,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.get('/benchmarks/<name>')
def view_benchmark_report(name: str):
    """Serve a specific benchmark strategy factsheet HTML."""
    clean_name = name.lower().strip()
    strat_dir = Path(__file__).resolve().parent / 'reports' / 'benchmarks' / clean_name
    report_file = strat_dir / 'factsheet.html'
    if not report_file.exists():
        return f"Benchmark '{clean_name}' not found.", 404
    return send_file(str(report_file))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        ticker = get_ticker()
        market_info = get_market_status()

        model_status = ensure_fresh_model(ticker)
        if not model_status['model_exists']:
            return jsonify({'error': 'No model found. Train in Colab first.'}), 404

        # Use V3+ prediction pipeline (uses latest available closed bars)
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
            'market_closed': not market_info['is_open'],
            'market_status': market_info,
        })
    except Exception as error:
        return jsonify({'error': str(error)}), 400


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
