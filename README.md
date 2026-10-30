# Stock Price Predictor & Autonomous Quantitative Trader (V3+)

An institutional-grade, intraday quantitative trading and price prediction system built with Python, scikit-learn, LightGBM, and the Alpaca Markets API.

---

## 🚀 Overview (V3+)

The V3+ engine replaces naive single-step regression with a path-aware, calibrated classification and autonomous trade execution architecture designed for institutional execution and risk management:

- **100% Alpaca SIP & Real-Time Data**: Zero dependency on unreliable scrapers (`yfinance`). Uses 09:30-anchored, 20-minute resampled bars (`resample_bars_20min`).
- **Institutional Feature Engine (`quant_core.py`)**: 31 stationary features including gap-adjusted fractional differentiation ($d=0.40$), normalized ATR barriers, intraday VWAP deviations, and SPY market beta.
- **Calibrated Ensemble Model**: Dual LightGBM (0.60) + Random Forest (0.40) probability blend with Platt calibration, trained on 10 years of intraday bars using Purged & Embargoed expanding walk-forward cross-validation.
- **Autonomous Execution Engine (`auto_trader.py`)**:
  - Positive Expected Value gate ($P_{\text{win}} \ge 0.58$ and $\text{EV} > 0$).
  - Dynamic ATR triple-barrier stops (1.50x TP, 0.75x SL) with a 1R breakeven ratchet.
  - Hard 15:45 ET auto-flatten rule & 15:30 ET entry cutoff.
  - High-signal trading windows (09:30–11:00 and 14:30–15:30 ET).
  - Daily kill-switch (-2.0% equity drawdown protection).
  - Short-selling eligibility and margin guard validation.
- **Interactive Web Dashboard (`app.py`)**: Real-time Flask dashboard with live model diagnostics, probability distributions, paper trading controls, and audit log.
- **SQLite Audit Trail & Auto-Migration (`db_migration.py`)**: Automatic idempotent schema migration recording ATR at entry, expected value, ratchet status, and confidence metrics.

---

## 🛠️ Project Structure

```text
├── quant_core.py             # Deliverable A: Core math, fracdiff, feature extraction, calibration
├── colab_cloud_trainer.py    # Deliverable B: Cloud 10-year walk-forward CV trainer (Google Colab)
├── data_loader.py            # Alpaca SIP / real-time bar fetching & feature preparation
├── broker.py                 # Alpaca paper trading client with short-selling & margin guards
├── auto_trader.py            # Autonomous trading engine with risk guards and dynamic stops
├── predict.py                # V3+ calibrated inference engine using dict model artifact
├── app.py                    # Flask web dashboard and REST API
├── config.py                 # Market-hours guards and configuration utilities
├── db_migration.py           # SQLite schema migration utility
├── ensemble.py               # Backward-compatible shim for DictModelAdapter
├── news_client.py            # MarketAux headline sentiment ingestion and rate limiter
├── train_model.py            # Protective deprecation guard against local 780-bar overwrites
├── stock_price_model.pkl     # Trained V3+ model artifact (plain dictionary)
├── model_metadata.json       # V3+ ensemble hyperparameters and OOS validation metrics
├── test_quant_invariants.py  # 12-test invariant verification test suite
├── tests/
│   ├── test_colab_trainer.py # 37 unit tests for cloud trainer and labeling math
│   └── test_pipeline.py      # Integration and regression tests
├── templates/
│   └── index.html            # Dashboard HTML
├── static/
│   ├── app.js                # Dashboard client script
│   └── style.css             # Dashboard styling
├── requirements.txt          # Python dependencies
└── v3_migration_state.md     # Full migration specification and verification audit
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.11–3.14)
- Alpaca Paper Trading Account (Free API keys at [alpaca.markets](https://alpaca.markets))

### 2. Clone & Install
```bash
git clone https://github.com/richardkaranjam-svg/stock-price-predictor.git
cd stock-price-predictor
pip install -r requirements.txt
```

### 3. Configure API Credentials
Copy `.env.example` to `.env` and fill in your Alpaca paper credentials:
```bash
cp .env.example .env
```
Contents of `.env`:
```ini
APCA_API_KEY_ID=your-paper-api-key-id
APCA_API_SECRET_KEY=your-paper-api-secret
APCA_API_BASE_URL=https://paper-api.alpaca.markets
MARKETAUX_API_TOKEN=your-optional-marketaux-token
```

---

## 📈 Running the System

### Run Model Inference
```bash
python predict.py
```
Evaluates the latest closed 20-minute bar for AAPL and outputs calibrated class probabilities, dynamic ATR barrier targets, and EV gate status.

### Start the Web Dashboard
```bash
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to view:
- Live 20-minute predictions & probabilities
- Current market-hours state
- Alpaca paper account equity and position
- Autonomous trading controls and trade execution history

### Model Training (Google Colab)
Local short-sample training is disabled to protect against lookahead bias and model wipeouts. To retrain the 10-year model:
1. Open Google Colab with Python 3.10+.
2. Upload `colab_cloud_trainer.py`.
3. Set your Alpaca API credentials in Colab secrets or environment variables.
4. Run the script to execute walk-forward cross-validation and download the exported `stock_price_model.pkl` and `model_metadata.json`.

---

## 🧪 Testing & Verification

Run the full invariant and pipeline test suites:

```bash
# 12 institutional invariant tests (zero-lookahead, gap isolation, closed bars, margin guards, etc.)
pytest test_quant_invariants.py

# Full unit and integration test suite
pytest tests/
```
