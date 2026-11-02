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

> **Scope note:** the client strategy-audit tooling (trade-log auditor, meta-labeling, benchmark portfolios, league table, Discord/Slack digests, interactive CLI, Docker files) lives on the `feature/strategy-audit` branch. `main` is focused on the V3+ predictor, trainer, paper trader and dashboard.

---

## 🛠️ Project Structure

```text
├── quant_core.py             # Core math, fracdiff, 31-feature extraction (43 with alphas), calibration
├── formulaic_alphas.py       # Causal WorldQuant-101 alpha features (optional V4 feature set)
├── colab_cloud_trainer.py    # Walk-forward CV cloud trainer (V3+ default, V4 with INCLUDE_ALPHAS=1)
├── data_loader.py            # Alpaca SIP / real-time bar fetching & feature preparation
├── broker.py                 # Alpaca paper trading client with short-selling & margin guards
├── auto_trader.py            # Autonomous trading engine with risk guards and dynamic stops
├── predict.py                # Calibrated inference; builds the feature set the model was trained on
├── backtest_report.py        # Out-of-sample backtest of the model's signals
├── evaluate_v3_model.py      # Walk-forward OOS signal generation for backtest_report.py
├── simulate_session.py       # 19-bar intraday trading session & signal lifecycle simulator
├── app.py                    # Flask web dashboard and REST API
├── config.py                 # Market-hours guards and configuration utilities
├── db_migration.py           # SQLite schema migration utility & signals persistence
├── stock_price_model.pkl     # Trained model artifact (plain dictionary)
├── model_metadata.json       # Ensemble hyperparameters and validation metrics
├── test_quant_invariants.py  # Invariant verification test suite
├── tests/                    # Unit and integration tests
│   ├── test_pipeline.py
│   ├── test_version_and_signals.py
│   ├── test_formulaic_alphas.py   # Includes no-lookahead / scale-invariance checks
│   ├── test_colab_trainer.py
│   ├── test_backtest_report.py
│   └── test_simulate_session.py
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

### Optional: Formulaic Alpha Features (V4)
```bash
python formulaic_alphas.py --optimize-portfolio
```
Ranks the 12 Kakushadze alphas by Spearman Rank IC across forward horizons. Every alpha is causal: rank operators use a trailing 50-bar window and price-level alphas are normalised, so appending future bars never changes past values (enforced by `tests/test_formulaic_alphas.py`). Train with `INCLUDE_ALPHAS=1` to produce a 43-feature V4 model; `predict.py` detects it from the artifact's `feature_columns` and builds matching features automatically.

### Out-of-Sample Backtest
```bash
python evaluate_v3_model.py
python backtest_report.py --signals path/to/signals.csv --bars-cache path/to/bars --out reports
```
Generates walk-forward OOS signals from the trained model and backtests them with ATR stops, slippage and commissions.

### Model Training (Google Colab)
Local short-sample training is disabled to protect against lookahead bias and model wipeouts. To retrain the 10-year model:
1. Open Google Colab with Python 3.11+ (check with `!python --version`).
2. Upload `colab_cloud_trainer.py` and `quant_core.py`.
3. Install the exact versions the server uses (see `requirements.txt`) and then **restart the runtime**:
   `!pip install -q numpy==2.4.6 pandas==3.0.3 scikit-learn==1.9.0 joblib==1.5.3 lightgbm==4.7.0 "alpaca-py>=0.40,<1" pytz`
   pip may print dependency-resolver warnings about `google-colab` (pandas) and `numba` (numpy). They are harmless for training, which uses neither.
4. After the restart, verify `numpy/pandas/sklearn/lightgbm` print `2.4.6 / 3.0.3 / 1.9.0 / 4.7.0` before training.
5. Set your Alpaca API credentials in Colab secrets or environment variables.
6. The server refuses models whose scikit-learn/lightgbm major.minor differ from the local install. Set `ALLOW_VERSION_MISMATCH=1` only as a temporary bypass.
4. Run the script to execute walk-forward cross-validation and download the exported `stock_price_model.pkl` and `model_metadata.json`.

---

## 🧪 Testing & Verification

Run the full invariant and pipeline test suites:

```bash
# Institutional invariant tests (zero-lookahead, gap isolation, closed bars, margin guards, etc.)
pytest test_quant_invariants.py

# Full unit and integration test suite
pytest tests/
```
