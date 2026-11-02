# V3+ Migration State Tracker
> **Resumption Protocol**: Open a new session and say _"Read `v3_migration_state.md` and resume the next incomplete task"_.
> Last updated: 2026-10-28

---

## Phase 0: Planning & State Tracker
- [x] Create `v3_migration_state.md` (this file)
- [x] User grants permission to begin Phase 1

---

## Phase 1: Deliverable A — `quant_core.py` (Shared Feature Engine & Math)
- [x] 1A.1 — Create `quant_core.py` module skeleton with imports, constants, `FEATURE_COLUMNS` list (31 features)
- [x] 1A.2 — Implement `resample_bars_20min()`: 09:30-anchored resampler with `offset='10min'`, intraday VWAP reset, 15:50 stub drop
- [x] 1A.3 — Implement `frac_diff_ffd()`: overnight-gap-free fractional differentiation (d=0.40, thres=1e-3) using gap-adjusted continuous intraday log-price series
- [x] 1A.4 — Implement `compute_features()`: all 31 stationary features (23 core + 8 institutional), ATR-normalized, feed-invariant
- [x] 1A.5 — Implement `compute_vectorized_uniqueness()`: O(N) zero-loop uniqueness via NumPy difference array
- [x] 1A.6 — Implement `compute_fold_sample_weights()`: per-fold information-clock weighting with Kish N_eff/N >= 0.70
- [x] 1A.7 — Implement `predict_calibrated_probabilities()`: LGB/RF blend (0.60/0.40) with Platt/Softmax calibration
- [x] 1A.8 — Implement `DictModelAdapter` class: `.predict_proba()`, `.predict()`, `.classes_` wrapper for plain dict artifacts
- [x] 1A.9 — Unit-test `quant_core.py` functions in isolation (sanity checks before integration)

---

## Phase 2: Deliverable B — `colab_cloud_trainer.py` (Cloud Training Pipeline)
- [x] 2B.1 — Create `colab_cloud_trainer.py` skeleton with Alpaca SIP streaming setup (DataFeed.SIP, Adjustment.ALL)
- [x] 2B.2 — Implement year-by-year AAPL + SPY 5-min bar streaming from 2016–present
- [x] 2B.3 — Implement symmetric path-aware High/Low dynamic ATR triple-barrier labeling (horizon capped at min(12, 15:40 ET))
- [x] 2B.4 — Implement purged & embargoed 5-fold expanding walk-forward CV with per-fold uniqueness weighting
- [x] 2B.5 — Implement LightGBM + RF training with dynamic `max_samples` and `subsample` from mean uniqueness
- [x] 2B.6 — Implement 85/15 fit/calibration split within each fold for Platt calibration coefficients
- [x] 2B.7 — Implement OOS profitability audit (Win Rate, Profit Factor, Sharpe, Max Drawdown) across all test folds
- [x] 2B.8 — Implement portable plain-dict artifact export (stock_price_model.pkl + model_metadata.json) with `google.colab.files.download()`

---

## Phase 3: Deliverable C — Local Workspace Integration (Gaps 1–5 + Tweaks)

### C.1 — `data_loader.py` (100% Alpaca, Zero yfinance)
- [x] 3C.1a — Remove all `yfinance` imports and code from `data_loader.py`
- [x] 3C.1b — Implement hybrid Alpaca SIP + real-time fetch (warm-up via SIP for `end <= now - 16m`, recent tail via SIP with IEX fallback)
- [x] 3C.1c — Implement strict closed-bar rule (filter out any bar where `bar_open + 20min > wall_clock_ET`)
- [x] 3C.1d — Wire feature computation through `quant_core.compute_features()`

### C.2 — `broker.py` (Short-Selling & Margin Guard Helpers)
- [x] 3C.2a — Add `check_short_eligibility(symbol)` method (equity >= $2000, shortable, easy_to_borrow)
- [x] 3C.2b — Add `get_short_safe_buying_power(safety_factor=0.80)` method
- [x] 3C.2c — Add `cancel_open_orders_for_symbol(symbol)` method

### C.3 — `predict.py` (V3+ Dict Artifact Inference)
- [x] 3C.3a — Refactor `predict.py` to load plain dict artifact and route through `quant_core.predict_calibrated_probabilities()`
- [x] 3C.3b — Return full V3+ prediction payload (class, confidence, probabilities, ATR barriers, EV, timestamps)

### C.4 — `auto_trader.py` (Execution Hardening)
- [x] 3C.4a — Add version/checksum verification on startup against `env_versions` in artifact
- [x] 3C.4b — Implement 15:45 ET auto-flatten & 15:30 ET entry cutoff with `broker.cancel_open_orders_for_symbol()`
- [x] 3C.4c — Implement high-signal windows (09:30–11:00, 14:30–15:30) & -2.0% daily kill-switch
- [x] 3C.4d — Implement EV gate: require P_win >= 0.58 AND ev_atr_value > 0 (log EV_GATE_REJECTED if blocked)
- [x] 3C.4e — Implement short selling guard via `broker.check_short_eligibility()` (log SHORT_GUARD_BLOCKED if blocked)
- [x] 3C.4f — Implement dynamic ATR stops (1.50x TP, 0.75x SL) & 1R breakeven ratchet (RATCHET_1R_BREAKEVEN)

### C.5 — SQLite Schema Auto-Migration
- [x] 3C.5a — Implement idempotent `migrate_paper_trades_schema(conn)` with PRAGMA table_info checks
- [x] 3C.5b — Add V3+ columns: atr_at_entry, ev_atr_value, take_profit_price, stop_loss_price, ratchet_triggered, confidence, model_version
- [x] 3C.5c — Update all INSERT statements to write V3+ columns and reason codes

### C.6 — `app.py`, `ensemble.py`, `train_model.py` & Frontend
- [x] 3C.6a — Neutralize `ensure_fresh_model()` in `app.py`: read-only staleness checker, never retrain
- [x] 3C.6b — Remove `train_model.py` import from `app.py`, add deprecation guard to `train_model.py`
- [x] 3C.6c — Convert `ensemble.py` to backward-compatible shim delegating to `quant_core.DictModelAdapter`
- [x] 3C.6d — Remove `yfinance` from `predict.py`, `train_model.py`, `app.py`, `news_client.py` (if present)
- [x] 3C.6e — Remove `yfinance` from `requirements.txt`
- [x] 3C.6f — [SUBAGENT] Update Flask routes to serve dynamic metadata from `model_metadata.json`
- [x] 3C.6g — [SUBAGENT] Update `templates/index.html` and `static/app.js` with dynamic V3+ labels
- [x] 3C.6h — [SUBAGENT] Update `static/style.css` if needed for new UI elements

---

## Phase 4: Deliverable D — `test_quant_invariants.py` (12-Test Verification Suite)
- [x] 4D.1 — [SUBAGENT] Create test scaffolding with pytest fixtures and docstrings
- [x] 4D.2 — Test 1: Zero Lookahead Invariant
- [x] 4D.3 — Test 2: Overnight Gap Isolation in FracDiff
- [x] 4D.4 — Test 3: Closed-Bar Slicing Invariant
- [x] 4D.5 — Test 4: 09:30 Bucket Alignment
- [x] 4D.6 — Test 5: 15:45 ET Auto-Flatten & 15:30 ET Cutoff
- [x] 4D.7 — Test 6: Short Selling & Margin Guard (broker.py)
- [x] 4D.8 — Test 7: __main__-Free Artifact & predict.py Dict Inference
- [x] 4D.9 — Test 8: O(N) Uniqueness Speed & Kish N_eff Invariant
- [x] 4D.10 — Test 9: Gap 1 Protection (app.py never overwrites V3+ model)
- [x] 4D.11 — Test 10: Gap 4 SQLite Schema Migration
- [x] 4D.12 — Test 11: Gap 5 Deprecation Protection (train_model.py raises RuntimeError)
- [x] 4D.13 — Test 12: Zero yfinance Dependency Audit

---

## Phase 5: Final Integration & Smoke Test
- [x] 5.1 — Run `pytest tests/test_pipeline.py` (existing V2 tests still pass)
- [x] 5.2 — Run `pytest test_quant_invariants.py` (all 12 V3+ tests pass)
- [x] 5.3 — Restart Flask server and verify dashboard loads with V3+ labels
- [x] 5.4 — Verify `paper_trades.db` schema migration ran successfully
- [x] 5.5 — Final `yfinance` grep across entire codebase confirms zero hits
- [x] 5.6 — Archive `compute_chop_rate.py` (depends on removed yfinance data_loader functions)

---

## Issue-to-Task Cross-Reference
| Issue # | Description | Resolved By |
|:---:|:---|:---|
| 1 | Sample Starvation & yfinance Ban | 2B.1–2B.2, 3C.1a, 3C.6d–e |
| 2 | app.py ensure_fresh_model() Wipeout | 3C.6a |
| 3 | predict.py Legacy .predict_proba() Crash | 1A.7–8, 3C.3a–b |
| 4 | broker.py Missing Short/Margin Guards | 3C.2a–c |
| 5 | SQLite Schema Missing V3+ Columns | 3C.5a–c |
| 6 | ensemble.py & train_model.py Dead Code | 3C.6b–c |
| 7 | 390-Minute Resampling Bug | 1A.2 |
| 8 | Train-vs-Live Horizon Mismatch | 2B.3 (horizon capped at 12/15:40) |
| 9 | Asymmetric 3-Class Labeling Bias | 2B.3 (symmetric path-aware) |
| 10 | Overlapping Label Serial Correlation | 1A.5–6, 2B.4 |
| 11 | Non-Stationarity & Secular Bull Drift | 1A.3–4 (frac_diff, SPY beta) |
| 12 | FracDiff Overnight Boundaries (Tweak 1) | 1A.3 |
| 13 | Alpaca SIP 15-Min Delay (Tweak 2) | 3C.1b |
| 14 | Frontend UI Label Sync (Tweak 3) | 3C.6f–h |
| 15 | Short Selling Margin Guards (Tweak 4) | 3C.2a–c, 3C.4e |
| 16 | Pickle Cross-Environment (Tweak 5) | 1A.7–8, 3C.4a |
| 17 | Uniqueness + Recency Weighting (Tweak 6) | 1A.5–6, 2B.4–5 |
| 18 | Overnight Gap & Unclosed Bar Lookahead | 1A.3, 3C.1c, 3C.4b |

---

## Phase 6: Scope Cleanup (2026-11-03)
Goal: bring `main` back to the V3+ objective (predict -> trade -> audit trail) after a drift into client-audit tooling.
- [x] 6.1 - Safety net: tag `pre-scope-cleanup` (= `b638ea8`) and branch `feature/strategy-audit` holding all off-topic work
- [x] 6.2 - Move off `main`: `audit_strategy`, `meta_label_optimizer`, `benchmark_strategies`, `batch_processor`, `generate_league_table`, `fetch_bars`, `quant_stats`, `cli`, `notifications`, `daily_digest`, `PITCH_AND_INTAKE_GUIDE.md`, Docker files and their tests
- [x] 6.3 - `app.py` / `index.html`: remove report, leaderboard, benchmark routes and `/api/v3/filter` webhook (kept `/paper/signals` feed)
- [x] 6.4 - Stop tracking generated `reports/`, `inbox/`, `outbox/` (they made GitHub report HTML as the main language); `.gitattributes` marks notebooks as documentation
- [x] 6.5 - Keep Gemini additions that serve the core: signal logging + outcome filling (`db_migration`), library version guard (`predict.check_library_versions`), `env_versions` in artifacts, pinned `requirements.txt`
- [x] 6.6 - **Fix V4 lookahead**: `formulaic_alphas.rank()` / `ts_rank()` are now trailing-window (50 / N bars); `alpha_012`, `alpha_041`, `alpha_054` normalised to be price-level independent. Regression tests: `test_alphas_have_no_lookahead`, `test_alphas_are_scale_invariant`, `test_rank_is_trailing_percentile`
- [x] 6.7 - **Fix train/serve skew**: `data_loader.create_features(include_alphas=...)`; `predict_v3` builds the feature set the loaded artifact was trained on (`_model_uses_alphas`). Regression test: `test_predict_v3_builds_43_features_for_v4_artifact`
- [x] 6.8 - Notebook defaults to `INCLUDE_ALPHAS=0` (V3+); README rewritten for `main`
- [ ] 6.9 - **Retrain in Colab on pinned versions** (sklearn 1.9.0 / lightgbm 4.7.0) so the version guard passes; then delete `ALLOW_VERSION_MISMATCH=1` from `.env`
- [ ] 6.10 - Retrain with `INCLUDE_ALPHAS=1` (now causal) and compare OOS metrics against V3+ before deploying V4
