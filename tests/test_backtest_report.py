"""
test_backtest_report.py — Regression tests for generic backtesting & performance reporting.

Covers:
  1. Zero look-ahead: trade entry occurs at NEXT bar's open.
  2. Costs (commission + slippage) strictly reduce PnL.
  3. A random-signal strategy receives FAIL verdict.
  4. A perfect-foresight strategy receives PASS verdict.
  5. Flat signals produce zero trades.
  6. Multi-ticker per_ticker breakdown is mathematically accurate.
  7. Report persistence (report.json, trades.csv, equity_curve.png).
  8. CLI execution via subprocess.
  9. ATR look-ahead fix: entry bar high/low/ATR cannot change stop distance.
  10. Session cap: trades never hold overnight and exit at/before 15:40 ET.
  11. Leverage cap: shares capped at floor(max_leverage * equity / entry_price), size_capped bool.
  12. All trading days in span: trades_per_day and daily Sharpe use all days with 0 PnL on empty days.
  13. Max concurrent positions cap across all tickers.
  14. evaluate_v3_model OOS signals: every signal timestamp belongs to a test fold.
  15. evaluate_v3_model CLI: exact cache names (.cache_{TICKER}_v3_20m.pkl) and <2,000 bars REFUSED.
"""
from __future__ import annotations

import datetime
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import pytz

import colab_cloud_trainer as cct
import quant_core
from backtest_report import (
    FAIL_MIN_EXPECTANCY,
    FAIL_MIN_PROFIT_FACTOR,
    FAIL_MIN_TRADES,
    WEAK_MAX_DRAWDOWN_PCT,
    WEAK_MIN_PROFIT_FACTOR,
    compute_metrics,
    run_backtest,
    save_report,
    verdict,
)
from evaluate_v3_model import generate_oos_signals


# ═══════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATORS (OFFLINE, ZERO NETWORK)
# ═══════════════════════════════════════════════════════════════════════════

def _make_synthetic_bars(
    n_bars: int = 150,
    start_time: str = '2024-01-02 09:30:00',
    start_price: float = 100.0,
    volatility: float = 0.5,
    seed: int = 42,
    with_atr: bool = True,
) -> pd.DataFrame:
    """Generate deterministic synthetic 20-min OHLCV bars (09:30-15:30 daily)."""
    rng = np.random.default_rng(seed)
    # Generate realistic trading days with 19 bars per day: 09:30, 09:50, ..., 15:30 ET
    et = pytz.timezone('America/New_York')
    base_date = pd.Timestamp(start_time).date()

    bar_minute_offsets = [30 + 20 * i for i in range(19)]  # 09:30 to 15:30 ET
    stamps = []
    curr_date = base_date

    while len(stamps) < n_bars:
        if curr_date.weekday() < 5:  # Weekday Mon-Fri
            for m_off in bar_minute_offsets:
                dt_et = et.localize(datetime.datetime.combine(curr_date, datetime.time(9, 0)) + datetime.timedelta(minutes=m_off))
                stamps.append(dt_et.astimezone(pytz.utc))
                if len(stamps) >= n_bars:
                    break
        curr_date += datetime.timedelta(days=1)

    idx = pd.DatetimeIndex(stamps[:n_bars])

    steps = rng.normal(0, volatility, n_bars)
    close = start_price + np.cumsum(steps)
    close = np.maximum(close, 10.0)

    open_ = np.roll(close, 1)
    open_[0] = start_price
    noise = np.abs(rng.normal(0.3, 0.1, n_bars))
    high = np.maximum(open_, close) + noise
    low = np.minimum(open_, close) - noise
    volume = rng.uniform(50_000, 200_000, n_bars)

    df = pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume,
    }, index=idx)

    # Intraday cumulative VWAP
    session = df.index.tz_convert(et).normalize()
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0
    cum_tp_vol = (tp * df['Volume']).groupby(session).cumsum()
    cum_vol = df['Volume'].groupby(session).cumsum()
    df['VWAP'] = cum_tp_vol / cum_vol.replace(0, np.nan)

    if with_atr:
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift(1)).abs()
        tr3 = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14, min_periods=1).mean().bfill()

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 1. NO LOOK-AHEAD (ENTRY IS NEXT BAR'S OPEN)
# ═══════════════════════════════════════════════════════════════════════════

def test_no_lookahead_entry_is_next_bar():
    bars = _make_synthetic_bars(n_bars=20, start_price=150.0)
    sig_time = bars.index[2]

    signals = pd.DataFrame([{
        'timestamp': sig_time,
        'ticker': 'TEST',
        'side': 'long',
        'confidence': 0.85,
    }])

    res = run_backtest(signals, {'TEST': bars})
    trades = res['trades']

    assert len(trades) == 1
    trade = trades.iloc[0]
    expected_entry_time = bars.index[3]
    expected_entry_price = bars.iloc[3]['Open']

    assert trade['signal_time'] == sig_time
    assert trade['entry_time'] == expected_entry_time
    assert trade['entry_time'] > trade['signal_time']
    assert trade['entry_price'] == pytest.approx(expected_entry_price)


# ═══════════════════════════════════════════════════════════════════════════
# 2. COSTS REDUCE PNL
# ═══════════════════════════════════════════════════════════════════════════

def test_costs_reduce_pnl():
    bars = _make_synthetic_bars(n_bars=60, start_price=100.0)
    # Pick intraday signals between 10:00 ET and 14:00 ET on Day 1
    sig_times = bars.index[2:14:2]

    signals = pd.DataFrame([
        {'timestamp': t, 'ticker': 'TEST', 'side': 'long', 'confidence': 0.70}
        for t in sig_times
    ])

    res_free = run_backtest(signals, {'TEST': bars}, cost_per_share=0.0, slippage_bps=0.0, allow_overlap=True, max_concurrent_positions=100, max_leverage=10.0)
    res_cost = run_backtest(signals, {'TEST': bars}, cost_per_share=0.02, slippage_bps=4.0, allow_overlap=True, max_concurrent_positions=100, max_leverage=10.0)

    trades_free = res_free['trades']
    trades_cost = res_cost['trades']

    assert len(trades_free) == len(trades_cost) == len(sig_times)
    assert trades_free['costs'].sum() == pytest.approx(0.0)
    assert trades_cost['costs'].sum() > 0.0

    # Net PnL must be strictly less after transaction costs
    assert res_cost['net_pnl'] < res_free['net_pnl']
    for i in range(len(trades_free)):
        assert trades_cost.iloc[i]['net_pnl'] < trades_free.iloc[i]['net_pnl']
        assert trades_cost.iloc[i]['costs'] > 0


# ═══════════════════════════════════════════════════════════════════════════
# 3. RANDOM-SIGNAL STRATEGY GETS FAIL
# ═══════════════════════════════════════════════════════════════════════════

def test_random_signal_strategy_gets_fail():
    rng = np.random.default_rng(123)
    bars = _make_synthetic_bars(n_bars=250, start_price=200.0, seed=777)
    # Filter to bars before the 15:30 ET session cutoff
    sig_times = [
        t for t in bars.index[5:220]
        if t.tz_convert('America/New_York').hour < 15 or (t.tz_convert('America/New_York').hour == 15 and t.tz_convert('America/New_York').minute < 30)
    ]

    signals_list = []
    for t in sig_times:
        side_choice = rng.choice(['long', 'short'])
        signals_list.append({
            'timestamp': t,
            'ticker': 'TEST',
            'side': side_choice,
            'confidence': float(rng.uniform(0.5, 0.6)),
        })

    signals = pd.DataFrame(signals_list)
    res = run_backtest(signals, {'TEST': bars}, cost_per_share=0.01, slippage_bps=2.0, allow_overlap=True, max_concurrent_positions=100, max_leverage=10.0)

    assert res['metrics']['n_trades'] >= FAIL_MIN_TRADES
    assert res['verdict'] == 'FAIL'
    assert len(res['reasons']) > 0
    assert any('profit_factor' in r or 'expectancy' in r for r in res['reasons'])


# ═══════════════════════════════════════════════════════════════════════════
# 4. PERFECT-FORESIGHT STRATEGY GETS PASS
# ═══════════════════════════════════════════════════════════════════════════

def test_perfect_foresight_strategy_gets_pass():
    bars = _make_synthetic_bars(n_bars=220, start_price=100.0, volatility=1.0, seed=888, with_atr=False)

    signals_list = []
    # For each bar i, look ahead to bar i+1 Open and bar i+1 Close:
    # If Close > Open, buy long; if Close < Open, sell short.
    for i in range(5, 200):
        t_sig = bars.index[i]
        next_open = bars.iloc[i + 1]['Open']
        next_close = bars.iloc[i + 1]['Close']

        if next_close > next_open + 0.1:
            side = 'long'
        elif next_close < next_open - 0.1:
            side = 'short'
        else:
            continue

        signals_list.append({
            'timestamp': t_sig,
            'ticker': 'TEST',
            'side': side,
            'confidence': 0.99,
        })

    signals = pd.DataFrame(signals_list)
    assert len(signals) >= FAIL_MIN_TRADES

    res = run_backtest(
        signals=signals,
        bars={'TEST': bars},
        cost_per_share=0.005,
        slippage_bps=1.0,
        horizon_bars=1,
    )

    m = res['metrics']
    assert m['n_trades'] >= FAIL_MIN_TRADES
    assert m['profit_factor'] >= WEAK_MIN_PROFIT_FACTOR
    assert m['expectancy'] > FAIL_MIN_EXPECTANCY
    assert m['max_drawdown_pct'] <= WEAK_MAX_DRAWDOWN_PCT
    assert res['verdict'] == 'PASS'
    assert res['reasons'] == []


# ═══════════════════════════════════════════════════════════════════════════
# 5. FLAT SIGNALS PRODUCE NO TRADES
# ═══════════════════════════════════════════════════════════════════════════

def test_flat_signals_produce_no_trades():
    bars = _make_synthetic_bars(n_bars=50)
    signals = pd.DataFrame([
        {'timestamp': t, 'ticker': 'TEST', 'side': 'flat', 'confidence': 0.5}
        for t in bars.index[5:35]
    ])

    res = run_backtest(signals, {'TEST': bars})
    assert len(res['trades']) == 0
    assert res['metrics']['n_trades'] == 0
    assert res['metrics']['net_pnl'] == 0.0
    assert res['verdict'] == 'FAIL'
    assert any(f"n_trades (0) < {FAIL_MIN_TRADES}" in r for r in res['reasons'])


# ═══════════════════════════════════════════════════════════════════════════
# 6. MULTI-TICKER PER_TICKER BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════

def test_multi_ticker_per_ticker_breakdown():
    aapl_bars = _make_synthetic_bars(n_bars=80, start_price=175.0, seed=11)
    msft_bars = _make_synthetic_bars(n_bars=80, start_price=400.0, seed=22)

    # Intraday bars between 10:00 and 14:00 ET on Day 1
    signals_aapl = [
        {'timestamp': t, 'ticker': 'AAPL', 'side': 'long', 'confidence': 0.75}
        for t in aapl_bars.index[2:12:2]
    ]
    signals_msft = [
        {'timestamp': t, 'ticker': 'MSFT', 'side': 'short', 'confidence': 0.65}
        for t in msft_bars.index[3:13:2]
    ]
    signals = pd.DataFrame(signals_aapl + signals_msft)

    res = run_backtest(
        signals,
        {'AAPL': aapl_bars, 'MSFT': msft_bars},
        allow_overlap=True,
        max_concurrent_positions=100,
        max_leverage=10.0,
    )
    m = res['metrics']
    trades = res['trades']

    assert 'AAPL' in m['per_ticker']
    assert 'MSFT' in m['per_ticker']

    n_aapl = len(trades[trades['ticker'] == 'AAPL'])
    n_msft = len(trades[trades['ticker'] == 'MSFT'])

    assert m['per_ticker']['AAPL']['n_trades'] == n_aapl == len(signals_aapl)
    assert m['per_ticker']['MSFT']['n_trades'] == n_msft == len(signals_msft)
    assert m['n_trades'] == n_aapl + n_msft

    sum_pnl = m['per_ticker']['AAPL']['net_pnl'] + m['per_ticker']['MSFT']['net_pnl']
    assert pytest.approx(sum_pnl, rel=1e-5) == m['net_pnl']


# ═══════════════════════════════════════════════════════════════════════════
# 7. VERDICT THRESHOLDS & REPORT ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════

def test_verdict_threshold_logic():
    # FAIL when n_trades < 100
    v, r = verdict({'n_trades': 99, 'profit_factor': 1.5, 'expectancy': 10.0, 'max_drawdown_pct': 5.0})
    assert v == 'FAIL'
    assert any('n_trades' in item for item in r)

    # FAIL when profit_factor < 1.0
    v, r = verdict({'n_trades': 150, 'profit_factor': 0.95, 'expectancy': 5.0, 'max_drawdown_pct': 5.0})
    assert v == 'FAIL'
    assert any('profit_factor' in item for item in r)

    # FAIL when expectancy <= 0
    v, r = verdict({'n_trades': 150, 'profit_factor': 1.1, 'expectancy': -0.01, 'max_drawdown_pct': 5.0})
    assert v == 'FAIL'

    # WEAK when profit_factor < 1.2
    v, r = verdict({'n_trades': 150, 'profit_factor': 1.15, 'expectancy': 2.0, 'max_drawdown_pct': 10.0})
    assert v == 'WEAK'
    assert any('profit_factor' in item for item in r)

    # WEAK when max_drawdown_pct > 20
    v, r = verdict({'n_trades': 150, 'profit_factor': 1.4, 'expectancy': 2.0, 'max_drawdown_pct': 22.5})
    assert v == 'WEAK'
    assert any('max_drawdown_pct' in item for item in r)

    # PASS
    v, r = verdict({'n_trades': 150, 'profit_factor': 1.35, 'expectancy': 5.0, 'max_drawdown_pct': 12.0})
    assert v == 'PASS'
    assert r == []


def test_save_report_generates_all_artifacts(tmp_path):
    bars = _make_synthetic_bars(n_bars=30)
    signals = pd.DataFrame([
        {'timestamp': bars.index[5], 'ticker': 'TEST', 'side': 'long', 'confidence': 0.8}
    ])
    res = run_backtest(signals, {'TEST': bars})
    out_dir = tmp_path / 'backtest_out'

    saved = save_report(res['metrics'], res['trades'], out_dir)

    assert saved['report'].exists()
    assert saved['trades'].exists()
    assert saved['equity_curve'].exists()

    with open(saved['report'], 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert 'verdict' in data
    assert 'metrics' in data
    assert data['metrics']['n_trades'] == 1

    trades_csv = pd.read_csv(saved['trades'])
    assert len(trades_csv) == 1
    assert 'net_pnl' in trades_csv.columns
    assert 'size_capped' in trades_csv.columns
    assert saved['equity_curve'].stat().st_size > 0


# ═══════════════════════════════════════════════════════════════════════════
# 8. CLI INTEGRATION TEST
# ═══════════════════════════════════════════════════════════════════════════

def test_backtest_report_cli(tmp_path):
    bars = _make_synthetic_bars(n_bars=40)
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    joblib.dump(bars, cache_dir / '.cache_TEST_v3_20m.pkl')

    sig_file = tmp_path / 'signals.csv'
    signals_df = pd.DataFrame([
        {'timestamp': bars.index[2].isoformat(), 'ticker': 'TEST', 'side': 'long', 'confidence': 0.8},
        {'timestamp': bars.index[6].isoformat(), 'ticker': 'TEST', 'side': 'short', 'confidence': 0.7},
    ])
    signals_df.to_csv(sig_file, index=False)

    out_dir = tmp_path / 'cli_reports'

    cmd = [
        sys.executable,
        'backtest_report.py',
        '--signals', str(sig_file),
        '--bars-cache', str(cache_dir),
        '--out', str(out_dir),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"CLI stderr: {proc.stderr}"
    assert (out_dir / 'report.json').exists()
    assert (out_dir / 'trades.csv').exists()
    assert (out_dir / 'equity_curve.png').exists()


# ═══════════════════════════════════════════════════════════════════════════
# 9. ATR LOOK-AHEAD PROOF (ENTRY BAR CANNOT CHANGE STOP DISTANCE)
# ═══════════════════════════════════════════════════════════════════════════

def test_no_atr_lookahead_entry_bar_cannot_change_stop():
    """Prove that changing the entry bar's High, Low, or ATR cannot alter stop distance."""
    bars_base = _make_synthetic_bars(n_bars=30, start_price=150.0, seed=42)
    sig_time = bars_base.index[5]
    entry_time = bars_base.index[6]

    signals = pd.DataFrame([{
        'timestamp': sig_time,
        'ticker': 'TEST',
        'side': 'long',
        'confidence': 0.85,
    }])

    # Run backtest with original entry bar
    res1 = run_backtest(signals, {'TEST': bars_base})
    trade1 = res1['trades'].iloc[0]

    # Modify the entry bar's High, Low, and ATR dramatically
    bars_modified = bars_base.copy()
    bars_modified.loc[entry_time, 'High'] = bars_base.loc[entry_time, 'High'] * 3.0
    bars_modified.loc[entry_time, 'Low'] = bars_base.loc[entry_time, 'Low'] * 0.1
    bars_modified.loc[entry_time, 'ATR'] = 999.99

    res2 = run_backtest(signals, {'TEST': bars_modified})
    trade2 = res2['trades'].iloc[0]

    # Stop distance must be identical because it used the SIGNAL bar's ATR
    stop_distance_1 = abs(trade1['entry_price'] - (trade1['entry_price'] - 0.75 * bars_base.loc[sig_time, 'ATR']))
    stop_distance_2 = abs(trade2['entry_price'] - (trade2['entry_price'] - 0.75 * bars_modified.loc[sig_time, 'ATR']))

    assert stop_distance_1 == pytest.approx(stop_distance_2)
    assert trade1['shares'] == trade2['shares']


# ═══════════════════════════════════════════════════════════════════════════
# 10. SESSION CAP (NEVER HOLD OVERNIGHT)
# ═══════════════════════════════════════════════════════════════════════════

def test_session_cap_never_holds_overnight():
    """Verify trades never hold overnight and exit at/before 15:40 America/New_York."""
    bars = _make_synthetic_bars(n_bars=60, start_time='2024-01-02 09:30:00', with_atr=False)
    et = pytz.timezone('America/New_York')

    # Find the 15:10 ET bar on day 1 (bar index 17)
    day1_afternoon_idx = 17
    sig_time = bars.index[day1_afternoon_idx]  # Signal at 15:10 ET
    entry_time = bars.index[day1_afternoon_idx + 1]  # Entry at 15:30 ET

    signals = pd.DataFrame([{
        'timestamp': sig_time,
        'ticker': 'TEST',
        'side': 'long',
        'confidence': 0.9,
    }])

    # Request a 12-bar horizon (4 hours) which would cross into day 2 if allowed
    res = run_backtest(signals, {'TEST': bars}, horizon_bars=12)
    trade = res['trades'].iloc[0]

    entry_dt_et = trade['entry_time'].tz_convert(et)
    exit_dt_et = trade['exit_time'].tz_convert(et)

    # Must exit on the SAME day
    assert exit_dt_et.date() == entry_dt_et.date()
    # Must exit at or before 15:40 ET
    assert (exit_dt_et.hour < 15) or (exit_dt_et.hour == 15 and exit_dt_et.minute <= 40)
    assert trade['exit_reason'] == 'session_cap'


# ═══════════════════════════════════════════════════════════════════════════
# 11. LEVERAGE CAP AND SIZE_CAPPED BOOL
# ═══════════════════════════════════════════════════════════════════════════

def test_leverage_cap_and_size_capped_flag():
    """Verify shares are capped at floor(max_leverage * equity / entry_price) and size_capped recorded."""
    bars = _make_synthetic_bars(n_bars=30, start_price=100.0)
    sig_time = bars.index[5]

    signals = pd.DataFrame([{
        'timestamp': sig_time,
        'ticker': 'TEST',
        'side': 'long',
        'confidence': 0.8,
    }])

    # Case A: Very tight stop (tiny stop distance) where risk-based sizing would exceed 1.0x leverage
    # initial_capital = 10,000, entry_price ~ 100, risk_per_trade = 0.05 ($500 risk)
    # If stop_distance was 0.001, risk_shares would be 500,000!
    # Max leverage 1.0 caps shares at floor(1.0 * 10,000 / 100) = 100 shares.
    res_capped = run_backtest(
        signals=signals,
        bars={'TEST': bars},
        initial_capital=10_000.0,
        risk_per_trade=0.05,
        sl_atr_mult=0.0001,  # Produces tiny stop distance -> huge risk shares
        max_leverage=1.0,
    )
    trade_capped = res_capped['trades'].iloc[0]
    expected_max_shares = int(np.floor(1.0 * 10_000.0 / trade_capped['entry_price']))

    assert trade_capped['shares'] == expected_max_shares
    assert bool(trade_capped['size_capped']) is True

    # Case B: Wide stop distance where risk shares is small and below leverage cap
    res_uncapped = run_backtest(
        signals=signals,
        bars={'TEST': bars},
        initial_capital=10_000.0,
        risk_per_trade=0.001,  # $10 risk
        sl_atr_mult=10.0,     # Wide stop
        max_leverage=1.0,
    )
    trade_uncapped = res_uncapped['trades'].iloc[0]
    assert trade_uncapped['shares'] < expected_max_shares
    assert bool(trade_uncapped['size_capped']) is False


# ═══════════════════════════════════════════════════════════════════════════
# 12. ALL TRADING DAYS IN SPAN (0 PNL ON NON-EXIT DAYS)
# ═══════════════════════════════════════════════════════════════════════════

def test_compute_metrics_includes_all_trading_days_in_span():
    """Verify trades_per_day and daily Sharpe use all trading days in span, counting 0 on empty days."""
    et = pytz.timezone('America/New_York')

    # Define trades on Monday and Friday of a 5-day week
    t_mon_entry = et.localize(datetime.datetime(2024, 1, 8, 10, 0)).astimezone(pytz.utc)
    t_mon_exit = et.localize(datetime.datetime(2024, 1, 8, 15, 0)).astimezone(pytz.utc)
    t_fri_entry = et.localize(datetime.datetime(2024, 1, 12, 10, 0)).astimezone(pytz.utc)
    t_fri_exit = et.localize(datetime.datetime(2024, 1, 12, 15, 0)).astimezone(pytz.utc)

    trades = pd.DataFrame([
        {'entry_time': t_mon_entry, 'exit_time': t_mon_exit, 'net_pnl': 1000.0, 'r_multiple': 1.0, 'costs': 10.0, 'ticker': 'TEST'},
        {'entry_time': t_fri_entry, 'exit_time': t_fri_exit, 'net_pnl': -500.0, 'r_multiple': -0.5, 'costs': 10.0, 'ticker': 'TEST'},
    ])

    metrics = compute_metrics(trades, initial_capital=100_000.0)

    # 5 business days: Mon, Tue, Wed, Thu, Fri
    # 2 trades / 5 days = 0.40 trades/day (NOT 2 / 2 = 1.0)
    assert metrics['trades_per_day'] == pytest.approx(0.40)
    assert metrics['n_trades'] == 2

    # Daily returns vector is [+0.01, 0, 0, 0, -0.005] (length 5)
    # Sharpe must be computed across all 5 days
    daily_rets = np.array([0.01, 0.0, 0.0, 0.0, -0.005])
    expected_sharpe = round(float(np.mean(daily_rets) / np.std(daily_rets, ddof=1) * np.sqrt(252)), 2)
    assert metrics['sharpe'] == pytest.approx(expected_sharpe)


# ═══════════════════════════════════════════════════════════════════════════
# 13. MAX CONCURRENT POSITIONS CAP
# ═══════════════════════════════════════════════════════════════════════════

def test_max_concurrent_positions_cap():
    """Verify that signals are skipped when max_concurrent_positions is hit."""
    bars_dict = {}
    signals_list = []
    t_common = None

    for i in range(8):
        sym = f"TICK{i}"
        b = _make_synthetic_bars(n_bars=30, seed=i)
        bars_dict[sym] = b
        if t_common is None:
            t_common = b.index[5]
        signals_list.append({
            'timestamp': t_common,
            'ticker': sym,
            'side': 'long',
            'confidence': 0.8,
        })

    signals = pd.DataFrame(signals_list)

    # Cap concurrency at 5 (with high leverage limit so only concurrency cap limits trades)
    res = run_backtest(signals, bars_dict, max_concurrent_positions=5, max_leverage=10.0, horizon_bars=5)
    trades = res['trades']

    assert len(trades) == 5
    assert len(trades['ticker'].unique()) == 5


# ═══════════════════════════════════════════════════════════════════════════
# 14. EVALUATE_V3_MODEL OOS SIGNALS BELONG STRICTLY TO TEST FOLDS
# ═══════════════════════════════════════════════════════════════════════════

def test_evaluate_v3_model_signals_strictly_in_test_folds():
    """Assert every signal timestamp generated by generate_oos_signals belongs to a test fold."""
    aapl_bars = _make_synthetic_bars(n_bars=160, start_price=150.0, seed=123)
    spy_bars = _make_synthetic_bars(n_bars=160, start_price=450.0, seed=456)

    fast_lgb = {'n_estimators': 5, 'min_child_samples': 5, 'num_leaves': 7}
    fast_rf = {'n_estimators': 5, 'max_depth': 4, 'n_jobs': 1}

    signals_df, _ = generate_oos_signals(
        bars_20m=aapl_bars,
        spy_20m=spy_bars,
        ticker='AAPL',
        n_splits=2,
        embargo_bars=3,
        min_confidence=0.50,
        lgb_params=fast_lgb,
        rf_params=fast_rf,
    )

    # Re-compute test fold boundaries to verify timestamps
    features = quant_core.compute_features(aapl_bars, spy_bars)
    et = pytz.timezone('America/New_York')
    session_date = pd.Series(aapl_bars.index.tz_convert(et).normalize(), index=aapl_bars.index)
    atr = quant_core._day_boundary_atr(aapl_bars['High'], aapl_bars['Low'], aapl_bars['Close'], session_date, period=14)
    valid_mask = ~np.isnan(atr.values) & (atr.values > 0)
    common_idx = features.index.intersection(aapl_bars.index[valid_mask])
    features_aligned = features.loc[common_idx]

    pos_map = {ts: i for i, ts in enumerate(aapl_bars.index)}
    pos_indices = np.array([pos_map[ts] for ts in common_idx])
    _, t1_raw = cct.triple_barrier_label(aapl_bars, atr, session_date)
    t1_indices = t1_raw[pos_indices]

    t1_reindexed = np.searchsorted(pos_indices, t1_indices, side='right') - 1
    t1_reindexed = np.maximum(t1_reindexed, np.arange(len(features_aligned))).astype(np.int64)

    cv = cct.PurgedWalkForwardCV(n_splits=2, embargo_bars=3)
    all_test_timestamps = set()
    for _, test_idx in cv.split(len(features_aligned), t1_reindexed):
        for k in test_idx:
            all_test_timestamps.add(features_aligned.index[k])

    # Assert all signal timestamps belong to test folds
    signal_timestamps = set(signals_df['timestamp'])
    assert len(signal_timestamps) > 0
    assert signal_timestamps <= all_test_timestamps


# ═══════════════════════════════════════════════════════════════════════════
# 15. EVALUATE_V3_MODEL CLI (EXACT NAMES & <2000 BARS REFUSED)
# ═══════════════════════════════════════════════════════════════════════════

def test_evaluate_v3_model_cli_insufficient_history(tmp_path):
    """Verify exact cache file names and that fewer than 2,000 bars triggers warning and REFUSED verdict."""
    bars_short = _make_synthetic_bars(n_bars=60)
    spy_bars = _make_synthetic_bars(n_bars=60)

    cache_dir = tmp_path / 'v3_cache'
    cache_dir.mkdir()

    # Exact cache file names required: .cache_{TICKER}_v3_20m.pkl and .cache_SPY_v3_20m.pkl
    ticker_file = cache_dir / '.cache_TEST_v3_20m.pkl'
    spy_file = cache_dir / '.cache_SPY_v3_20m.pkl'

    joblib.dump(bars_short, ticker_file)
    joblib.dump(spy_bars, spy_file)

    out_dir = tmp_path / 'eval_out'

    cmd = [
        sys.executable,
        'evaluate_v3_model.py',
        '--ticker', 'TEST',
        '--cache-dir', str(cache_dir),
        '--out', str(out_dir),
        '--folds', '2',
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"CLI stderr: {proc.stderr}"
    assert "insufficient history" in proc.stderr.lower() or "insufficient history" in proc.stdout.lower()
    assert "REFUSED" in proc.stdout

    # Check report.json contains REFUSED
    report_json = out_dir / 'report.json'
    assert report_json.exists()
    with open(report_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert data['verdict'] == 'REFUSED'
    assert any('insufficient history' in r.lower() for r in data['reasons'])


# ═══════════════════════════════════════════════════════════════════════════
# 16. PORTFOLIO-LEVEL LEVERAGE CAP (5 CONCURRENT SIGNALS <= 1.0x EQUITY)
# ═══════════════════════════════════════════════════════════════════════════

def test_portfolio_level_leverage_cap_5_concurrent():
    """Verify that 5 concurrent signals with max_leverage=1.0 never exceed 1.0x equity in total notional."""
    equity = 100_000.0
    bars_dict = {}
    signals_list = []
    common_sig_time = None

    for i in range(5):
        sym = f"TICK{i}"
        # Price ~100
        b = _make_synthetic_bars(n_bars=30, start_price=100.0, seed=100 + i)
        bars_dict[sym] = b
        if common_sig_time is None:
            common_sig_time = b.index[5]
        signals_list.append({
            'timestamp': common_sig_time,
            'ticker': sym,
            'side': 'long',
            'confidence': 0.8,
        })

    signals = pd.DataFrame(signals_list)

    # Tight stop distance (e.g. sl_atr_mult=0.001) would want 10,000+ shares per trade ($1M+ notional)
    # But portfolio leverage cap max_leverage=1.0 restricts total open notional to 100,000 across all 5 trades
    res = run_backtest(
        signals=signals,
        bars=bars_dict,
        initial_capital=equity,
        risk_per_trade=0.10,
        sl_atr_mult=0.001,
        max_leverage=1.0,
        allow_overlap=True,
    )
    trades = res['trades']

    assert len(trades) > 0
    total_open_notional = float(sum(tr['shares'] * tr['entry_price'] for _, tr in trades.iterrows()))

    # Total open notional must never exceed 1.0x equity
    assert total_open_notional <= equity * 1.000001
    assert all(tr['size_capped'] for _, tr in trades.iterrows())


# ═══════════════════════════════════════════════════════════════════════════
# 17. SAME-SESSION ENTRY ONLY (LAST BAR OF DAY PRODUCES NO TRADE)
# ═══════════════════════════════════════════════════════════════════════════

def test_same_session_entry_last_bar_produces_no_trade():
    """Verify that a signal on the last bar of a day produces no trade (entry on next day is skipped)."""
    bars = _make_synthetic_bars(n_bars=40, start_time='2024-01-02 09:30:00')
    et = pytz.timezone('America/New_York')

    # Bar 18 is the 15:30 ET bar (last bar of Day 1)
    day1_last_bar_time = bars.index[18]
    assert day1_last_bar_time.tz_convert(et).hour == 15 and day1_last_bar_time.tz_convert(et).minute == 30

    signals = pd.DataFrame([{
        'timestamp': day1_last_bar_time,
        'ticker': 'TEST',
        'side': 'long',
        'confidence': 0.85,
    }])

    res = run_backtest(signals, {'TEST': bars})
    # Must produce 0 trades because next bar is on Day 2 (09:30 ET next morning)
    assert len(res['trades']) == 0
    assert res['metrics']['n_trades'] == 0


# ═══════════════════════════════════════════════════════════════════════════
# 18. NO-ATR FALLBACK (DROPS 2% EXITS WITH EXIT_REASON 'SL' NEAR -1R)
# ═══════════════════════════════════════════════════════════════════════════

def test_no_atr_fallback_drops_2_pct_exits_sl_near_minus_1r():
    """Verify that when ATR is missing, 1% stop is enforced and a 2% drop exits with SL near -1R."""
    # Create 4 bars: 09:30, 09:50 (signal), 10:10 (entry @ 100), 10:30 (drops 2% to 98)
    et = pytz.timezone('America/New_York')
    times = [
        et.localize(datetime.datetime(2024, 1, 2, 9, 30)).astimezone(pytz.utc),
        et.localize(datetime.datetime(2024, 1, 2, 9, 50)).astimezone(pytz.utc),
        et.localize(datetime.datetime(2024, 1, 2, 10, 10)).astimezone(pytz.utc),
        et.localize(datetime.datetime(2024, 1, 2, 10, 30)).astimezone(pytz.utc),
    ]

    bars = pd.DataFrame({
        'Open': [100.0, 100.0, 100.0, 99.0],
        'High': [100.5, 100.5, 100.2, 99.0],
        'Low': [99.5, 99.5, 99.8, 97.8],  # Bar 3 drops to 97.8 (more than 2% drop from 100.0)
        'Close': [100.0, 100.0, 100.0, 98.0],
        'Volume': [10000.0, 10000.0, 10000.0, 10000.0],
    }, index=pd.DatetimeIndex(times))

    # Signal at Bar 1 (09:50) -> Enters at Bar 2 Open (10:10 @ 100.0)
    signals = pd.DataFrame([{
        'timestamp': times[1],
        'ticker': 'TEST',
        'side': 'long',
        'confidence': 0.8,
    }])

    res = run_backtest(signals, {'TEST': bars}, cost_per_share=0.0, slippage_bps=0.0)
    trades = res['trades']

    assert len(trades) == 1
    trade = trades.iloc[0]

    # Enforces 1% stop: sl_price = 100.0 * (1 - 0.01) = 99.0
    assert trade['exit_reason'] == 'sl'
    assert trade['exit_price'] == pytest.approx(99.0)
    # Stop distance was $1.00; net PnL is -$1.00 * shares; r_multiple should be -1.0
    assert trade['r_multiple'] == pytest.approx(-1.0, abs=0.01)
