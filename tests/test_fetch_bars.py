"""Offline tests for fetch_bars.py (Alpaca streaming is mocked)."""
import joblib
import numpy as np
import pandas as pd
import pytest

import fetch_bars


def _fake_5min(start_utc, days=15):
    """Synthetic regular-session 5-min bars (09:30–16:00 ET) for business days."""
    idx = []
    for day in pd.bdate_range(start_utc.tz_convert('America/New_York').date(), periods=days):
        session = pd.date_range(f'{day.date()} 09:30', f'{day.date()} 15:55',
                                freq='5min', tz='America/New_York')
        idx.extend(session.tz_convert('UTC'))
    idx = pd.DatetimeIndex(idx)
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.2, len(idx)))
    df = pd.DataFrame({'open': close, 'high': close + 0.3, 'low': close - 0.3,
                       'close': close, 'volume': 1000.0}, index=idx)
    df.attrs['feed'] = 'sip'
    return df


def _write_signals(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_read_signals_validates_columns_and_sides(tmp_path):
    bad_cols = _write_signals(tmp_path / 'a.csv', [{'timestamp': '2025-03-03T15:00:00Z', 'ticker': 'AAPL'}])
    with pytest.raises(ValueError, match='missing columns'):
        fetch_bars.read_signals(bad_cols)

    bad_side = _write_signals(tmp_path / 'b.csv', [
        {'timestamp': '2025-03-03T15:00:00Z', 'ticker': 'AAPL', 'side': 'buy'}])
    with pytest.raises(ValueError, match='Invalid side'):
        fetch_bars.read_signals(bad_side)


def test_read_signals_assumes_utc_for_naive_timestamps(tmp_path):
    path = _write_signals(tmp_path / 's.csv', [
        {'timestamp': '2025-03-03 15:00:00', 'ticker': ' aapl ', 'side': 'LONG'}])
    signals = fetch_bars.read_signals(path)
    assert str(signals['timestamp'].dt.tz) == 'UTC'
    assert signals.loc[0, 'ticker'] == 'AAPL'
    assert signals.loc[0, 'side'] == 'long'


def test_fetch_saves_backtest_ready_cache_and_survives_bad_ticker(tmp_path, monkeypatch):
    calls = []

    def fake_stream(ticker, client=None, feed=None, start_dt=None, **kwargs):
        calls.append((ticker, start_dt))
        if ticker == 'BAD':
            raise RuntimeError('NO bars for BAD')
        return _fake_5min(pd.Timestamp(start_dt))

    monkeypatch.setattr(fetch_bars.cct, 'stream_alpaca_bars', fake_stream)
    path = _write_signals(tmp_path / 's.csv', [
        {'timestamp': '2025-03-17T15:00:00Z', 'ticker': 'AAPL', 'side': 'long'},
        {'timestamp': '2025-03-18T15:00:00Z', 'ticker': 'BAD', 'side': 'short'},
    ])
    signals = fetch_bars.read_signals(path)
    out = tmp_path / 'bars'
    result = fetch_bars.fetch_bars_for_signals(signals, out, client=object(), feed='sip')

    assert result['saved'] == ['AAPL']
    assert 'BAD' in result['failed']
    # Warm-up: download starts WARMUP_DAYS before the first signal
    assert calls[0][1] <= (pd.Timestamp('2025-03-17T15:00:00Z')
                           - pd.Timedelta(days=fetch_bars.WARMUP_DAYS)).to_pydatetime()

    bars = joblib.load(out / '.cache_AAPL_v3_20m.pkl')
    assert {'Open', 'High', 'Low', 'Close', 'Volume', 'ATR'} <= set(bars.columns)
    assert bars['ATR'].dropna().gt(0).all()
    minutes = set(bars.index.tz_convert('America/New_York').strftime('%H:%M'))
    assert '09:30' in minutes and '15:30' in minutes and '15:50' not in minutes

    # Cached tickers are skipped on rerun (no new download)
    n_calls = len(calls)
    fetch_bars.fetch_bars_for_signals(signals, out, client=object(), feed='sip')
    assert [c[0] for c in calls[n_calls:]] == ['BAD']


def test_cache_is_loadable_by_backtest_engine(tmp_path, monkeypatch):
    from backtest_report import load_bars_from_cache, run_backtest

    monkeypatch.setattr(fetch_bars.cct, 'stream_alpaca_bars',
                        lambda ticker, client=None, feed=None, start_dt=None, **k: _fake_5min(pd.Timestamp(start_dt)))
    path = _write_signals(tmp_path / 's.csv', [
        {'timestamp': '2025-03-17T15:10:00Z', 'ticker': 'AAPL', 'side': 'long'}])
    signals = fetch_bars.read_signals(path)
    fetch_bars.fetch_bars_for_signals(signals, tmp_path / 'bars', client=object(), feed='sip')

    bars = load_bars_from_cache(tmp_path / 'bars', tickers=['AAPL'])
    assert 'AAPL' in bars
    res = run_backtest(signals, bars)
    assert res['metrics']['n_trades'] == 1
