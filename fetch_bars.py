"""
fetch_bars.py — Download 20-minute bars for every ticker in a customer's signals CSV.

Reads a signals CSV (timestamp, ticker, side[, confidence]), streams historical
Alpaca bars covering the signal date range (plus warm-up for ATR), resamples to
09:30-anchored 20-minute bars, adds a day-boundary ATR column, and saves one
``.cache_{TICKER}_v3_20m.pkl`` per ticker — the exact format backtest_report.py
loads with ``--bars-cache``.

Usage:
    python fetch_bars.py --signals customers/c1/signals.csv --out customers/c1/bars
    python fetch_bars.py --signals customers/c1/signals.csv --out customers/c1/bars --run customers/c1/report
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
import pytz

import colab_cloud_trainer as cct
import quant_core

WARMUP_DAYS = 10          # calendar days before the first signal (ATR warm-up)
AFTER_DAYS = 2            # calendar days after the last signal (trade exits)
REQUIRED_COLUMNS = {'timestamp', 'ticker', 'side'}
VALID_SIDES = {'long', 'short', 'flat'}


def read_signals(path: Path) -> pd.DataFrame:
    """Load and validate a signals CSV. Raises ValueError with a clear message."""
    signals = pd.read_csv(path)
    signals.columns = [str(c).strip().lower() for c in signals.columns]
    missing = REQUIRED_COLUMNS - set(signals.columns)
    if missing:
        raise ValueError(f"Signals CSV is missing columns: {', '.join(sorted(missing))}")
    if signals.empty:
        raise ValueError("Signals CSV has no rows.")

    try:
        ts = pd.to_datetime(signals['timestamp'], utc=False)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Unparseable timestamp values: {error}") from error
    if ts.dt.tz is None:
        print("  ⚠ Timestamps have no timezone — assuming UTC. Confirm with the customer.")
        ts = ts.dt.tz_localize('UTC')
    signals['timestamp'] = ts.dt.tz_convert('UTC')

    signals['ticker'] = signals['ticker'].astype(str).str.strip().str.upper()
    signals['side'] = signals['side'].astype(str).str.strip().str.lower()
    bad_sides = sorted(set(signals['side']) - VALID_SIDES)
    if bad_sides:
        raise ValueError(f"Invalid side values {bad_sides}; use long, short or flat.")
    return signals


def summarize_signals(signals: pd.DataFrame) -> Dict[str, Any]:
    """Basic facts about a signals file (printed before downloading)."""
    active = signals[signals['side'].isin({'long', 'short'})]
    return {
        'tickers': sorted(signals['ticker'].unique().tolist()),
        'start': signals['timestamp'].min(),
        'end': signals['timestamp'].max(),
        'n_rows': int(len(signals)),
        'n_active': int(len(active)),
    }


def add_atr(bars_20m: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Attach the same day-boundary ATR the V3+ trainer uses."""
    out = bars_20m.copy()
    et = pytz.timezone('America/New_York')
    session_date = pd.Series(out.index.tz_convert(et).normalize(), index=out.index)
    out['ATR'] = quant_core._day_boundary_atr(
        out['High'], out['Low'], out['Close'], session_date, period=period,
    )
    return out


def fetch_ticker_bars(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    client: Any = None,
    feed: Optional[str] = None,
) -> pd.DataFrame:
    """Stream 5-min bars from ``start`` and return 20-min bars with ATR, trimmed to ``end``."""
    raw = cct.stream_alpaca_bars(
        ticker, client=client, feed=feed, start_dt=start.to_pydatetime(),
    )
    raw = raw.rename(columns={c: str(c).capitalize() for c in raw.columns
                              if str(c).lower() in {'open', 'high', 'low', 'close', 'volume'}})
    bars = quant_core.resample_bars_20min(raw)
    # resample_bars_20min returns naive US/Eastern wall-clock times; restore UTC so
    # bars line up with UTC signal timestamps in backtest_report.run_backtest.
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize('America/New_York', ambiguous='infer').tz_convert('UTC')
    bars = bars[(bars.index >= start) & (bars.index <= end)]
    if bars.empty:
        raise RuntimeError(f"No bars for {ticker} between {start.date()} and {end.date()}.")
    bars = add_atr(bars)
    bars.attrs['feed'] = raw.attrs.get('feed', feed)
    return bars


def fetch_bars_for_signals(
    signals: pd.DataFrame,
    out_dir: Path,
    client: Any = None,
    feed: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Download and save bars for every ticker in ``signals``. Returns a summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    info = summarize_signals(signals)
    start = info['start'] - timedelta(days=WARMUP_DAYS)
    end = info['end'] + timedelta(days=AFTER_DAYS)

    if client is None:
        from alpaca.data.historical import StockHistoricalDataClient
        key, secret = cct._resolve_alpaca_credentials()
        client = StockHistoricalDataClient(key, secret)
        if feed is None:
            feed = cct._probe_alpaca_credentials(client, key, secret)

    saved: List[str] = []
    failed: Dict[str, str] = {}
    for ticker in info['tickers']:
        path = out_dir / f'.cache_{ticker}_v3_20m.pkl'
        if path.exists() and not overwrite:
            print(f"  ✓ {ticker}: already cached ({path.name}), skipping")
            saved.append(ticker)
            continue
        print(f"  📡 {ticker}: downloading {start.date()} → {end.date()}...")
        try:
            bars = fetch_ticker_bars(ticker, start, end, client=client, feed=feed)
        except Exception as error:  # one bad ticker must not stop the others
            failed[ticker] = str(error).splitlines()[0][:200]
            print(f"  ✗ {ticker}: {failed[ticker]}")
            continue
        joblib.dump(bars, path)
        saved.append(ticker)
        print(f"  ✓ {ticker}: {len(bars)} bars saved")

    return {**info, 'saved': saved, 'failed': failed, 'out_dir': out_dir}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download 20-min bars for a signals CSV.")
    parser.add_argument('--signals', required=True, help="Path to the customer's signals CSV.")
    parser.add_argument('--out', required=True, help="Folder to save .cache_{TICKER}_v3_20m.pkl files.")
    parser.add_argument('--feed', choices=['sip', 'iex'], default=None, help="Force a data feed.")
    parser.add_argument('--overwrite', action='store_true', help="Re-download tickers already cached.")
    parser.add_argument('--run', metavar='REPORT_DIR', default=None,
                        help="After downloading, run backtest_report.py and write the report here.")
    args = parser.parse_args(argv)

    try:
        signals = read_signals(Path(args.signals))
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    info = summarize_signals(signals)
    print(f"\nSignals: {info['n_rows']} rows ({info['n_active']} long/short), "
          f"{len(info['tickers'])} tickers, {info['start']} → {info['end']}")
    if info['n_active'] < 100:
        print(f"  ⚠ Only {info['n_active']} long/short signals — the verdict will FAIL "
              f"the 100-trade minimum. Ask the customer for a longer history.")

    try:
        result = fetch_bars_for_signals(signals, Path(args.out), feed=args.feed,
                                        overwrite=args.overwrite)
    except RuntimeError as error:  # e.g. Alpaca authentication failure
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if result['failed']:
        print(f"\n⚠ No data for: {', '.join(result['failed'])}. "
              f"Their signals will be skipped — tell the customer.")
    if not result['saved']:
        print("Error: no bars downloaded for any ticker.", file=sys.stderr)
        return 1

    if args.run:
        from backtest_report import main as backtest_main
        sys.argv = ['backtest_report.py', '--signals', str(args.signals),
                    '--bars-cache', str(args.out), '--out', str(args.run)]
        backtest_main()
    else:
        print(f"\nNext: python backtest_report.py --signals {args.signals} "
              f"--bars-cache {args.out} --out <report_dir>")
    return 0


if __name__ == '__main__':
    sys.exit(main())

