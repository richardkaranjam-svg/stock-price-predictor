"""
colab_cloud_trainer.py — V3+ Free Cloud Training Pipeline.

Run in Google Colab or Kaggle to train the LightGBM + RF ensemble on
2016–Present Alpaca SIP data (52,000+ bars) in RAM without saving raw
CSVs locally.  Exports a portable plain-dict artifact:
  - stock_price_model.pkl  (< 5 MB)
  - model_metadata.json

Resolves Issues: #1, #8, #9, #10, #13, #16, #17

IMPORTANT: This file imports from quant_core.py which must be uploaded
to the Colab/Kaggle runtime alongside this script.

Usage in Colab (pick ONE method):

  Method A — Colab Secrets (recommended, keys never in notebook):
      1. Click the key icon (🔑) in the left sidebar
      2. Add secrets:  APCA_API_KEY_ID  and  APCA_API_SECRET_KEY
      3. Toggle "Notebook access" ON for each
      4. In a code cell:
            import colab_cloud_trainer
            colab_cloud_trainer.main()

  Method B — Inline environment variables in a single cell:
      import os
      os.environ['APCA_API_KEY_ID']     = 'PKxxx'
      os.environ['APCA_API_SECRET_KEY'] = 'xxxxx'
      import colab_cloud_trainer
      colab_cloud_trainer.main()

  Method C — Shell one-liner (env vars on the SAME line):
      !APCA_API_KEY_ID='PKxxx' APCA_API_SECRET_KEY='xxxxx' python colab_cloud_trainer.py

  Feed selection: SIP is auto-detected (historical SIP older than 15 min is
  included in Alpaca's FREE Basic plan, since 2016).  On a subscription-403
  the trainer falls back to IEX and auto-detects IEX's first available bar.
  Force a feed with  os.environ['ALPACA_DATA_FEED'] = 'sip' | 'iex'.
  NOTE: Colab Secrets (userdata) only work in the notebook kernel (Methods A/B),
  not inside a '!python' subprocess — use Method C for subprocess runs.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Ensure quant_core is importable (works in Colab where CWD has both files)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant_core

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TICKER = 'AAPL'
BENCHMARK = 'SPY'
START_YEAR = 2016
BAR_INTERVAL_MINUTES = 5  # Stream 5-min bars, resample to 20-min
STREAM_TIMEFRAME = '5Min'

# Triple-barrier parameters
TP_ATR_MULT = 1.50   # Take-profit multiplier
SL_ATR_MULT = 0.75   # Stop-loss multiplier
MAX_HORIZON_BARS = 12  # Capped to fit within 15:40 ET same day

# Training parameters
N_CV_FOLDS = 5
EMBARGO_BARS = 15
MIN_CONFIDENCE = 0.58
SLIPPAGE_BPS = 3  # 3 bps round-trip

# LightGBM defaults (tuned per-fold via uniqueness)
LGB_PARAMS = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.80,
    'min_child_samples': 50,
    'n_estimators': 500,
    'verbose': -1,
    'random_state': 42,
}

# RF defaults
RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 12,
    'min_samples_leaf': 50,
    'n_jobs': -1,
    'random_state': 42,
}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: BIAS-FREE 2016–PRESENT SIP STREAMING  (Issue #1, #13)
# ═══════════════════════════════════════════════════════════════════════════

# Earliest year with history per feed.  SIP history starts 2016 on BOTH the
# free Basic plan and Algo Trader Plus (Basic only blocks the latest 15 min).
# IEX history starts ~2020.  These are fallback HINTS only — the real
# boundary is discovered at runtime by _discover_first_bar().
_FEED_HISTORY_HINT_YEAR = {'sip': 2016, 'iex': 2020}

MAX_CHUNK_ATTEMPTS = 5            # per 30-day chunk (on top of alpaca-py's own 429 retries)
RETRY_BASE_DELAY_S = 2.0          # exponential backoff base (tests set this to 0)
MAX_FAILED_CHUNK_FRACTION = 0.05  # abort if > 5% of chunks are permanently lost
_RETRYABLE_KINDS = ('rate_limit', 'transient')
_INVISIBLE_CHARS = ('\u200b', '\u200c', '\u200d', '\u2060', '\ufeff', '\u00a0')


def _api_status_code(exc: BaseException) -> Optional[int]:
    """Best-effort HTTP status extraction from alpaca-py / requests errors.

    ``str(alpaca.common.exceptions.APIError)`` is ONLY the JSON response body
    (e.g. ``'{"message": "forbidden."}'``) — it never contains the numeric
    status, so substring checks like ``'401' in str(e)`` silently never match.
    """
    code = getattr(exc, 'status_code', None)
    if code is None:
        code = getattr(getattr(exc, 'response', None), 'status_code', None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def _classify_alpaca_error(exc: BaseException) -> str:
    """Classify a failure: 'auth' | 'subscription' | 'rate_limit' | 'transient' | 'other'.

    Alpaca market data answers BAD / MISSING credentials with HTTP 403
    ("forbidden.") as well as 401, while a plan restriction is a 403 whose
    body mentions the subscription ("subscription does not permit querying
    recent SIP data").  Status code AND body must both be inspected.
    """
    status = _api_status_code(exc)
    msg = str(exc).lower()
    is_subscription_msg = 'subscription' in msg or 'permit' in msg

    if status == 401:
        return 'auth'
    if status == 403:
        return 'subscription' if is_subscription_msg else 'auth'
    if status == 429:
        return 'rate_limit'
    if status is not None and 500 <= status < 600:
        return 'transient'
    if status is None:
        net_errors: tuple = (ConnectionError, TimeoutError)
        try:
            import requests
            net_errors += (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
        except ImportError:
            pass
        if isinstance(exc, net_errors):
            return 'transient'
        if is_subscription_msg:
            return 'subscription'
        if 'unauthorized' in msg or 'forbidden' in msg or 'authorization required' in msg:
            return 'auth'
        if 'too many requests' in msg or 'rate limit' in msg:
            return 'rate_limit'
    return 'other'


def _call_with_retry(fn, what: str, max_attempts: Optional[int] = None):
    """Call ``fn()``; retry ONLY rate-limit / transient errors with exponential
    backoff.  Auth, subscription and other errors are re-raised immediately."""
    max_attempts = max_attempts or MAX_CHUNK_ATTEMPTS
    delay = RETRY_BASE_DELAY_S
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            kind = _classify_alpaca_error(e)
            if kind not in _RETRYABLE_KINDS or attempt == max_attempts:
                raise
            status = _api_status_code(e)
            print(f"  ↻ {what}: {kind} (HTTP {status or type(e).__name__}) "
                  f"— retry {attempt}/{max_attempts - 1} in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)


def _clean_secret(value: Any) -> str:
    """Strip whitespace, wrapping quotes and invisible copy-paste characters."""
    if value is None:
        return ''
    s = str(value)
    for ch in _INVISIBLE_CHARS:
        s = s.replace(ch, '')
    return s.strip().strip('"').strip("'").strip()


def _resolve_alpaca_credentials() -> Tuple[str, str]:
    """Resolve keys from ``os.environ`` → Colab Secrets, sanitize them, and
    write them back to ``os.environ`` so the SPY stream and any ``!python``
    subprocess launched from the notebook inherit the clean values."""
    env_key = _clean_secret(os.environ.get('APCA_API_KEY_ID'))
    env_secret = _clean_secret(os.environ.get('APCA_API_SECRET_KEY'))
    key, secret = env_key, env_secret
    source = 'os.environ / shell'

    # Colab Secrets take PRIORITY over os.environ.  os.environ persists for the
    # whole kernel session (an old cell, or this function's own write-back
    # below, can leave STALE keys there), whereas Secrets are what the user
    # just edited in the sidebar.
    try:
        from google.colab import userdata  # type: ignore
    except ImportError:
        userdata = None  # Not inside a Colab notebook kernel
    if userdata is not None:
        try:
            sec_key = _clean_secret(userdata.get('APCA_API_KEY_ID'))
            sec_secret = _clean_secret(userdata.get('APCA_API_SECRET_KEY'))
            if sec_key and sec_secret:
                if (env_key and env_key != sec_key) or (env_secret and env_secret != sec_secret):
                    print(f"  ⚠ os.environ held DIFFERENT (stale) Alpaca keys "
                          f"(key …{env_key[-4:]}) than Colab Secrets "
                          f"(key …{sec_key[-4:]}) — using Colab Secrets.")
                key, secret = sec_key, sec_secret
                source = 'Colab Secrets (google.colab.userdata)'
        except Exception as e:  # SecretNotFoundError / NotebookAccessError
            if not (env_key and env_secret):
                print(f"  ⚠ Colab Secrets lookup failed ({type(e).__name__}: {e}).\n"
                      f"    → Both secrets must exist AND 'Notebook access' must be ON.\n"
                      f"    → userdata.get() only works in the notebook kernel; for\n"
                      f"      '!python colab_cloud_trainer.py' pass keys as env vars.")

    if not key or not secret:
        raise RuntimeError(
            'Alpaca credentials not found.  Use ONE of these methods:\n'
            '\n'
            '  1. Colab Secrets (recommended):\n'
            '     Click 🔑 in the left sidebar → add APCA_API_KEY_ID\n'
            '     and APCA_API_SECRET_KEY → toggle "Notebook access" ON\n'
            '\n'
            '  2. In a notebook cell BEFORE calling main():\n'
            '     import os\n'
            '     os.environ["APCA_API_KEY_ID"]     = "PKxxx"\n'
            '     os.environ["APCA_API_SECRET_KEY"] = "xxxxx"\n'
            '\n'
            '  3. Shell one-liner (keys on the SAME line):\n'
            '     !APCA_API_KEY_ID="PKxxx" APCA_API_SECRET_KEY="xxxxx" '
            'python colab_cloud_trainer.py\n'
        )

    os.environ['APCA_API_KEY_ID'] = key
    os.environ['APCA_API_SECRET_KEY'] = secret
    print(f"  🔐 Credentials from {source}: key {key[:2]}…{key[-4:]} "
          f"(len={len(key)}), secret len={len(secret)}")
    return key, secret


def _discover_first_bar(client, ticker: str, feed, start_dt: datetime,
                        end_dt: datetime) -> Optional[pd.Timestamp]:
    """UTC timestamp of the earliest DAILY bar for ``ticker`` on ``feed`` in
    [start_dt, end_dt], or None.  One tiny request (bars are ascending)."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import Adjustment

    req = StockBarsRequest(
        symbol_or_symbols=ticker, timeframe=TimeFrame(1, TimeFrameUnit.Day),
        start=start_dt, end=end_dt, adjustment=Adjustment.ALL, feed=feed, limit=1,
    )
    bars = _call_with_retry(lambda: client.get_stock_bars(req),
                            f"{ticker} first-bar discovery ({feed.value})")
    bar_list = (getattr(bars, 'data', None) or {}).get(ticker) or []
    if not bar_list:
        return None
    ts = pd.Timestamp(bar_list[0].timestamp)
    return ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')


def _bars_to_frame(bars, ticker: str) -> Optional[pd.DataFrame]:
    """alpaca-py BarSet → OHLCV DataFrame with a tz-aware UTC index (or None)."""
    bar_list = (getattr(bars, 'data', None) or {}).get(ticker) or []
    if not bar_list:
        return None
    idx = pd.DatetimeIndex(pd.to_datetime([b.timestamp for b in bar_list], utc=True),
                           name='timestamp')
    return pd.DataFrame({
        'Open': [float(b.open) for b in bar_list],
        'High': [float(b.high) for b in bar_list],
        'Low': [float(b.low) for b in bar_list],
        'Close': [float(b.close) for b in bar_list],
        'Volume': [float(b.volume) for b in bar_list],
    }, index=idx)


def _probe_alpaca_credentials(client, key: str, secret: str) -> str:
    """Confirm credentials and detect the best feed BEFORE the monthly loop.

    Returns ``'sip'`` or ``'iex'``.  Raises RuntimeError on an auth failure
    (HTTP 401, or HTTP 403 *without* subscription wording — Alpaca market
    data uses 403 "forbidden." for bad keys) so we never stream with bad keys.

    SIP history older than 15 minutes is included in the FREE Basic plan, so
    the SIP probe normally succeeds; a subscription-403 falls back to IEX.
    Set env ``ALPACA_DATA_FEED=sip|iex`` to force a feed.
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import Adjustment, DataFeed

    # Fixed, closed historical window (a Tue–Wed in 2023) — never "recent".
    probe_start = datetime(2023, 1, 3, tzinfo=timezone.utc)
    probe_end = datetime(2023, 1, 5, tzinfo=timezone.utc)

    def _probe(feed):
        req = StockBarsRequest(
            symbol_or_symbols='AAPL', timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=probe_start, end=probe_end, adjustment=Adjustment.ALL, feed=feed,
        )
        return _call_with_retry(lambda: client.get_stock_bars(req),
                                f"credential probe ({feed.value})", max_attempts=3)

    # ── Step 1: confirm credentials on IEX (every account has IEX) ──
    try:
        _probe(DataFeed.IEX)
        print(f"  🔑 Credentials OK  (key …{key[-6:]}  len={len(key)})")
    except Exception as e:
        kind = _classify_alpaca_error(e)
        if kind == 'auth':
            raise RuntimeError(_auth_failure_message(key, secret, e)) from e
        if kind in _RETRYABLE_KINDS:
            raise RuntimeError(
                f"Alpaca market-data API unreachable after retries "
                f"(HTTP {_api_status_code(e)}): {e}.  Check network / "
                f"https://status.alpaca.markets and re-run."
            ) from e
        # A subscription error on IEX is unexpected; SIP probe decides below.
        print(f"  ⚠ IEX probe returned a non-auth error ({kind}): {e}")

    forced = os.environ.get('ALPACA_DATA_FEED', '').strip().lower()
    if forced in ('sip', 'iex'):
        print(f"  📌 ALPACA_DATA_FEED={forced} — skipping SIP auto-detection.")
        return forced

    # ── Step 2: check SIP access (historical SIP is free on Basic plan) ──
    try:
        _probe(DataFeed.SIP)
        print("  📡 SIP feed confirmed — full consolidated volume, history since 2016.")
        return 'sip'
    except Exception as e:
        kind = _classify_alpaca_error(e)
        if kind == 'auth':
            raise RuntimeError(_auth_failure_message(key, secret, e)) from e
        print(f"  ℹ️  SIP not available ({kind}, HTTP {_api_status_code(e)}): {e}\n"
              f"      → Falling back to IEX (history begins ~2020; exact start "
              f"auto-detected).")
        return 'iex'


def _auth_failure_message(key: str, secret: str, exc: BaseException) -> str:
    """Boxed, actionable auth-failure message (never prints full keys)."""
    status = _api_status_code(exc)
    return (
        f"\n"
        f"╔══════════════════════════════════════════════════════╗\n"
        f"║    ALPACA AUTHENTICATION FAILED  (HTTP {str(status or '?'):>3})          ║\n"
        f"╠══════════════════════════════════════════════════════╣\n"
        f"║  Your API keys are being REJECTED by Alpaca.         ║\n"
        f"║  (market data returns 403 'forbidden.' OR 401)       ║\n"
        f"║                                                      ║\n"
        f"║  Key ID : …{key[-6:]:>6}  (total length: {len(key):>3})         ║\n"
        f"║  Secret : …{secret[-6:]:>6}  (total length: {len(secret):>3})         ║\n"
        f"║                                                      ║\n"
        f"║  Most common causes:                                 ║\n"
        f"║  1. Wrong key — go to app.alpaca.markets →           ║\n"
        f"║     Paper Trading → API Keys → regenerate            ║\n"
        f"║  2. Paper account was reset (keys must be re-made)   ║\n"
        f"║  3. Key ID / Secret swapped or truncated on paste    ║\n"
        f"║  4. Key was revoked or belongs to a deleted account  ║\n"
        f"╚══════════════════════════════════════════════════════╝\n"
        f"Original error: {exc}"
    )


def stream_alpaca_bars(
    ticker: str,
    start_year: int = START_YEAR,
    timeframe: str = STREAM_TIMEFRAME,
    client: Any = None,
    feed: Optional[str] = None,
    start_dt: Optional[datetime] = None,
) -> pd.DataFrame:
    """Stream historical bars from Alpaca Market Data API (SIP → IEX fallback).

    Uses ``Adjustment.ALL`` for split/dividend-adjusted prices and streams in
    30-day chunks (bounded RAM on free Colab).  Historical SIP is free on the
    Basic plan as long as ``end <= now - 15 min``.

    Robustness guarantees:
      * Auth failures (401, or 403 without subscription wording) abort at once.
      * 429 / 5xx / network errors are retried with exponential backoff; a
        chunk is only dropped after ``MAX_CHUNK_ATTEMPTS`` and the run aborts
        if more than ``MAX_FAILED_CHUNK_FRACTION`` of chunks are lost.
      * A subscription-403 mid-stream switches to IEX ONCE, discards the
        already-streamed SIP chunks (no mixed-feed volume splice) and
        restarts from the auto-detected first IEX bar.
      * Empty months (pre-history, holidays) are skipped, never fatal.

    Parameters
    ----------
    ticker : str
        Stock symbol (e.g., 'AAPL').
    start_year : int
        First year of data to fetch (ignored if ``start_dt`` is given).
    timeframe : str
        Bar interval (default '5Min').
    client : StockHistoricalDataClient, optional
        Re-use an authenticated client (skips credential resolution/probe).
    feed : {'sip', 'iex'}, optional
        Force a feed (used by ``main`` to stream AAPL & SPY on the SAME feed).
    start_dt : datetime, optional
        Exact tz-aware start (used to align AAPL & SPY to a common start).

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with tz-aware UTC DatetimeIndex.  ``df.attrs['feed']``
        records the feed actually used.
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import Adjustment, DataFeed

    if client is None:
        key, secret = _resolve_alpaca_credentials()
        client = StockHistoricalDataClient(key, secret)
        if feed is None:
            # ── Upfront credential + feed probe (fail fast before the loop) ──
            print("  🔍 Probing Alpaca credentials and feed access...")
            feed = _probe_alpaca_credentials(client, key, secret)
    best_feed = (feed or 'sip').lower()

    # Parse timeframe
    tf_map = {
        '1Min': TimeFrame(1, TimeFrameUnit.Minute),
        '5Min': TimeFrame(5, TimeFrameUnit.Minute),
        '15Min': TimeFrame(15, TimeFrameUnit.Minute),
        '1Hour': TimeFrame(1, TimeFrameUnit.Hour),
        '1Day': TimeFrame(1, TimeFrameUnit.Day),
    }
    tf = tf_map.get(timeframe, TimeFrame(5, TimeFrameUnit.Minute))

    # End time: now - 16 minutes (Basic plan blocks only the latest 15 min of SIP)
    end_dt = datetime.now(timezone.utc) - timedelta(minutes=16)

    requested_start = start_dt or datetime(start_year, 1, 1, tzinfo=timezone.utc)
    if requested_start.tzinfo is None:
        requested_start = requested_start.replace(tzinfo=timezone.utc)

    def _feed_start(feed_enum) -> datetime:
        """Fast-forward past empty pre-history months: ask Alpaca for the
        first daily bar on this feed instead of trusting a hard-coded year."""
        try:
            first = _discover_first_bar(client, ticker, feed_enum, requested_start, end_dt)
        except Exception as e:
            kind = _classify_alpaca_error(e)
            if kind in ('auth', 'subscription'):
                raise  # handled by the caller's feed-switch / auth logic
            first = None
            hint = _FEED_HISTORY_HINT_YEAR.get(feed_enum.value, requested_start.year)
            print(f"  ⚠ First-bar discovery failed ({kind}: {e}); "
                  f"using {hint}-01-01 hint.")
            return max(requested_start, datetime(hint, 1, 1, tzinfo=timezone.utc))
        if first is None:
            raise RuntimeError(
                f"Alpaca {feed_enum.value.upper()} feed has NO bars for {ticker} "
                f"between {requested_start.date()} and {end_dt.date()}.  "
                f"Check the symbol (it may be delisted or misspelled)."
            )
        # Midnight UTC of the first bar's day (never earlier than requested).
        first_day = datetime(first.year, first.month, first.day, tzinfo=timezone.utc)
        if first_day > requested_start + timedelta(days=7):
            print(f"  ⏭️  {ticker}: earliest {feed_enum.value.upper()} bar is "
                  f"{first_day.date()} — fast-forwarding from {requested_start.date()}.")
        return max(requested_start, first_day)

    active_feed = DataFeed.SIP if best_feed == 'sip' else DataFeed.IEX
    feed_switched = (best_feed == 'iex')   # already on IEX → never switch again
    try:
        effective_start = _feed_start(active_feed)
    except Exception as e:
        if active_feed == DataFeed.SIP and _classify_alpaca_error(e) == 'subscription':
            print(f"  ℹ️  SIP subscription error for {ticker} — switching to IEX.")
            active_feed, feed_switched = DataFeed.IEX, True
            effective_start = _feed_start(active_feed)
        elif _classify_alpaca_error(e) == 'auth':
            raise RuntimeError(
                f"Alpaca rejected credentials while streaming {ticker} "
                f"(HTTP {_api_status_code(e)}).  Regenerate keys at "
                f"app.alpaca.markets and restart.\nOriginal error: {e}"
            ) from e
        else:
            raise

    all_chunks: list[pd.DataFrame] = []
    failed_chunks: list[str] = []
    n_chunks_attempted = 0
    chunk_start = effective_start
    last_error: Optional[str] = None

    print(f"📡 Streaming {ticker} {timeframe} bars from {effective_start.date()} to now "
          f"({active_feed.value.upper()} feed)...")

    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=30), end_dt)
        label = chunk_start.strftime('%Y-%m-%d')
        n_chunks_attempted += 1

        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=tf,
            start=chunk_start,
            end=chunk_end,
            adjustment=Adjustment.ALL,
            feed=active_feed,
            limit=None,
        )
        try:
            bars = _call_with_retry(lambda: client.get_stock_bars(request),
                                    f"{ticker} {label}")
        except Exception as e:
            last_error = str(e)
            kind = _classify_alpaca_error(e)

            # ── Auth: keys rejected mid-stream — fail immediately ──
            if kind == 'auth':
                raise RuntimeError(
                    f"Alpaca rejected credentials (HTTP {_api_status_code(e)}) while "
                    f"streaming {ticker} {label}.  Regenerate keys at "
                    f"app.alpaca.markets and restart.\nOriginal error: {e}"
                ) from e

            # ── Subscription-403 on SIP: switch to IEX ONCE and restart ──
            # Previously-streamed SIP chunks are discarded so the series never
            # splices SIP volume (~100% of tape) with IEX volume (~2-3%).
            if kind == 'subscription' and active_feed == DataFeed.SIP and not feed_switched:
                print(f"  ℹ️  SIP subscription error at {label} — restarting "
                      f"{ticker} on DataFeed.IEX (discarding {len(all_chunks)} SIP chunks).")
                active_feed, feed_switched = DataFeed.IEX, True
                all_chunks.clear()
                failed_chunks.clear()
                n_chunks_attempted = 0
                chunk_start = _feed_start(active_feed)
                continue
            if kind == 'subscription':
                raise RuntimeError(
                    f"Alpaca subscription error on {active_feed.value.upper()} for "
                    f"{ticker} {label} — no further fallback available.\n"
                    f"Original error: {e}"
                ) from e

            # ── Retries exhausted / unknown error: record the gap loudly ──
            failed_chunks.append(label)
            print(f"  ✖ {label}: chunk LOST after retries ({kind}): {e}")
            chunk_start = chunk_end
            continue

        chunk_df = _bars_to_frame(bars, ticker)
        if chunk_df is not None and not chunk_df.empty:
            all_chunks.append(chunk_df)
            print(f"  {chunk_start.strftime('%Y-%m')}: {len(chunk_df)} bars "
                  f"({active_feed.value.upper()})")
        # Empty chunk (holiday stretch / pre-history) is NOT an error.

        chunk_start = chunk_end
        time.sleep(0.25)  # Rate limit courtesy (Basic plan: 200 req/min)

    if not all_chunks:
        detail = f" Last error encountered: {last_error}" if last_error else ""
        raise RuntimeError(
            f"No data received for {ticker} from Alpaca "
            f"({active_feed.value.upper()}, {effective_start.date()}–now).{detail}\n"
            f"Check your API keys and ensure your Alpaca paper/live account is active."
        )

    if failed_chunks:
        frac = len(failed_chunks) / max(1, n_chunks_attempted)
        msg = (f"{ticker}: {len(failed_chunks)}/{n_chunks_attempted} chunks lost "
               f"({frac:.1%}): {', '.join(failed_chunks[:10])}"
               f"{' …' if len(failed_chunks) > 10 else ''}")
        if frac > MAX_FAILED_CHUNK_FRACTION:
            raise RuntimeError(
                f"{msg}\nExceeds MAX_FAILED_CHUNK_FRACTION="
                f"{MAX_FAILED_CHUNK_FRACTION:.0%} — refusing to train on a "
                f"holey dataset.  Last error: {last_error}"
            )
        warnings.warn(msg)
        print(f"  ⚠ {msg}")

    combined = pd.concat(all_chunks).sort_index()
    combined = combined[~combined.index.duplicated(keep='last')]
    combined.attrs['feed'] = active_feed.value

    print(f"✅ {ticker}: {len(combined)} total {timeframe} bars "
          f"({combined.index[0].date()} → {combined.index[-1].date()}, "
          f"{active_feed.value.upper()})")
    return combined


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: SYMMETRIC PATH-AWARE TRIPLE-BARRIER LABELING  (Issue #8, #9)
# ═══════════════════════════════════════════════════════════════════════════

def triple_barrier_label(
    bars_20m: pd.DataFrame,
    atr: pd.Series,
    session_date: pd.Series,
    tp_mult: float = TP_ATR_MULT,
    sl_mult: float = SL_ATR_MULT,
    max_horizon: int = MAX_HORIZON_BARS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric path-aware High/Low ATR triple-barrier labeling.

    For each bar i (entries up to 15:30 ET), evaluates BOTH:
      - **Long path**: +tp_mult×ATR TP / -sl_mult×ATR SL
      - **Short path**: -tp_mult×ATR TP / +sl_mult×ATR SL

    Checks intrabar High and Low with **pessimistic tie-breaking**:
    if both TP and SL are touched in the same bar, SL wins.

    Label assignment:
      +1 if Long TP hits first (and Short doesn't)
      -1 if Short TP hits first (and Long doesn't)
       0 otherwise (both expire, or ambiguous)

    Horizon is capped at min(max_horizon, bars until 15:40 ET same day).

    Parameters
    ----------
    bars_20m : pd.DataFrame
        20-minute OHLCV bars.
    atr : pd.Series
        ATR values aligned to bars_20m index.
    session_date : pd.Series
        Normalized trading date for each bar.
    tp_mult, sl_mult : float
        ATR multipliers for take-profit and stop-loss.
    max_horizon : int
        Maximum forward bars to evaluate.

    Returns
    -------
    tuple of (labels, t1_indices)
        - labels: int array of {-1, 0, +1}
        - t1_indices: int array of actual end-of-label index for each sample
    """
    n = len(bars_20m)
    close = bars_20m['Close'].values.astype(np.float64)
    high = bars_20m['High'].values.astype(np.float64)
    low = bars_20m['Low'].values.astype(np.float64)
    atr_vals = atr.values.astype(np.float64)
    dates = session_date.values

    # Compute minutes_since_open for entry cutoff
    idx = bars_20m.index
    try:
        hours = idx.hour
        minutes = idx.minute
    except AttributeError:
        hours = np.array([t.hour for t in idx])
        minutes = np.array([t.minute for t in idx])
    mins_since_open = (hours - 9) * 60 + (minutes - 30)

    labels = np.zeros(n, dtype=np.int64)
    t1_indices = np.zeros(n, dtype=np.int64)

    for i in range(n):
        # Entry cutoff: must be <= 15:30 ET (360 mins since open)
        if mins_since_open[i] > 360:
            labels[i] = 0
            t1_indices[i] = i
            continue

        if np.isnan(atr_vals[i]) or atr_vals[i] <= 0:
            labels[i] = 0
            t1_indices[i] = i
            continue

        entry_price = close[i]
        atr_i = atr_vals[i]
        long_tp = entry_price + tp_mult * atr_i
        long_sl = entry_price - sl_mult * atr_i
        short_tp = entry_price - tp_mult * atr_i
        short_sl = entry_price + sl_mult * atr_i

        long_result = 0   # 0=no hit, +1=TP, -1=SL
        short_result = 0
        long_hit_bar = max_horizon + 1
        short_hit_bar = max_horizon + 1
        actual_t1 = i

        for j in range(i + 1, min(i + max_horizon + 1, n)):
            # Same-day constraint: stop at 15:40 ET
            if dates[j] != dates[i]:
                break
            if mins_since_open[j] > 370:  # 15:40 ET
                break

            actual_t1 = j

            # --- Long path evaluation ---
            if long_result == 0:
                # Pessimistic: check SL first (low), then TP (high)
                if low[j] <= long_sl:
                    long_result = -1
                    long_hit_bar = j - i
                elif high[j] >= long_tp:
                    long_result = 1
                    long_hit_bar = j - i

            # --- Short path evaluation ---
            if short_result == 0:
                # Pessimistic: check SL first (high), then TP (low)
                if high[j] >= short_sl:
                    short_result = -1
                    short_hit_bar = j - i
                elif low[j] <= short_tp:
                    short_result = 1
                    short_hit_bar = j - i

            # Both paths resolved
            if long_result != 0 and short_result != 0:
                break

        t1_indices[i] = actual_t1

        # --- Label assignment ---
        if long_result == 1 and short_result != 1:
            labels[i] = 1   # Long TP hit, Short didn't → UP
        elif short_result == 1 and long_result != 1:
            labels[i] = -1  # Short TP hit, Long didn't → DOWN
        elif long_result == 1 and short_result == 1:
            # Both TPs hit — earlier one wins
            if long_hit_bar < short_hit_bar:
                labels[i] = 1
            elif short_hit_bar < long_hit_bar:
                labels[i] = -1
            else:
                labels[i] = 0  # Same bar — ambiguous → CHOP
        else:
            labels[i] = 0   # Neither TP hit → CHOP

    return labels, t1_indices


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: PURGED & EMBARGOED WALK-FORWARD CV  (Issue #10, #17)
# ═══════════════════════════════════════════════════════════════════════════

class PurgedWalkForwardCV:
    """Expanding-window walk-forward cross-validator with purge & embargo.

    Ensures zero information leakage between train and test folds by
    removing any training sample whose label horizon overlaps with the
    test set's start, plus an additional embargo buffer.

    Parameters
    ----------
    n_splits : int
        Number of walk-forward folds.
    embargo_bars : int
        Number of bars to embargo between train and test.
    """

    def __init__(self, n_splits: int = N_CV_FOLDS, embargo_bars: int = EMBARGO_BARS):
        self.n_splits = n_splits
        self.embargo_bars = embargo_bars

    def split(
        self,
        n_samples: int,
        t1_indices: np.ndarray,
    ):
        """Generate purged train/test index pairs.

        Parameters
        ----------
        n_samples : int
            Total number of samples.
        t1_indices : np.ndarray
            Label horizon end indices.

        Yields
        ------
        tuple of (train_indices, test_indices)
        """
        t1_indices = np.asarray(t1_indices)
        if len(t1_indices) != n_samples:
            raise ValueError(f"t1_indices length {len(t1_indices)} != n_samples {n_samples}")
        test_size = n_samples // (self.n_splits + 1)
        if test_size <= self.embargo_bars:
            raise ValueError(
                f"Dataset too small for {self.n_splits}-fold purged CV: "
                f"{n_samples} samples → test_size={test_size} <= embargo={self.embargo_bars}."
            )

        for fold in range(self.n_splits):
            test_start = test_size * (fold + 1)
            # Last fold absorbs the remainder so no tail samples are silently unused.
            test_end = n_samples if fold == self.n_splits - 1 else min(test_start + test_size, n_samples)
            test_idx = np.arange(test_start, test_end)
            if len(test_idx) == 0:
                print(f"  ⚠ CV fold {fold + 1}: empty test split — skipped")
                continue

            # Purge: remove training samples whose labels overlap test start
            purge_cutoff = test_start - self.embargo_bars
            train_mask = t1_indices[:test_start] < purge_cutoff
            train_idx = np.where(train_mask)[0]

            if len(train_idx) == 0:
                print(f"  ⚠ CV fold {fold + 1}: empty train split after purge — skipped")
                continue

            yield train_idx, test_idx


def _align_proba_to_classes(model: Any, proba: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Place ``predict_proba`` columns at their TRUE encoded class (0=DOWN,
    1=CHOP, 2=UP) using ``model.classes_``.

    Right-padding zeros is WRONG when a class is missing from a fold: with
    ``classes_=[1, 2]`` RF returns 2 columns (CHOP, UP) and LightGBM (with
    ``num_class=3``) returns 3 columns whose first two are (CHOP, UP) —
    padding would silently relabel them as (DOWN, CHOP).
    """
    proba = np.asarray(proba, dtype=np.float64)
    classes = np.asarray(getattr(model, 'classes_', np.arange(proba.shape[1]))).astype(int)
    out = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
    for col, cls in enumerate(classes[:proba.shape[1]]):
        if 0 <= cls < n_classes:
            out[:, cls] = proba[:, col]
    row_sum = out.sum(axis=1, keepdims=True)
    return np.divide(out, row_sum, out=np.full_like(out, 1.0 / n_classes), where=row_sum > 0)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def train_v3_model(
    features: pd.DataFrame,
    labels: np.ndarray,
    t1_indices: np.ndarray,
    uniqueness: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train the V3+ LightGBM + RF ensemble with purged walk-forward CV.

    Inside each training fold:
      - Split first 85% for tree fitting, last 15% for calibration
      - Compute per-fold sample weights using information-clock weighting
      - Dynamically set RF max_samples and LGB subsample from mean uniqueness
      - Fit Platt calibration on the calibration split

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix (n_samples × 31).
    labels : np.ndarray
        Integer labels {-1, 0, +1}.
    t1_indices : np.ndarray
        Label horizon end indices.
    uniqueness : np.ndarray
        Per-sample uniqueness values.

    Returns
    -------
    tuple of (artifact_dict, metrics_dict)
    """
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, log_loss

    X = features.values.astype(np.float64)
    # Map labels to 0,1,2 for training (0=DOWN, 1=CHOP, 2=UP)
    y = (np.asarray(labels) + 1).astype(np.int64)  # -1→0, 0→1, 1→2
    n_samples = len(X)

    # --- Input contract guards ---
    if n_samples == 0:
        raise ValueError("train_v3_model received an empty feature matrix.")
    if not (len(y) == len(t1_indices) == len(uniqueness) == n_samples):
        raise ValueError(
            f"Length mismatch: X={n_samples}, labels={len(y)}, "
            f"t1={len(t1_indices)}, uniqueness={len(uniqueness)}"
        )
    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains NaN/inf — compute_features() "
                         "warm-up rows must be dropped before training.")
    if list(features.columns) != list(quant_core.FEATURE_COLUMNS):
        raise ValueError("Feature columns/order differ from quant_core.FEATURE_COLUMNS.")

    cv = PurgedWalkForwardCV(n_splits=N_CV_FOLDS, embargo_bars=EMBARGO_BARS)

    # Collect OOS predictions for profitability audit
    oos_preds = []
    oos_true = []
    oos_probs = []

    # Track best calibration coefficients
    all_calib_coefs = []
    all_calib_intercepts = []

    fold_metrics = []

    for fold_num, (train_idx, test_idx) in enumerate(cv.split(n_samples, t1_indices)):
        print(f"\n  === Fold {fold_num + 1}/{N_CV_FOLDS} ===")
        print(f"  Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")

        # --- 85/15 fit/calibration split within training fold ---
        split_point = int(len(train_idx) * 0.85)
        fit_idx = train_idx[:split_point]
        cal_idx = train_idx[split_point:]

        # Purge calibration boundary
        if len(cal_idx) > 0:
            cal_purge_mask = t1_indices[fit_idx] < cal_idx[0]
            fit_idx = fit_idx[cal_purge_mask]

        if len(fit_idx) < 100 or len(cal_idx) < 20:
            print(f"  ⚠ Skipping fold (too few samples after purge)")
            continue

        X_fit, y_fit = X[fit_idx], y[fit_idx]
        X_cal, y_cal = X[cal_idx], y[cal_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # --- Per-fold sample weights ---
        fold_w, mean_u, n_eff = quant_core.compute_fold_sample_weights(
            uniqueness[fit_idx]
        )
        print(f"  Weights: mean_u={mean_u:.4f}, N_eff/N={n_eff:.4f}")

        # --- Dynamic subsampling from uniqueness ---
        rf_max_samples = float(np.clip(mean_u, 0.12, 0.35))
        lgb_subsample = float(np.clip(mean_u * 1.5, 0.25, 0.60))

        # --- LightGBM ---
        lgb_params = LGB_PARAMS.copy()
        lgb_params['subsample'] = lgb_subsample
        lgb_params['subsample_freq'] = 1
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_fit, y_fit, sample_weight=fold_w)

        # --- RandomForest ---
        rf_params = RF_PARAMS.copy()
        rf_params['max_samples'] = rf_max_samples
        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(X_fit, y_fit, sample_weight=fold_w)

        # --- Blend predictions ---
        # Class-aware 3-column alignment (handles classes missing in a fold)
        p_lgb = _align_proba_to_classes(lgb_model, lgb_model.predict_proba(X_cal))
        p_rf = _align_proba_to_classes(rf_model, rf_model.predict_proba(X_cal))

        p_blend_cal = 0.60 * p_lgb + 0.40 * p_rf

        # --- Platt calibration on cal_idx ---
        p_clipped = np.clip(p_blend_cal, 1e-15, 1 - 1e-15)
        logits = np.log(p_clipped / (1 - p_clipped))

        # Simple per-class logistic calibration
        calib_coef = np.ones(3)
        calib_intercept = np.zeros(3)
        for c in range(3):
            y_binary = (y_cal == c).astype(np.float64)
            if y_binary.sum() < 5 or (1 - y_binary).sum() < 5:
                continue
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(max_iter=1000, solver='lbfgs')
            lr.fit(logits[:, c:c+1], y_binary)
            calib_coef[c] = float(lr.coef_[0, 0])
            calib_intercept[c] = float(lr.intercept_[0])

        all_calib_coefs.append(calib_coef)
        all_calib_intercepts.append(calib_intercept)

        # --- OOS evaluation ---
        p_lgb_test = _align_proba_to_classes(lgb_model, lgb_model.predict_proba(X_test))
        p_rf_test = _align_proba_to_classes(rf_model, rf_model.predict_proba(X_test))
        p_blend_test = 0.60 * p_lgb_test + 0.40 * p_rf_test

        # Apply calibration
        p_clipped_test = np.clip(p_blend_test, 1e-15, 1 - 1e-15)
        logits_test = np.log(p_clipped_test / (1 - p_clipped_test))
        cal_logits_test = logits_test * calib_coef + calib_intercept
        exp_l = np.exp(cal_logits_test - cal_logits_test.max(axis=1, keepdims=True))
        p_cal_test = exp_l / exp_l.sum(axis=1, keepdims=True)

        pred_classes = p_cal_test.argmax(axis=1)
        acc = accuracy_score(y_test, pred_classes)
        y_test_orig = y_test - 1  # Back to {-1, 0, 1}
        pred_orig = pred_classes - 1

        dir_mask = (pred_orig != 0) & (y_test_orig != 0)
        dir_acc = float((pred_orig[dir_mask] == y_test_orig[dir_mask]).mean()) if dir_mask.sum() > 0 else 0.0

        print(f"  OOS Accuracy: {acc:.2%}, Directional: {dir_acc:.2%}")

        fold_metrics.append({
            'fold': fold_num + 1,
            'train_size': len(fit_idx),
            'test_size': len(test_idx),
            'accuracy': acc,
            'directional_accuracy': dir_acc,
            'mean_uniqueness': mean_u,
            'n_eff_ratio': n_eff,
        })

        oos_true.extend(y_test_orig.tolist())
        oos_preds.extend(pred_orig.tolist())
        oos_probs.extend(p_cal_test.tolist())

    # ═══════════════════════════════════════════════════════════════════
    # FINAL TRAINING on all data
    # ═══════════════════════════════════════════════════════════════════
    print("\n  === Final Training on Full Dataset ===")

    if not fold_metrics:
        warnings.warn("No CV fold produced OOS metrics (all folds skipped) — "
                      "metrics will be zero; inspect dataset size / label balance.")

    # Split 85/15 for final calibration
    final_split = int(n_samples * 0.85)
    fit_idx_final = np.arange(final_split)
    cal_idx_final = np.arange(final_split, n_samples)
    if len(cal_idx_final) == 0 or final_split == 0:
        raise ValueError(f"Too few samples ({n_samples}) for the final 85/15 fit/calibration split.")

    # Purge boundary
    purge_mask = t1_indices[fit_idx_final] < cal_idx_final[0]
    fit_idx_final = fit_idx_final[purge_mask]

    # Export contract: quant_core.predict_calibrated_probabilities() assumes
    # booster/RF columns are exactly [DOWN, CHOP, UP].  LightGBM label-encodes
    # classes_, so a final model missing a class would silently shift columns.
    present = set(np.unique(y[fit_idx_final]).tolist())
    if present != {0, 1, 2}:
        missing = sorted({0, 1, 2} - present)
        raise ValueError(
            f"Final training set is missing label class(es) {[m - 1 for m in missing]} "
            f"(of -1/0/+1) — refusing to export a misaligned artifact."
        )

    final_w, final_mean_u, final_n_eff = quant_core.compute_fold_sample_weights(
        uniqueness[fit_idx_final]
    )

    rf_max_samples_final = float(np.clip(final_mean_u, 0.12, 0.35))
    lgb_subsample_final = float(np.clip(final_mean_u * 1.5, 0.25, 0.60))

    # Final LightGBM
    lgb_params_final = LGB_PARAMS.copy()
    lgb_params_final['subsample'] = lgb_subsample_final
    lgb_params_final['subsample_freq'] = 1
    final_lgb = lgb.LGBMClassifier(**lgb_params_final)
    final_lgb.fit(X[fit_idx_final], y[fit_idx_final], sample_weight=final_w)

    # Final RandomForest
    rf_params_final = RF_PARAMS.copy()
    rf_params_final['max_samples'] = rf_max_samples_final
    final_rf = RandomForestClassifier(**rf_params_final)
    final_rf.fit(X[fit_idx_final], y[fit_idx_final], sample_weight=final_w)

    # Final calibration
    p_lgb_cal = _align_proba_to_classes(final_lgb, final_lgb.predict_proba(X[cal_idx_final]))
    p_rf_cal = _align_proba_to_classes(final_rf, final_rf.predict_proba(X[cal_idx_final]))
    p_blend_final = 0.60 * p_lgb_cal + 0.40 * p_rf_cal
    p_clipped_final = np.clip(p_blend_final, 1e-15, 1 - 1e-15)
    logits_final = np.log(p_clipped_final / (1 - p_clipped_final))

    final_calib_coef = np.ones(3)
    final_calib_intercept = np.zeros(3)
    y_cal_final = y[cal_idx_final]
    for c in range(3):
        y_bin = (y_cal_final == c).astype(np.float64)
        if y_bin.sum() < 5 or (1 - y_bin).sum() < 5:
            continue
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=1000, solver='lbfgs')
        lr.fit(logits_final[:, c:c+1], y_bin)
        final_calib_coef[c] = float(lr.coef_[0, 0])
        final_calib_intercept[c] = float(lr.intercept_[0])

    # --- OOS Profitability Audit ---
    oos_true_arr = np.array(oos_true)
    oos_preds_arr = np.array(oos_preds)
    oos_probs_arr = np.array(oos_probs)

    # Simulate trades passing confidence + EV gate
    trade_mask = np.zeros(len(oos_preds_arr), dtype=bool)
    for i in range(len(oos_preds_arr)):
        if oos_preds_arr[i] == 0:
            continue
        p_win = oos_probs_arr[i][2] if oos_preds_arr[i] == 1 else oos_probs_arr[i][0]
        if p_win < MIN_CONFIDENCE:
            continue
        ev = p_win * TP_ATR_MULT - (1 - p_win) * SL_ATR_MULT - 2 * SLIPPAGE_BPS / 10000
        if ev <= 0:
            continue
        trade_mask[i] = True

    n_trades = trade_mask.sum()
    if n_trades > 0:
        traded_preds = oos_preds_arr[trade_mask]
        traded_true = oos_true_arr[trade_mask]
        wins = (traded_preds == traded_true) & (traded_preds != 0)
        losses = (traded_preds != traded_true) & (traded_preds != 0)
        win_rate = wins.sum() / max(1, (wins.sum() + losses.sum()))
        gross_win = wins.sum() * TP_ATR_MULT
        gross_loss = losses.sum() * SL_ATR_MULT
        profit_factor = gross_win / max(gross_loss, 1e-10)

        # Simplified Sharpe (per-trade)
        trade_returns = np.where(wins, TP_ATR_MULT, np.where(losses, -SL_ATR_MULT, 0))
        sharpe = float(np.mean(trade_returns) / max(np.std(trade_returns), 1e-10) * np.sqrt(252))

        # Max drawdown
        cum = np.cumsum(trade_returns)
        running_max = np.maximum.accumulate(cum)
        drawdown = running_max - cum
        max_dd = float(drawdown.max()) if len(drawdown) > 0 else 0.0
    else:
        win_rate = 0.0
        profit_factor = 0.0
        sharpe = 0.0
        max_dd = 0.0

    oos_acc = float(accuracy_score(oos_true_arr, oos_preds_arr)) if len(oos_true_arr) > 0 else 0.0
    dir_mask_oos = (oos_preds_arr != 0) & (oos_true_arr != 0)
    oos_dir_acc = float((oos_preds_arr[dir_mask_oos] == oos_true_arr[dir_mask_oos]).mean()) if dir_mask_oos.sum() > 0 else 0.0

    print(f"\n📊 OOS Profitability Audit:")
    print(f"  Total OOS samples: {len(oos_true_arr)}")
    print(f"  Trades taken (conf+EV gate): {n_trades}")
    print(f"  Win Rate:        {win_rate:.2%}")
    print(f"  Profit Factor:   {profit_factor:.2f}")
    print(f"  Sharpe Ratio:    {sharpe:.2f}")
    print(f"  Max Drawdown:    {max_dd:.2f} ATR")
    print(f"  OOS Accuracy:    {oos_acc:.2%}")
    print(f"  OOS Directional: {oos_dir_acc:.2%}")

    # ═══════════════════════════════════════════════════════════════════
    # EXPORT ARTIFACT
    # ═══════════════════════════════════════════════════════════════════

    # Get LightGBM native model string
    lgb_model_str = final_lgb.booster_.model_to_string()

    # Version info for cross-environment safety (Tweak 5)
    import sklearn
    env_versions = {
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'scikit-learn': sklearn.__version__,
        'lightgbm': lgb.__version__,
    }

    artifact = {
        'model_version': 'V3+',
        'lgb_model_str': lgb_model_str,
        'rf_model': final_rf,
        'calib_coef': final_calib_coef.tolist(),
        'calib_intercept': final_calib_intercept.tolist(),
        'feature_columns': list(quant_core.FEATURE_COLUMNS),
        'config': {
            'ticker': TICKER,
            'benchmark': BENCHMARK,
            'bar_interval_minutes': 20,
            'tp_atr_mult': TP_ATR_MULT,
            'sl_atr_mult': SL_ATR_MULT,
            'max_horizon_bars': MAX_HORIZON_BARS,
            'min_confidence': MIN_CONFIDENCE,
            'slippage_bps': SLIPPAGE_BPS,
            'n_cv_folds': N_CV_FOLDS,
            'embargo_bars': EMBARGO_BARS,
        },
        'env_versions': env_versions,
    }

    metrics = {
        'accuracy': oos_acc,
        'directional_accuracy': oos_dir_acc,
        'f1_weighted': float(f1_score(oos_true_arr, oos_preds_arr, average='weighted', zero_division=0)) if len(oos_true_arr) > 0 else 0.0,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe,
        'max_drawdown_atr': max_dd,
        'n_trades_oos': int(n_trades),
        'slippage_bps': SLIPPAGE_BPS,  # read by predict.py EV calculation
        'fold_metrics': fold_metrics,
    }

    return artifact, metrics


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Complete V3+ training pipeline: stream → label → train → export."""
    print("=" * 60)
    print("  V3+ Cloud Training Pipeline")
    print("=" * 60)

    # --- Step 1: Stream data ---
    # ONE client, ONE credential probe, ONE feed for both symbols, so AAPL and
    # SPY volumes/VWAPs come from the same tape (no SIP/IEX mixing).
    from alpaca.data.historical import StockHistoricalDataClient
    print("\n📡 Step 1: Streaming AAPL + SPY bars from Alpaca...")
    key, secret = _resolve_alpaca_credentials()
    client = StockHistoricalDataClient(key, secret)
    print("  🔍 Probing Alpaca credentials and feed access...")
    feed = _probe_alpaca_credentials(client, key, secret)

    aapl_5m = stream_alpaca_bars(TICKER, START_YEAR, client=client, feed=feed)
    feed = aapl_5m.attrs.get('feed', feed)          # AAPL may have fallen back to IEX
    spy_5m = stream_alpaca_bars(BENCHMARK, START_YEAR, client=client, feed=feed,
                                start_dt=aapl_5m.index[0].to_pydatetime())
    spy_feed = spy_5m.attrs.get('feed', feed)
    if spy_feed != feed:
        # SPY was forced onto IEX mid-stream → re-stream AAPL on the same feed.
        print(f"  ↺ SPY ended on {spy_feed.upper()} — re-streaming {TICKER} on "
              f"{spy_feed.upper()} for feed consistency.")
        feed = spy_feed
        aapl_5m = stream_alpaca_bars(TICKER, START_YEAR, client=client, feed=feed,
                                     start_dt=spy_5m.index[0].to_pydatetime())

    # Common date window (both tz-aware UTC) — no orphan SPY/AAPL head/tail.
    common_start = max(aapl_5m.index[0], spy_5m.index[0])
    common_end = min(aapl_5m.index[-1], spy_5m.index[-1])
    aapl_5m = aapl_5m.loc[common_start:common_end]
    spy_5m = spy_5m.loc[common_start:common_end]
    print(f"  Feed: {feed.upper()} | common window {common_start.date()} → {common_end.date()}")

    # --- Step 2: Resample to 20-minute bars ---
    print("\n📐 Step 2: Resampling to 20-minute bars (09:30-anchored)...")
    aapl_20m = quant_core.resample_bars_20min(aapl_5m)
    spy_20m = quant_core.resample_bars_20min(spy_5m)
    if aapl_20m is None or aapl_20m.empty:
        raise RuntimeError(f"{TICKER}: no regular-trading-hours bars after resampling.")
    # Pre-align SPY onto AAPL's exact bar index, forward-filling ONLY within
    # the same session.  compute_features() does reindex(method='ffill'),
    # which would otherwise carry the PRIOR day's last SPY bar into a day
    # where SPY bars are missing (stale cross-day spy_ret / beta).  With an
    # exact-index frame its ffill is a no-op; residual NaNs (no same-day SPY
    # bar yet) are removed by compute_features()' own dropna().
    if spy_20m is not None and not spy_20m.empty:
        n_spy_raw = len(spy_20m)
        missing = aapl_20m.index.difference(spy_20m.index)
        spy_20m = spy_20m.reindex(aapl_20m.index)
        spy_20m = spy_20m.groupby(spy_20m.index.normalize()).ffill()
        n_unfilled = int(spy_20m['Close'].isna().sum())
        print(f"  AAPL: {len(aapl_20m)} bars, SPY: {n_spy_raw} bars | "
              f"{len(missing)} AAPL bars lack an exact SPY bar "
              f"({len(missing) - n_unfilled} same-session ffilled, {n_unfilled} dropped)")
    else:
        print(f"  AAPL: {len(aapl_20m)} bars, SPY: 0 bars (SPY features zero-filled)")

    # --- Step 3: Compute features ---
    print("\n🔧 Step 3: Computing 31 V3+ features...")
    features = quant_core.compute_features(aapl_20m, spy_20m)
    print(f"  Feature matrix: {features.shape}")

    # --- Step 4: Triple-barrier labeling ---
    print("\n🏷️  Step 4: ATR triple-barrier labeling...")
    import pytz
    et = pytz.timezone('America/New_York')
    if aapl_20m.index.tzinfo is None:
        session_date = pd.Series(aapl_20m.index.normalize(), index=aapl_20m.index)
    else:
        session_date = pd.Series(
            aapl_20m.index.tz_convert(et).normalize(), index=aapl_20m.index
        )

    atr = quant_core._day_boundary_atr(
        aapl_20m['High'], aapl_20m['Low'], aapl_20m['Close'],
        session_date, period=14
    )

    labels, t1_indices = triple_barrier_label(
        aapl_20m, atr, session_date,
        tp_mult=TP_ATR_MULT, sl_mult=SL_ATR_MULT, max_horizon=MAX_HORIZON_BARS,
    )

    # Align features to labels
    valid_mask = ~np.isnan(atr.values) & (atr.values > 0)
    common_idx = features.index.intersection(aapl_20m.index[valid_mask])
    features = features.loc[common_idx]

    # Map back to positional indices for labels
    pos_map = {ts: i for i, ts in enumerate(aapl_20m.index)}
    pos_indices = np.array([pos_map[ts] for ts in common_idx])
    labels = labels[pos_indices]
    t1_indices = t1_indices[pos_indices]

    if len(features) == 0:
        raise RuntimeError("No samples left after aligning features with valid ATR bars.")

    # Reindex t1_indices to be relative to the filtered dataset.
    # pos_indices is sorted, so searchsorted(side='right') - 1 maps each raw
    # t1 to the last KEPT bar <= t1 (conservative: never shortens beyond the
    # true horizon).  The old `.get(t1, N-1)` fallback sent any dropped t1 to
    # the LAST sample, inflating concurrency and purging it from every fold.
    t1_reindexed = np.searchsorted(pos_indices, t1_indices, side='right') - 1
    # Ensure t1 >= sample index
    sample_indices = np.arange(len(features))
    t1_reindexed = np.maximum(t1_reindexed, sample_indices).astype(np.int64)

    from collections import Counter
    dist = Counter(labels.tolist())
    total = len(labels)
    print(f"  Labels: UP={dist.get(1,0)} ({dist.get(1,0)/total:.1%}), "
          f"CHOP={dist.get(0,0)} ({dist.get(0,0)/total:.1%}), "
          f"DOWN={dist.get(-1,0)} ({dist.get(-1,0)/total:.1%})")

    # --- Step 5: Compute uniqueness ---
    print("\n📊 Step 5: Computing sample uniqueness (O(N))...")
    uniqueness = quant_core.compute_vectorized_uniqueness(t1_reindexed)
    print(f"  Mean uniqueness: {uniqueness.mean():.4f}")

    # --- Step 6: Train ---
    print("\n🏋️  Step 6: Purged Walk-Forward CV + Training...")
    artifact, metrics = train_v3_model(features, labels, t1_reindexed, uniqueness)

    # --- Step 7: Export ---
    print("\n💾 Step 7: Exporting artifact...")
    joblib.dump(artifact, 'stock_price_model.pkl', compress=3)
    pkl_size = os.path.getsize('stock_price_model.pkl') / 1024
    print(f"  stock_price_model.pkl: {pkl_size:.0f} KB")

    metadata = {
        'ticker': TICKER,
        'model_type': 'LightGBM_RandomForest_Ensemble_V3+',
        'model_version': 'V3+',
        'interval': '20m',
        'horizon_bars': MAX_HORIZON_BARS,
        'labeling': 'ATR_triple_barrier_symmetric',
        'tp_atr_mult': TP_ATR_MULT,
        'sl_atr_mult': SL_ATR_MULT,
        'min_confidence': MIN_CONFIDENCE,
        'slippage_bps': SLIPPAGE_BPS,
        'data_feed': feed,
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'training_start': str(features.index[0].date()) if len(features) > 0 else None,
        'training_end': str(features.index[-1].date()) if len(features) > 0 else None,
        'training_bars': len(features),
        'features': list(quant_core.FEATURE_COLUMNS),
        'news_training_source': None,
        'metrics': metrics,
    }

    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print("  model_metadata.json: saved")

    # --- Auto-download in Colab ---
    try:
        from google.colab import files
        files.download('stock_price_model.pkl')
        files.download('model_metadata.json')
        print("\n📥 Files downloaded to your browser!")
    except ImportError:
        print("\n📁 Files saved to current directory. Copy to your local project.")

    print("\n" + "=" * 60)
    print("  ✅ V3+ Training Complete!")
    print("=" * 60)

    return artifact, metadata


if __name__ == '__main__':
    main()

