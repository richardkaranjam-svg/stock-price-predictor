#!/usr/bin/env python
"""
data_loader.py — V3+ Data Loader (100% Alpaca SIP, Zero yfinance).

All market data flows exclusively through the Alpaca Market Data API.
Implements hybrid SIP + real-time fetch (Tweak 2) and strict closed-bar
slicing (Issue #18).

Resolves Issues: #1, #7, #13, #18
"""
import os
import time
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

import quant_core

REQUIRED_COLUMNS = {'Close', 'Volume', 'High', 'Low'}


def _get_alpaca_data_client():
    """Create an Alpaca StockHistoricalDataClient using broker credentials."""
    from broker import load_local_credentials
    load_local_credentials()

    from alpaca.data.historical import StockHistoricalDataClient
    key = os.environ.get('APCA_API_KEY_ID', '')
    secret = os.environ.get('APCA_API_SECRET_KEY', '')
    if not key or not secret:
        raise RuntimeError('Set APCA_API_KEY_ID and APCA_API_SECRET_KEY first.')
    return StockHistoricalDataClient(key, secret)


def normalize_downloaded_data(data):
    """Return a usable OHLCV DataFrame — works with any data source."""
    if data is None or data.empty:
        raise ValueError("Data source returned no data. Check the ticker and date range.")

    normalized = data.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)

    # Standardize column names (Alpaca uses lowercase)
    rename_map = {
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume',
    }
    normalized = normalized.rename(columns=rename_map)

    missing_columns = REQUIRED_COLUMNS - set(normalized.columns)
    if missing_columns:
        missing = ', '.join(sorted(missing_columns))
        raise ValueError(f"Downloaded data is missing required columns: {missing}")

    normalized = normalized.sort_index()
    return normalized


def download_intraday_data(
    ticker: str,
    interval: str = '20m',
    period: str = '60d',
    warmup_days: int = 25,
) -> pd.DataFrame:
    """Download intraday bars from Alpaca SIP and resample to 20-minute bars.

    Hybrid SIP + real-time fetch (Tweak 2):
      - Warm-up: now - warmup_days to now - 16 min → DataFeed.SIP (free, 100% volume)
      - Recent tail: now - 16 min to now → SIP first, IEX fallback on HTTP 403

    Parameters
    ----------
    ticker : str
        Stock symbol.
    interval : str
        Target bar interval ('20m' supported).
    period : str
        Ignored (kept for backward compatibility). Uses warmup_days instead.
    warmup_days : int
        Number of calendar days of warm-up history.

    Returns
    -------
    pd.DataFrame
        20-minute OHLCV bars with VWAP column.
    """
    import joblib
    from datetime import datetime, timezone, timedelta
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import Adjustment, DataFeed

    # --- Cache check (20-minute TTL) ---
    cache_file = Path(__file__).resolve().parent / f'.cache_{ticker}_v3_20m.pkl'
    if cache_file.exists():
        cache_time = cache_file.stat().st_mtime
        if time.time() - cache_time < 1200:
            data = joblib.load(cache_file)
            print(f"Loaded {len(data)} V3+ intraday bars from disk cache")
            return data

    client = _get_alpaca_data_client()
    now = datetime.now(timezone.utc)
    tf_5min = TimeFrame(5, TimeFrameUnit.Minute)

    # --- Part 1: Historical warm-up via SIP (with IEX fallback for free tier) ---
    hist_start = now - timedelta(days=warmup_days)
    hist_end = now - timedelta(minutes=16)

    print(f"[INFO] Fetching {ticker} 5-min bars from Alpaca ({warmup_days}d warm-up)...")
    all_bars = []

    for feed in [DataFeed.SIP, DataFeed.IEX]:
        try:
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=tf_5min,
                start=hist_start,
                end=hist_end,
                adjustment=Adjustment.ALL,
                feed=feed,
            )
            bars = client.get_stock_bars(request)
            if bars and bars.data and ticker in bars.data:
                for b in bars.data[ticker]:
                    all_bars.append({
                        'timestamp': b.timestamp,
                        'Open': float(b.open),
                        'High': float(b.high),
                        'Low': float(b.low),
                        'Close': float(b.close),
                        'Volume': float(b.volume),
                    })
            if all_bars:
                break
        except Exception as e:
            if feed == DataFeed.SIP:
                print(f"  [WARN] SIP warm-up failed ({e}), trying IEX...")
            else:
                print(f"  [WARN] IEX warm-up also failed: {e}")

    # --- Part 2: Recent tail via SIP (with IEX fallback) ---
    recent_start = hist_end
    recent_end = now

    for feed in [DataFeed.SIP, DataFeed.IEX]:
        try:
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=tf_5min,
                start=recent_start,
                end=recent_end,
                adjustment=Adjustment.ALL,
                feed=feed,
            )
            bars = client.get_stock_bars(request)
            if bars and bars.data and ticker in bars.data:
                for b in bars.data[ticker]:
                    all_bars.append({
                        'timestamp': b.timestamp,
                        'Open': float(b.open),
                        'High': float(b.high),
                        'Low': float(b.low),
                        'Close': float(b.close),
                        'Volume': float(b.volume),
                    })
            break
        except Exception as e:
            if feed == DataFeed.SIP:
                print(f"  [WARN] SIP recent tail failed ({e}), falling back to IEX...")
            else:
                print(f"  [WARN] IEX fallback also failed: {e}")

    if not all_bars:
        raise RuntimeError(f"No data received from Alpaca for {ticker}.")

    # --- Build DataFrame ---
    df = pd.DataFrame(all_bars)
    df.index = pd.DatetimeIndex(df['timestamp'])
    df = df.drop(columns=['timestamp']).sort_index()
    df = df[~df.index.duplicated(keep='last')]

    print(f"  Received {len(df)} raw 5-min bars")

    # --- Resample to 20-min using quant_core (Issue #7 fix) ---
    bars_20m = quant_core.resample_bars_20min(df)
    print(f"  Resampled to {len(bars_20m)} 20-min bars (09:30-anchored)")

    # --- Cache ---
    joblib.dump(bars_20m, cache_file)

    return bars_20m


def download_spy_intraday(warmup_days: int = 25) -> pd.DataFrame:
    """Download SPY 20-minute bars for cross-sectional features.

    Uses the same hybrid SIP + IEX approach as download_intraday_data.
    Returns empty DataFrame on failure (graceful degradation).
    """
    try:
        return download_intraday_data('SPY', warmup_days=warmup_days)
    except Exception as e:
        print(f"  [WARN] SPY data unavailable ({e}), SPY features will be zeroed")
        return pd.DataFrame()


def create_features(data: pd.DataFrame, spy_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Create the V3+ 31-feature matrix via quant_core.

    Parameters
    ----------
    data : pd.DataFrame
        20-minute OHLCV bars (from download_intraday_data).
    spy_data : pd.DataFrame or None
        Matching SPY bars. If None, attempts to download SPY automatically.

    Returns
    -------
    pd.DataFrame
        Feature matrix with 31 columns matching quant_core.FEATURE_COLUMNS.
    """
    if data is None or data.empty:
        raise ValueError('At least 65 trading bars are required to create features.')
    if len(data) < 65:
        raise ValueError('At least 65 trading bars are required to create features.')

    # Auto-download SPY if not provided
    if spy_data is None:
        spy_data = download_spy_intraday()

    return quant_core.compute_features(data, spy_data)


# --- Kaggle/news utilities (preserved from V2) ---

import joblib


def download_kaggle_dataset(dataset_handle, output_dir='kaggle_data'):
    """Download an optional Kaggle dataset using its public handle.

    Kaggle credentials are required by kagglehub. This function is kept
    separate from market data loading so training still works
    without Kaggle access.
    """
    try:
        import kagglehub
    except ImportError as error:
        raise ImportError(
            'Install Kaggle support with: pip install -r requirements-kaggle.txt'
        ) from error

    download_path = kagglehub.dataset_download(dataset_handle)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    return Path(download_path)


def load_kaggle_news_csv(csv_path):
    """Load common Kaggle news CSV layouts into date/title/ticker columns."""
    news = pd.read_csv(csv_path)
    normalized = {str(column).strip().lower(): column for column in news.columns}
    date_column = next(
        (normalized[name] for name in ('date', 'publish date', 'published_at', 'timestamp') if name in normalized),
        None,
    )
    title_column = next(
        (normalized[name] for name in ('title', 'headline', 'headline_text') if name in normalized),
        None,
    )
    if date_column is None or title_column is None:
        raise ValueError('Kaggle news CSV needs a date and title/headline column.')

    result = pd.DataFrame({
        'date': pd.to_datetime(news[date_column], errors='coerce', utc=True),
        'title': news[title_column].fillna('').astype(str),
    })
    ticker_column = next(
        (normalized[name] for name in ('ticker', 'symbol') if name in normalized),
        None,
    )
    result['ticker'] = news[ticker_column].astype(str).str.upper() if ticker_column else ''
    return result.dropna(subset=['date']).sort_values('date').reset_index(drop=True)


def historical_news_sentiment(csv_path, feature_index, ticker):
    """Align timestamped headlines to market dates without using future news."""
    news = load_kaggle_news_csv(csv_path)
    ticker = ticker.upper()
    if news['ticker'].ne('').any():
        news = news[(news['ticker'] == '') | (news['ticker'] == ticker)]
    if news.empty:
        return pd.Series(0.0, index=feature_index)

    def score_title(title):
        words = set(title.lower().replace('-', ' ').split())
        positive = len(words & POSITIVE_NEWS_WORDS)
        negative = len(words & NEGATIVE_NEWS_WORDS)
        return (positive - negative) / (positive + negative) if positive or negative else 0.0

    daily = news.assign(
        market_date=pd.to_datetime(news['date'].dt.tz_convert(None).dt.normalize()).astype('datetime64[ns]'),
        score=news['title'].map(score_title),
    ).groupby('market_date')['score'].mean().reset_index()
    target = pd.DataFrame({
        'market_date': pd.to_datetime(pd.DatetimeIndex(feature_index).tz_localize(None).normalize()).astype('datetime64[ns]')
    })
    aligned = pd.merge_asof(
        target.sort_values('market_date'), daily.sort_values('market_date'),
        on='market_date', direction='backward',
    )['score'].fillna(0.0).to_numpy()
    return pd.Series(aligned, index=feature_index)


POSITIVE_NEWS_WORDS = {
    'beat', 'beats', 'growth', 'gain', 'gains', 'improve', 'improved',
    'profit', 'strong', 'surge', 'up', 'upgrade', 'bullish', 'record',
}
NEGATIVE_NEWS_WORDS = {
    'bearish', 'below', 'decline', 'drop', 'falls', 'loss', 'miss',
    'misses', 'risk', 'sell', 'weak', 'downgrade', 'lawsuit', 'cut',
}


def get_news_sentiment(ticker):
    """Estimate recent headline sentiment from MarketAux / news_client.

    Falls back to neutral (0.0) if no news provider is configured.
    """
    try:
        from news_client import current_news_summary
        articles, preferred = current_news_summary(ticker)
        all_articles = preferred if not preferred.empty else articles
        scores = []
        for title in all_articles['title'].head(20):
            words = set(str(title).lower().replace('-', ' ').split())
            positive = len(words & POSITIVE_NEWS_WORDS)
            negative = len(words & NEGATIVE_NEWS_WORDS)
            if positive or negative:
                scores.append((positive - negative) / (positive + negative))
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0


def get_news_summary(ticker):
    """Return current MarketAux headline sentiment and source counts."""
    try:
        from news_client import current_news_summary
        from_news, preferred = current_news_summary(ticker)
        articles = preferred if not preferred.empty else from_news
        scores = []
        for title in articles['title'].head(20):
            words = set(str(title).lower().replace('-', ' ').split())
            positive = len(words & POSITIVE_NEWS_WORDS)
            negative = len(words & NEGATIVE_NEWS_WORDS)
            if positive or negative:
                scores.append((positive - negative) / (positive + negative))
        return {
            'sentiment': float(np.mean(scores)) if scores else 0.0,
            'headline_count': len(articles),
            'source_count': int(articles['source'].nunique()) if not articles.empty else 0,
            'provider': 'MarketAux',
        }
    except (RuntimeError, requests.RequestException, ValueError, KeyError):
        pass

    return {'sentiment': 0.0, 'headline_count': 0, 'provider': 'none'}
