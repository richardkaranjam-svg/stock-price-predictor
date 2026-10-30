import os
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

import pandas as pd
import requests

MARKETAUX_URL = 'https://api.marketaux.com/v1/news/all'
PREFERRED_SOURCES = {'reuters.com', 'bloomberg.com', 'wsj.com', 'wallstreetjournal.com'}
CACHE_PATH = Path(__file__).resolve().parent / 'marketaux_cache.json'
CACHE_TTL_SECONDS = 1800
DAILY_LIMIT = 100
HISTORICAL_RESERVE = 20
USAGE_PATH = Path(__file__).resolve().parent / 'marketaux_usage.json'
USAGE_LOCK = Lock()


def _token():
    token = os.environ.get('MARKETAUX_API_TOKEN')
    if not token:
        raise RuntimeError('Set MARKETAUX_API_TOKEN to use MarketAux news.')
    return token


def usage_status():
    now = datetime.now(timezone.utc)
    try:
        usage = json.loads(USAGE_PATH.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        usage = {}
    window_started = usage.get('window_started')
    try:
        started_at = datetime.fromisoformat(window_started) if window_started else None
    except ValueError:
        started_at = None
    if started_at is None or now - started_at >= timedelta(hours=24):
        usage = {'window_started': now.isoformat(), 'current': 0, 'historical': 0}
    total = usage.get('current', 0) + usage.get('historical', 0)
    reset_at = datetime.fromisoformat(usage['window_started']) + timedelta(hours=24)
    return {**usage, 'total': total, 'remaining': max(0, DAILY_LIMIT - total),
            'reset_at': reset_at.isoformat()}


def _consume_request(kind):
    if kind not in {'current', 'historical'}:
        raise ValueError('Invalid MarketAux request type.')
    with USAGE_LOCK:
        status = usage_status()
        if status['remaining'] <= 0:
            raise RuntimeError('MarketAux daily limit reached; usage resets at midnight UTC.')
        if kind == 'historical' and status['historical'] >= DAILY_LIMIT - HISTORICAL_RESERVE:
            raise RuntimeError('Historical MarketAux budget is reserved for current-news requests.')
        usage = {
            'window_started': status['window_started'],
            'current': status['current'] + (kind == 'current'),
            'historical': status['historical'] + (kind == 'historical'),
        }
        USAGE_PATH.write_text(json.dumps(usage, indent=2), encoding='utf-8')


def _source_domain(article):
    source = article.get('source') or article.get('source_name') or ''
    domain = article.get('source_domain') or article.get('domain') or source
    return str(domain).lower().removeprefix('www.')


def fetch_marketaux_news(ticker, published_after=None, published_before=None,
                         max_pages=1, preferred_only=False, request_kind='current'):
    """Fetch MarketAux articles and return a normalized timestamped table."""
    rows = []
    for page in range(1, max_pages + 1):
        _consume_request(request_kind)
        params = {
            'api_token': _token(),
            'symbols': ticker.upper(),
            'language': 'en',
            'limit': 100,
            'page': page,
        }
        if published_after:
            params['published_after'] = str(published_after)
        if published_before:
            params['published_before'] = str(published_before)
        response = requests.get(MARKETAUX_URL, params=params, timeout=20)
        response.raise_for_status()
        articles = response.json().get('data', [])
        if not articles:
            break
        for article in articles:
            domain = _source_domain(article)
            if preferred_only and domain not in PREFERRED_SOURCES:
                continue
            rows.append({
                'date': article.get('published_at') or article.get('published_on'),
                'title': article.get('title', ''),
                'ticker': ticker.upper(),
                'source': domain,
                'url': article.get('url', ''),
            })
        if len(articles) < 100:
            break
    result = pd.DataFrame(rows, columns=['date', 'title', 'ticker', 'source', 'url'])
    if result.empty:
        return result
    result['date'] = pd.to_datetime(result['date'], errors='coerce', utc=True)
    return result.dropna(subset=['date']).drop_duplicates(subset=['date', 'title', 'url'])


def fetch_historical_news(ticker, start_date, end_date=None):
    """Fetch history in 90-day windows to limit API usage."""
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date() if end_date else date.today()
    chunks = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=90), end)
        chunks.append(fetch_marketaux_news(
            ticker, cursor, chunk_end, max_pages=1, request_kind='historical'
        ))
        cursor = chunk_end
    if not chunks:
        return pd.DataFrame(columns=['date', 'title', 'ticker', 'source', 'url'])
    return pd.concat(chunks, ignore_index=True).drop_duplicates(
        subset=['date', 'title', 'url']
    ).sort_values('date')


def current_news_summary(ticker):
    cache = {}
    if CACHE_PATH.exists():
        try:
            cache = json.loads(CACHE_PATH.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            cache = {}
    cached = cache.get(ticker.upper())
    now = datetime.now(timezone.utc).timestamp()
    if cached and now - cached.get('fetched_at', 0) < CACHE_TTL_SECONDS:
        articles = pd.DataFrame(cached.get('articles', []))
        preferred = articles[articles['source'].isin(PREFERRED_SOURCES)] if not articles.empty else articles
        return articles, preferred

    articles = fetch_marketaux_news(ticker, max_pages=1)
    cache[ticker.upper()] = {
        'fetched_at': now,
        'articles': articles.to_dict('records'),
    }
    CACHE_PATH.write_text(json.dumps(cache, default=str), encoding='utf-8')
    preferred = articles[articles['source'].isin(PREFERRED_SOURCES)] if not articles.empty else articles
    return articles, preferred