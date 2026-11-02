"""
config.py - Shared configuration and market-hours utilities.

Import `is_market_open()` in predict.py and auto_trader.py to prevent
unintended orders outside market hours.
"""
import os
from datetime import datetime, time, timedelta, timezone
from pathlib import Path


def load_local_env():
    """Load simple KEY=VALUE entries from the project-local .env file."""
    env_path = Path(__file__).resolve().parent / '.env'
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()


def get_et_now() -> datetime:
    """Return the current wall-clock datetime in US Eastern."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo('America/New_York'))
    except Exception:
        try:
            import pytz
            return datetime.now(pytz.timezone('America/New_York'))
        except Exception:
            return datetime.now(timezone(timedelta(hours=-5)))


# ---------------------------------------------------------------------------
# Market-hours guard
# ---------------------------------------------------------------------------

def get_market_status() -> dict:
    """Return a detailed status dict of US equity market state."""
    if os.getenv('MARKET_HOURS_CHECK', 'on').lower() == 'off':
        return {
            'is_open': True,
            'reason': 'Market hours check disabled by config (MARKET_HOURS_CHECK=off)',
            'override': True,
        }

    now_et = get_et_now()
    weekday = now_et.weekday()
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_name = weekday_names[weekday] if weekday < 7 else 'Unknown'
    current_time = now_et.time()
    time_str = now_et.strftime('%H:%M:%S ET')

    if weekday >= 5:
        return {
            'is_open': False,
            'is_weekend': True,
            'day': day_name,
            'current_time_et': time_str,
            'reason': f'Weekend ({day_name}) - US equity markets are closed Saturday & Sunday',
        }

    market_open = time(9, 30)
    market_close = time(16, 0)
    if current_time < market_open:
        return {
            'is_open': False,
            'is_weekend': False,
            'day': day_name,
            'current_time_et': time_str,
            'reason': f'Pre-market ({time_str}) - Regular trading starts at 09:30 ET',
        }
    elif current_time > market_close:
        return {
            'is_open': False,
            'is_weekend': False,
            'day': day_name,
            'current_time_et': time_str,
            'reason': f'After-hours ({time_str}) - Regular trading closed at 16:00 ET',
        }

    return {
        'is_open': True,
        'is_weekend': False,
        'day': day_name,
        'current_time_et': time_str,
        'reason': 'US equity market is open (09:30–16:00 ET)',
    }


def is_market_open() -> bool:
    """
    Return True only when the US equity market is currently open.

    Rules:
    - Monday–Friday (weekday 0–4)
    - 09:30 – 16:00 Eastern Time

    Override: set the environment variable MARKET_HOURS_CHECK=off to
    disable the guard (useful for testing outside market hours).
    """
    return get_market_status()['is_open']


def next_market_open_seconds() -> int:
    """
    Return the number of seconds until the next market open (09:30 ET).
    Used by auto_trader.py to sleep until trading is possible.
    """
    now_et = get_et_now()

    # Advance to next weekday 09:30 ET
    target = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    if now_et >= target or now_et.weekday() >= 5:
        target += timedelta(days=1)
        # Skip to Monday if weekend
        while target.weekday() >= 5:
            target += timedelta(days=1)
        target = target.replace(hour=9, minute=30, second=0, microsecond=0)

    delta = (target - now_et).total_seconds()
    return max(int(delta), 60)  # at least 60 seconds


def is_high_signal_window() -> bool:
    """
    Return True only during high-momentum trading windows.
    Skips the 11:00–14:30 ET midday chop zone (~60% of false signals).
    Window 1: Opening momentum 09:30–11:00 ET (0–90 min)
    Window 2: Closing flow     14:30–15:30 ET (300–360 min)
    V3+: Ends at 15:30 ET (not 16:00) to honor the 15:30 ET entry cutoff.
    """
    if os.getenv('MARKET_HOURS_CHECK', 'on').lower() == 'off':
        return True

    now_et = get_et_now()
    if now_et.weekday() >= 5:
        return False

    mins = (now_et.hour - 9) * 60 + (now_et.minute - 30)
    return (0 <= mins <= 90) or (300 <= mins <= 360)
