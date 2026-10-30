"""
config.py — Shared configuration and market-hours utilities.

Import `is_market_open()` in predict.py and auto_trader.py to prevent
forecasts and trades when the US equity market is closed.
"""
import os
from datetime import datetime, time

# ---------------------------------------------------------------------------
# Market-hours guard
# ---------------------------------------------------------------------------

def is_market_open() -> bool:
    """
    Return True only when the US equity market is currently open.

    Rules:
    - Monday–Friday (weekday 0–4)
    - 09:30 – 16:00 Eastern Time

    Override: set the environment variable MARKET_HOURS_CHECK=off to
    disable the guard (useful for testing outside market hours).
    """
    if os.getenv('MARKET_HOURS_CHECK', 'on').lower() == 'off':
        return True

    try:
        import pytz
        et = pytz.timezone('America/New_York')
        now_et = datetime.now(et)
    except ImportError:
        # pytz not available — fall back to UTC offset -5 (ET standard)
        from datetime import timezone, timedelta
        now_et = datetime.now(timezone(timedelta(hours=-5)))

    # Weekend check
    if now_et.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False

    market_open  = time(9, 30)
    market_close = time(16, 0)
    current_time = now_et.time()

    return market_open <= current_time <= market_close


def next_market_open_seconds() -> int:
    """
    Return the number of seconds until the next market open (09:30 ET).
    Used by auto_trader.py to sleep until trading is possible.
    """
    try:
        import pytz
        from datetime import timedelta
        et = pytz.timezone('America/New_York')
        now_et = datetime.now(et)
    except ImportError:
        from datetime import timezone, timedelta
        now_et = datetime.now(timezone(timedelta(hours=-5)))

    # Advance to next weekday 09:30 ET
    target = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    if now_et >= target or now_et.weekday() >= 5:
        target += __import__('datetime').timedelta(days=1)
        # Skip to Monday if weekend
        while target.weekday() >= 5:
            target += __import__('datetime').timedelta(days=1)
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

    try:
        import pytz
        et = pytz.timezone('America/New_York')
        now_et = datetime.now(et)
    except ImportError:
        from datetime import timezone, timedelta
        now_et = datetime.now(timezone(timedelta(hours=-5)))

    if now_et.weekday() >= 5:
        return False

    mins = (now_et.hour - 9) * 60 + (now_et.minute - 30)
    return (0 <= mins <= 90) or (300 <= mins <= 360)


