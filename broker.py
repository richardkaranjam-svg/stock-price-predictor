"""
broker.py — Alpaca Paper Trading Client with V3+ Short-Selling & Margin Guards.

Resolves Gap 3 (Issue #4) and Tweak 4 (Issue #15):
  - check_short_eligibility(): equity >= $2000, shortable, easy_to_borrow
  - get_short_safe_buying_power(): RegT buying power with safety factor
  - cancel_open_orders_for_symbol(): cancel resting orders before flatten/reverse
"""
import os
from pathlib import Path


def load_local_credentials():
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


load_local_credentials()


def get_alpaca_client():
    try:
        from alpaca.trading.client import TradingClient
    except ImportError as error:
        raise RuntimeError('Install Alpaca support with: pip install alpaca-py') from error

    key = os.environ.get('APCA_API_KEY_ID')
    secret = os.environ.get('APCA_API_SECRET_KEY')
    if not key or not secret:
        raise RuntimeError('Set APCA_API_KEY_ID and APCA_API_SECRET_KEY first.')
    if os.environ.get('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets') != 'https://paper-api.alpaca.markets':
        raise RuntimeError('Only Alpaca paper trading is enabled by this application.')
    return TradingClient(key, secret, paper=True)


def get_paper_account():
    account = get_alpaca_client().get_account()
    return {
        'status': getattr(account.status, 'value', str(account.status)),
        'buying_power': str(account.buying_power),
        'portfolio_value': str(account.portfolio_value),
        'equity': str(account.equity),
    }


def submit_paper_market_order(ticker, side, quantity):
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    if side not in {'buy', 'sell'}:
        raise ValueError('Order side must be buy or sell.')
    try:
        quantity = float(quantity)
    except (TypeError, ValueError) as error:
        raise ValueError('Quantity must be a positive number.') from error
    if quantity <= 0:
        raise ValueError('Quantity must be a positive number.')

    order = MarketOrderRequest(
        symbol=ticker.upper(),
        qty=quantity,
        side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    submitted = get_alpaca_client().submit_order(order_data=order)
    return {
        'id': str(submitted.id),
        'ticker': submitted.symbol,
        'side': submitted.side.value,
        'quantity': str(submitted.qty),
        'status': submitted.status.value,
    }


def submit_paper_limit_order(ticker, side, quantity, limit_price):
    """Submit a limit order — fills only at limit_price or better, avoiding slippage."""
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest

    if side not in {'buy', 'sell'}:
        raise ValueError('Order side must be buy or sell.')
    quantity = float(quantity)
    limit_price = float(limit_price)
    if quantity <= 0 or limit_price <= 0:
        raise ValueError('Quantity and limit_price must be positive.')

    order = LimitOrderRequest(
        symbol=ticker.upper(),
        qty=quantity,
        side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        limit_price=round(limit_price, 2),
    )
    submitted = get_alpaca_client().submit_order(order_data=order)
    return {
        'id': str(submitted.id),
        'ticker': submitted.symbol,
        'side': submitted.side.value,
        'quantity': str(submitted.qty),
        'limit_price': str(limit_price),
        'status': submitted.status.value,
    }


def cancel_open_orders(ticker):
    """Cancel all pending/open orders for a ticker before placing a new one.

    This prevents the Alpaca 'insufficient qty available' (40310000) error that
    occurs when shares are held_for_orders by a previous unexecuted order.
    Returns the number of orders cancelled.
    """
    client = get_alpaca_client()
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        open_orders = client.get_orders(filter=GetOrdersRequest(
            symbol=ticker.upper(),
            status=QueryOrderStatus.OPEN,
        ))
        count = 0
        for order in open_orders:
            try:
                client.cancel_order_by_id(str(order.id))
                count += 1
            except Exception:
                pass
        return count
    except Exception:
        return 0


def cancel_open_orders_for_symbol(symbol: str) -> int:
    """Cancel any resting open stop/bracket orders for a symbol.

    Called before flattening or reversing positions so Alpaca never
    rejects an exit due to locked shares.

    Parameters
    ----------
    symbol : str
        Stock symbol.

    Returns
    -------
    int
        Number of orders cancelled.
    """
    return cancel_open_orders(symbol)


def get_available_qty(ticker):
    """Return the quantity of shares available to sell (not held for orders)."""
    client = get_alpaca_client()
    try:
        position = client.get_open_position(ticker.upper())
        return float(position.qty_available)
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# V3+ SHORT-SELLING & MARGIN GUARD HELPERS (Gap 3, Tweak 4)
# ═══════════════════════════════════════════════════════════════════════════

def check_short_eligibility(symbol: str) -> dict:
    """Verify whether the account and asset allow short selling.

    Checks:
      (a) Account equity >= $2,000 (Reg T minimum for margin)
      (b) Asset is shortable
      (c) Asset is easy_to_borrow (no locate fees / forced buy-ins)

    Parameters
    ----------
    symbol : str
        Stock symbol to check.

    Returns
    -------
    dict
        {
            'allowed': bool,
            'reason': str,
            'equity': float,
            'regt_buying_power': float,
        }
    """
    client = get_alpaca_client()

    try:
        account = client.get_account()
        equity = float(account.equity)
        regt_bp = float(account.regt_buying_power)
    except Exception as e:
        return {
            'allowed': False,
            'reason': f'Could not fetch account info: {e}',
            'equity': 0.0,
            'regt_buying_power': 0.0,
        }

    if equity < 2000.0:
        return {
            'allowed': False,
            'reason': f'Account equity ${equity:.2f} < $2,000 Reg T minimum for short selling',
            'equity': equity,
            'regt_buying_power': regt_bp,
        }

    try:
        asset = client.get_asset(symbol.upper())
        shortable = bool(asset.shortable)
        easy_to_borrow = bool(asset.easy_to_borrow)
    except Exception as e:
        return {
            'allowed': False,
            'reason': f'Could not fetch asset info for {symbol}: {e}',
            'equity': equity,
            'regt_buying_power': regt_bp,
        }

    if not shortable:
        return {
            'allowed': False,
            'reason': f'{symbol} is not shortable',
            'equity': equity,
            'regt_buying_power': regt_bp,
        }

    if not easy_to_borrow:
        return {
            'allowed': False,
            'reason': f'{symbol} is not easy_to_borrow (high locate fees / forced buy-in risk)',
            'equity': equity,
            'regt_buying_power': regt_bp,
        }

    return {
        'allowed': True,
        'reason': 'Short selling allowed',
        'equity': equity,
        'regt_buying_power': regt_bp,
    }


def get_short_safe_buying_power(safety_factor: float = 0.80) -> float:
    """Return the safe buying power for short orders.

    Applies a safety factor to RegT buying power so short orders never
    fail Alpaca's ~3% short collar margin buffer.

    Parameters
    ----------
    safety_factor : float
        Fraction of RegT buying power to use (default 0.80 = 80%).

    Returns
    -------
    float
        Safe buying power in dollars.
    """
    try:
        account = get_alpaca_client().get_account()
        return float(account.regt_buying_power) * safety_factor
    except Exception:
        return 0.0
