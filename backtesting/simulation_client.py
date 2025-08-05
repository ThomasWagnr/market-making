import logging
import itertools
from datetime import datetime, timezone
from typing import List, Dict, Any, Callable

# Note the '..' for importing from the parent directory
from ..execution_client import ExecutionClient

logger = logging.getLogger(__name__)

class SimulatedExchange(ExecutionClient):
    """
    A simulated execution client for backtesting. It uses a stream of
    historical public trades to determine if and when simulated orders are filled,
    correctly handling partial fills.
    """
    def __init__(self, orders_list: List[Dict[str, Any]]):
        self.active_bids: Dict[int, Dict[str, Any]] = {}
        self.active_asks: Dict[int, Dict[str, Any]] = {}
        self.simulated_orders_log = orders_list
        self.order_id_generator = itertools.count(1)
        self.current_timestamp = None

    def set_time(self, timestamp: float):
        """Updates the internal clock of the simulated exchange."""
        self.current_timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc)

    def place_order(self, side: str, price: float, size: float) -> int | None:
        """Simulates placing an order by adding it to the active order list."""
        order_id = next(self.order_id_generator)
        
        order_details = {'price': price, 'size': size}
        
        if side.upper() == "BUY":
            self.active_bids[order_id] = order_details
        else: # SELL
            self.active_asks[order_id] = order_details
            
        self.simulated_orders_log.append({
            'timestamp': self.current_timestamp.isoformat() if self.current_timestamp else None,
            'action': 'PLACE', 'order_id': order_id,
            'side': side.upper(), 'price': price, 'size': size
        })
        logger.debug(f"SimExchange: Placed {side} order {order_id} for {size:.2f} @ {price:.3f}")
        return order_id

    def cancel_order(self, order_id: int) -> bool:
        """Simulates canceling an order by removing it from the active list."""
        if order_id in self.active_bids:
            del self.active_bids[order_id]
            self.simulated_orders_log.append({
                'timestamp': self.current_timestamp.isoformat() if self.current_timestamp else None,
                'action': 'CANCEL', 'order_id': order_id
            })
            return True
        elif order_id in self.active_asks:
            del self.active_asks[order_id]
            self.simulated_orders_log.append({
                'timestamp': self.current_timestamp.isoformat() if self.current_timestamp else None,
                'action': 'CANCEL', 'order_id': order_id
            })
            return True
        return False

    def check_for_fills(self, historical_trade: dict, bot_check_fills_func: Callable):
        """
        The core matching logic. This is called by the backtest engine for every
        historical trade to see if it would fill any of our simulated orders,
        handling partial fills for both our orders and the historical trade.
        """
        trade_price = float(historical_trade['price'])
        remaining_trade_size = float(historical_trade['size'])

        # --- Check for Ask Fills (Our Sell Orders) ---
        # Sort our asks by price (lowest price first) to ensure price priority
        sorted_ask_ids = sorted(self.active_asks, key=lambda oid: self.active_asks[oid]['price'])
        
        for order_id in sorted_ask_ids:
            if remaining_trade_size <= 1e-9: break  # Historical trade is fully consumed

            order = self.active_asks[order_id]
            if trade_price >= order['price']:
                # A fill occurs
                fill_size = min(remaining_trade_size, order['size'])
                
                logger.debug(f"SimExchange: Matched ASK order {order_id} for {fill_size:.2f} @ {order['price']:.3f}")
                
                # Tell the bot it has a fill for the specific amount
                bot_check_fills_func({'price': order['price'], 'size': fill_size})
                
                # Reduce our order's remaining size and the trade's remaining size
                self.active_asks[order_id]['size'] -= fill_size
                remaining_trade_size -= fill_size
                
                # If our order is fully filled, remove it from the active list
                if self.active_asks[order_id]['size'] <= 1e-9:
                    del self.active_asks[order_id]

        # --- Check for Bid Fills (Our Buy Orders) ---
        # Sort our bids by price (highest price first) to ensure price priority
        sorted_bid_ids = sorted(self.active_bids, key=lambda oid: self.active_bids[oid]['price'], reverse=True)

        for order_id in sorted_bid_ids:
            if remaining_trade_size <= 1e-9: break # Historical trade is fully consumed

            order = self.active_bids[order_id]
            if trade_price <= order['price']:
                fill_size = min(remaining_trade_size, order['size'])
                
                logger.debug(f"SimExchange: Matched BID order {order_id} for {fill_size:.2f} @ {order['price']:.3f}")
                
                bot_check_fills_func({'price': order['price'], 'size': fill_size})
                
                self.active_bids[order_id]['size'] -= fill_size
                remaining_trade_size -= fill_size
                
                if self.active_bids[order_id]['size'] <= 1e-9:
                    del self.active_bids[order_id]