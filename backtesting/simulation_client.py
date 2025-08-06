import logging
import itertools
from datetime import datetime, timezone
from typing import List, Dict, Any, Callable

from execution_client import ExecutionClient
from order_book import OrderBook

logger = logging.getLogger(__name__)

class SimulatedExchange(ExecutionClient):
    """
    A simulated execution client for backtesting that models queue priority (FIFO).
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

    def place_order(self, side: str, price: float, size: float, order_book: OrderBook) -> int | None:
        """Simulates placing an order and records its position in the queue."""
        order_id = next(self.order_id_generator)

        volume_ahead = 0.0
        
        if side.upper() == "BUY":
            # For a bid, volume ahead is the existing volume AT or BETTER (higher) than our price
            volume_ahead = sum(s for p, s in order_book.bids.items() if -p >= price)
            self.active_bids[order_id] = {'price': price, 'size': size, 'volume_ahead': volume_ahead}
        else: # SELL
            # For an ask, volume ahead is the existing volume AT or BETTER (lower) than our price
            volume_ahead = sum(s for p, s in order_book.asks.items() if p <= price)
            self.active_asks[order_id] = {'price': price, 'size': size, 'volume_ahead': volume_ahead}
            
            
        self.simulated_orders_log.append({
            'timestamp': self.current_timestamp.isoformat() if self.current_timestamp else None,
            'action': 'PLACE', 'order_id': order_id,
            'side': side.upper(), 'price': price, 'size': size,
            'initial_volume_ahead': volume_ahead
        })
        logger.debug(f"SimExchange: Placed {side} order {order_id} for {size:.2f} @ {price:.3f}. Volume ahead in queue: {volume_ahead:.2f}")
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
        handling partial fills and FIFO queue priority.
        """
        trade_price = float(historical_trade['price'])
        remaining_trade_size = float(historical_trade['size'])

        # --- Check for Ask Fills (Our Sell Orders) ---
        # Sort our asks by price (lowest price first) to ensure price priority
        sorted_ask_ids = sorted(self.active_asks, key=lambda oid: self.active_asks[oid]['price'])
        
        for order_id in sorted_ask_ids:
            if remaining_trade_size <= 1e-9: break
            if order_id not in self.active_asks: continue

            order = self.active_asks[order_id]
            if trade_price >= order['price']:
                eats_ahead = min(remaining_trade_size, order['volume_ahead'])
                order['volume_ahead'] -= eats_ahead
                remaining_trade_size -= eats_ahead

                if remaining_trade_size <= 1e-9: continue # Trade was consumed by the queue

                # If we are at the front of the queue, we can get filled
                if order['volume_ahead'] <= 1e-9:
                    fill_size = min(remaining_trade_size, order['size'])
                    bot_check_fills_func({'price': order['price'], 'size': fill_size, 'asset_id': historical_trade['asset_id']})
                    
                    order['size'] -= fill_size
                    remaining_trade_size -= fill_size
                    
                    if order['size'] <= 1e-9:
                        del self.active_asks[order_id]

        # --- Check for Bid Fills (Our Buy Orders) ---
        # Sort our bids by price (highest price first) to ensure price priority
        sorted_bid_ids = sorted(self.active_bids, key=lambda oid: self.active_bids[oid]['price'], reverse=True)

        for order_id in sorted_bid_ids:
            if remaining_trade_size <= 1e-9: break
            if order_id not in self.active_bids: continue

            order = self.active_bids[order_id]
            if trade_price <= order['price']:
                eats_ahead = min(remaining_trade_size, order['volume_ahead'])
                order['volume_ahead'] -= eats_ahead
                remaining_trade_size -= eats_ahead

                if remaining_trade_size <= 1e-9: continue

                if order['volume_ahead'] <= 1e-9:
                    fill_size = min(remaining_trade_size, order['size'])
                    bot_check_fills_func({'price': order['price'], 'size': fill_size, 'asset_id': historical_trade['asset_id']})
                    
                    order['size'] -= fill_size
                    remaining_trade_size -= fill_size
                    
                    if order['size'] <= 1e-9:
                        del self.active_bids[order_id]