import logging
import itertools
from datetime import datetime, timezone
from typing import List, Dict, Any, Callable

from execution_client import ExecutionClient
from order_book import OrderBook

logger = logging.getLogger(__name__)

class SimulatedExchange(ExecutionClient):
    """
    A simulated execution client for backtesting.
    """
    def __init__(self, orders_list: List[Dict[str, Any]]):
        self.simulated_orders_log = orders_list
        self.order_id_generator = itertools.count(1)
        self.current_timestamp = None
        self._live_order_ids: set[int] = set()

    def set_time(self, timestamp: float):
        """Updates the internal clock of the simulated exchange."""
        self.current_timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc)

    def place_order(self, side: str, price: float, size: float, order_book: OrderBook) -> int | None:
        """Logs order placement and returns an order id; no internal queue state is kept."""
        order_id = next(self.order_id_generator)
        self._live_order_ids.add(order_id)

        # For diagnostics only: compute volume ahead snapshot from the book
        if side.upper() == "BUY":
            volume_ahead = sum(s for p, s in order_book.bids.items() if -p >= price)
        else:  # SELL
            volume_ahead = sum(s for p, s in order_book.asks.items() if p <= price)

        self.simulated_orders_log.append({
            'timestamp': self.current_timestamp.isoformat() if self.current_timestamp else None,
            'action': 'PLACE', 'order_id': order_id,
            'side': side.upper(), 'price': price, 'size': size,
            'initial_volume_ahead': float(volume_ahead)
        })
        logger.debug(
            "SimExchange: Placed %s order %s for %.2f @ %.3f. Snapshot vol ahead: %.2f",
            side, order_id, size, price, volume_ahead
        )
        return order_id

    def cancel_order(self, order_id: int) -> bool:
        """Logs cancellation; no internal queue state is kept."""
        if order_id in self._live_order_ids:
            self._live_order_ids.remove(order_id)
            self.simulated_orders_log.append({
                'timestamp': self.current_timestamp.isoformat() if self.current_timestamp else None,
                'action': 'CANCEL', 'order_id': order_id
            })
            return True
        # Even if unknown, log the intent for completeness
        self.simulated_orders_log.append({
            'timestamp': self.current_timestamp.isoformat() if self.current_timestamp else None,
            'action': 'CANCEL', 'order_id': order_id, 'note': 'unknown_order_id'
        })
        return False

    def check_for_fills(self, historical_trade: dict, bot_check_fills_func: Callable):
        """
        Unified matching: simply forward the trade event to the bot's fill checker.
        The bot owns queue position tracking and fill processing.
        """
        bot_check_fills_func(historical_trade)