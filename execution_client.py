import logging
import itertools
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class ExecutionClient:
    """Abstract base class for an execution client."""
    def place_order(self, side: str, price: float, size: float) -> int | None:
        raise NotImplementedError
    
    def cancel_order(self, order_id: int):
        raise NotImplementedError

class DryRunExecutionClient(ExecutionClient):
    """A simulated execution client that just logs actions."""
    def __init__(self, orders_list: list, fills_list: list):
        self.order_id_generator = itertools.count(1)
        self.simulated_orders = orders_list # Reference to the bot's list
        self.simulated_fills = fills_list   # Reference to the bot's list

    def place_order(self, side: str, price: float, size: float) -> int | None:
        order_id = next(self.order_id_generator)
        logger.info(f"[DRY RUN] PLACING {side} order for {size:.2f} @ {price:.3f} (Simulated ID: {order_id})")
        
        self.simulated_orders.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': 'PLACE', 'order_id': order_id,
            'side': side, 'price': price, 'size': size
        })
        return order_id
        
    def cancel_order(self, order_id: int):
        logger.info(f"[DRY RUN] CANCELING order {order_id}")
        self.simulated_orders.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': 'CANCEL', 'order_id': order_id
        })