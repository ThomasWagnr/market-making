# execution_client.py

import logging
import itertools
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ExecutionClient(ABC):
    """Abstract base class for an execution client."""

    @abstractmethod
    def place_order(self, side: str, price: float, size: float) -> int | None:
        """Places an order on the exchange."""
        raise NotImplementedError
    
    @abstractmethod
    def cancel_order(self, order_id: int) -> bool:
        """Cancels an order on the exchange. Returns True on success."""
        raise NotImplementedError

class DryRunExecutionClient(ExecutionClient):
    """A simulated execution client that tracks order state and logs actions."""
    
    def __init__(self, orders_list: List[Dict[str, Any]], fills_list: List[Dict[str, Any]]):
        self.order_id_generator = itertools.count(1)
        self.simulated_orders_log = orders_list 
        self.simulated_fills_log = fills_list   
        
        self.active_orders: Dict[int, Dict[str, Any]] = {}

    def place_order(self, side: str, price: float, size: float) -> int | None:
        """Simulates placing an order and adds it to the active order list."""
        order_id = next(self.order_id_generator)
        logger.info(f"[DRY RUN] PLACING {side} order for {size:.2f} @ {price:.3f} (Simulated ID: {order_id})")
        
        order_details = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': 'PLACE', 'order_id': order_id,
            'side': side, 'price': price, 'size': size
        }
        self.simulated_orders_log.append(order_details)
        
        self.active_orders[order_id] = order_details
        
        return order_id
        
    def cancel_order(self, order_id: int) -> bool:
        """Simulates canceling an order and removes it from the active list."""
        if order_id in self.active_orders:
            logger.info(f"[DRY RUN] CANCELING order {order_id}")
            self.simulated_orders_log.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'action': 'CANCEL', 'order_id': order_id
            })
            del self.active_orders[order_id]
            return True
        else:
            logger.warning(f"[DRY RUN] Attempted to cancel non-existent order ID: {order_id}")
            return False