from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from order_book import OrderBook

class BaseStrategy(ABC):
    """Abstract base class for a market making strategy."""

    @abstractmethod
    def calculate_quotes(
        self,
        order_book: OrderBook,
        inventory_position: float,
        time_horizon: float,
        total_bid_size: float,
        total_ask_size: float,
        current_time=None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Calculates the bid and ask quotes the bot should place.

        Returns:
            (bid_quotes, ask_quotes) where each is a list of dicts:
            {'price': float, 'size': float}
        """
        raise NotImplementedError

    @abstractmethod
    def get_size_tolerance(self, order_book: OrderBook, active_order: Dict[str, Any]) -> float:
        """Calculates the re-quoting size tolerance for a given active order."""
        raise NotImplementedError