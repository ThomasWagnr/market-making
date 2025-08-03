from abc import ABC, abstractmethod
from order_book import OrderBook

class BaseStrategy(ABC):
    """Abstract base class for a market making strategy."""

    @abstractmethod
    def calculate_quotes(self, order_book: OrderBook, inventory_position: float, time_horizon: float) -> tuple[float | None, float | None]:
        """
        Calculates the bid and ask prices the bot should quote.

        Args:
            order_book: The current order book state.
            inventory_position: The bot's current inventory of the asset.
            time_horizon: The fraction of the trading session remaining (1.0 to 0.0).

        Returns:
            A tuple containing the desired (bid_price, ask_price).
            Returns (None, None) if no quote should be made.
        """
        pass