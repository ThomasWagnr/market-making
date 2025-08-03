import math
import numpy as np
from collections import deque
from .base_strategy import BaseStrategy
from order_book import OrderBook

TICK_SIZE = 0.001

class AvellanedaStoikovStrategy(BaseStrategy):
    """Implements the Avellaneda-Stoikov market making model."""

    def __init__(self, gamma: float = 0.1, lookback_period: int = 100):
        self.gamma = gamma
        self.lookback_period = lookback_period
        self.mid_price_history = deque(maxlen=lookback_period)
        self.volatility = 0.0
    
    def calculate_quotes(self, order_book: OrderBook, inventory_position: float, time_horizon: float) -> tuple[float | None, float | None]:
        best_bid = order_book.best_bid
        best_ask = order_book.best_ask

        if best_bid is None or best_ask is None:
            return None, None

        mid_price = (best_bid + best_ask) / 2.0
        self.mid_price_history.append(mid_price)

        if len(self.mid_price_history) < self.lookback_period:
            return None, None

        self.volatility = np.std(list(self.mid_price_history))
        if self.volatility == 0:
            return None, None

        inventory_term = inventory_position * self.gamma * self.volatility**2 * time_horizon
        reservation_price = mid_price - inventory_term

        # Assuming a constant liquidity parameter 'k' for simplicity
        liquidity_term = math.log(1 + self.gamma / 1.5)
        spread = self.gamma * self.volatility**2 * time_horizon + (2 / self.gamma) * liquidity_term

        our_bid = self._round_to_tick(reservation_price - (spread / 2))
        our_ask = self._round_to_tick(reservation_price + (spread / 2))
        
        return our_bid, our_ask

    def _round_to_tick(self, price: float) -> float:
        return round(price / TICK_SIZE) * TICK_SIZE