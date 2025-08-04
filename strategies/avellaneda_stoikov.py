import math
import numpy as np
from collections import deque
from .base_strategy import BaseStrategy
from order_book import OrderBook
import logging

logger = logging.getLogger(__name__)
TICK_SIZE = 0.001

class AvellanedaStoikovStrategy(BaseStrategy):
    """Implements the Avellaneda-Stoikov market making model."""

    def __init__(self, gamma: float = 0.1, lookback_period: int = 100, trend_skew: bool = True):
        self.gamma = gamma
        self.lookback_period = lookback_period
        self.trend_skew = trend_skew
        self.mid_price_history = deque(maxlen=lookback_period)
        self.volatility = 0.0

    def _estimate_liquidity(self, order_book: OrderBook, mid_price: float) -> float:
        """Estimates the liquidity parameter 'k' from the order book depth."""
        # Look at orders within 2% of the mid-price
        price_range = mid_price * 0.02
        
        bid_volume_in_range = sum(size for price, size in order_book.bids.items() 
                                if mid_price - price <= price_range)
        ask_volume_in_range = sum(size for price, size in order_book.asks.items() 
                                if price - mid_price <= price_range)
        
        # Simple estimation for k (volume per unit of price)
        k = (bid_volume_in_range + ask_volume_in_range) / (2 * price_range + 1e-9)
        
        # We can cap or scale k to keep it within a reasonable range
        return max(1.0, k / 1000) # Scale down the raw volume
    
    def calculate_quotes(self, order_book: OrderBook, inventory_position: float, time_horizon: float) -> tuple[float | None, float | None]:
        best_bid = order_book.best_bid
        best_ask = order_book.best_ask

        if best_bid is None or best_ask is None:
            return None, None

        mid_price = order_book.mid_price
        self.mid_price_history.append(mid_price)

        if len(self.mid_price_history) < self.lookback_period:
            logger.info(f"Strategy warming up. Data collected: {len(self.mid_price_history)}/{self.lookback_period}")
            return None, None

        k = self._estimate_liquidity(order_book, mid_price)
         
        self.volatility = np.std(list(self.mid_price_history))
        if self.volatility == 0:
            return None, None

        skew = 0.0
        if self.trend_skew:
            if len(self.mid_price_history) > 20:
                long_term_price = self.mid_price_history[-20]
                short_term_price = self.mid_price_history[-1]
                # The skew will be a small positive number for an uptrend, negative for a downtrend
                skew = short_term_price - long_term_price

        inventory_term = inventory_position * self.gamma * self.volatility**2 * time_horizon
        reservation_price = mid_price - inventory_term + skew

        liquidity_term = math.log(1 + self.gamma / k)
        spread = self.gamma * self.volatility**2 * time_horizon + (2 / self.gamma) * liquidity_term

        our_bid = self._round_to_tick(reservation_price - (spread / 2))
        our_ask = self._round_to_tick(reservation_price + (spread / 2))

        if our_bid >= our_ask or our_bid < 0 or our_ask > 1:
            logger.warning(f"Strategy calculated invalid quotes. Bid: {our_bid:.3f}, Ask: {our_ask:.3f}. Standing down.")
            return None, None

        logger.info(f"Strategy calculated quotes. Bid: {our_bid:.3f}, Ask: {our_ask:.3f}")
        return our_bid, our_ask

    def _round_to_tick(self, price: float) -> float:
        return round(price / TICK_SIZE) * TICK_SIZE