import math
import numpy as np
import pandas as pd
from collections import deque
from scipy.stats import linregress
from .base_strategy import BaseStrategy
from order_book import OrderBook
import logging

logger = logging.getLogger(__name__)
TICK_SIZE = 0.001

class AvellanedaStoikovStrategy(BaseStrategy):
    """Implements the Avellaneda-Stoikov market making model."""

    def __init__(self, 
                gamma: float = 10, 
                lookback_period: int = 100, 
                ewma_span: int = 20, 
                trend_skew: bool = True,
                trend_window: int = 20,
                max_skew: float = 0.005,
                k_scaling_factor: float = 10.0):

        self.gamma = gamma
        self.lookback_period = lookback_period
        self.trend_skew = trend_skew
        self.ewma_span = ewma_span
        self.trend_window = trend_window
        self.max_skew = max_skew
        self.k_scaling_factor = k_scaling_factor
        self.mid_price_history = deque(maxlen=lookback_period)

    def _estimate_liquidity(self, order_book: OrderBook) -> float:
        """
        Estimates the liquidity parameter 'k' by calculating the
        Volume-Weighted Average Spread (VWAS) of the top book levels.
        """

        mid_price = order_book.mid_price

        # --- VWAS Calculation ---
        total_bid_volume = 0
        weighted_bid_distance = 0
        # Look at the top 5 levels of the bid book
        for price_key, size in list(order_book.bids.items())[:5]:
            price = -price_key
            distance_from_mid = mid_price - price
            weighted_bid_distance += size * distance_from_mid
            total_bid_volume += size

        total_ask_volume = 0
        weighted_ask_distance = 0
        # Look at the top 5 levels of the ask book
        for price, size in list(order_book.asks.items())[:5]:
            distance_from_mid = price - mid_price
            weighted_ask_distance += size * distance_from_mid
            total_ask_volume += size

        # Avoid division by zero if a side of the book is empty
        if total_bid_volume == 0 or total_ask_volume == 0:
            # Fallback to a simple spread-based measure if the book is one-sided
            return 1.0

        # Calculate the average distance of liquidity from the mid-price for each side
        avg_bid_distance = weighted_bid_distance / total_bid_volume
        avg_ask_distance = weighted_ask_distance / total_ask_volume
        vwas = avg_bid_distance + avg_ask_distance

        # --- Convert VWAS to k ---
        # k should be inversely proportional to the VWAS. A wide VWAS means low liquidity (low k).
        # The scaling factor is a key tuning parameter that translates VWAS into the right magnitude for 'k'.
        if vwas < 1e-9:
            return 1000.0 # High liquidity if spread is zero

        k = 1.0 / vwas
        return max(1.0, k * self.k_scaling_factor)

    def _calculate_trend_skew(self) -> float:
        """Calculates trend by fitting a regression line to recent mid-prices."""
        if len(self.mid_price_history) < self.trend_window:
            return 0.0

        recent_prices = list(self.mid_price_history)[-self.trend_window:]

        time_steps = np.arange(self.trend_window)
        regression = linregress(time_steps, recent_prices)
        slope = regression.slope
        
        return np.clip(slope, -self.max_skew, self.max_skew)
    
    def calculate_quotes(self, order_book: OrderBook, inventory_position: float, time_horizon: float) -> tuple[float | None, float | None]:
        mid_price = order_book.mid_price
        if mid_price is None:
            return None, None

        self.mid_price_history.append(mid_price)

        if len(self.mid_price_history) < self.lookback_period:
            logger.info(f"Strategy warming up. Data collected: {len(self.mid_price_history)}/{self.lookback_period}")
            return None, None

        price_series = pd.Series(list(self.mid_price_history))
        price_changes = price_series.diff().dropna()
        measured_volatility = price_changes.ewm(span=self.ewma_span).std().iloc[-1]
        effective_volatility = max(measured_volatility, 0.0001) if pd.notna(measured_volatility) else 0.0001

        k = self._estimate_liquidity(order_book)

        skew = self._calculate_trend_skew() if self.trend_skew else 0.0
        inventory_term = inventory_position * self.gamma * effective_volatility**2 * time_horizon
        reservation_price = mid_price - inventory_term + skew

        liquidity_term = math.log(1 + self.gamma / k)
        spread = self.gamma * effective_volatility**2 * time_horizon + (2 / self.gamma) * liquidity_term

        our_bid = self._round_to_tick(reservation_price - (spread / 2))
        our_ask = self._round_to_tick(reservation_price + (spread / 2))

        if our_bid >= our_ask or our_bid < 0 or our_ask > 1:
            logger.warning(f"Strategy calculated invalid quotes. Bid: {our_bid:.3f}, Ask: {our_ask:.3f}. Standing down.")
            return None, None

        logger.info(f"Strategy calculated quotes. Bid: {our_bid:.3f}, Ask: {our_ask:.3f}")
        return our_bid, our_ask

    def _round_to_tick(self, price: float) -> float:
        return round(price / TICK_SIZE) * TICK_SIZE