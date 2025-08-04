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

    def __init__(self, gamma: float = 10, lookback_period: int = 100, ewma_span: int = 20, trend_skew: bool = True):
        self.gamma = gamma
        self.lookback_period = lookback_period
        self.trend_skew = trend_skew
        self.ewma_span = ewma_span
        self.mid_price_history = deque(maxlen=lookback_period)

    def _estimate_liquidity(self, order_book: OrderBook) -> float:
        """Estimates the liquidity parameter 'k' from the order book depth."""
        
        mid_price = order_book.mid_price
        market_spread = order_book.spread
        base_range = mid_price * 0.02
        price_range = max(base_range, market_spread)

        bid_volume_in_range = sum(size for price, size in order_book.bids.items()
                                if mid_price - (-price) <= price_range) # Note: price is negative key
        ask_volume_in_range = sum(size for price, size in order_book.asks.items() 
                                if price - mid_price <= price_range)
        
        # Simple estimation for k (volume per unit of price)
        k = (bid_volume_in_range + ask_volume_in_range) / (2 * price_range + 1e-9)
        
        # We can cap or scale k to keep it within a reasonable range
        return max(1.0, k / 1000) # Scale down the raw volume

    def _calculate_trend_skew(self) -> float:
        """Calculates trend by fitting a regression line to recent mid-prices."""
        trend_window = 20
        if len(self.mid_price_history) < trend_window:
            return 0.0
        
        recent_prices = list(self.mid_price_history)[-trend_window:]
        time_steps = np.arange(trend_window)
        regression = linregress(time_steps, recent_prices)
        slope = regression.slope
        max_skew = 0.005 # Cap the skew at half a cent
        
        return np.clip(slope, -max_skew, max_skew)

    def _get_calm_market_quotes(self, order_book: OrderBook) -> tuple[float, float]:
        """
        Calculates quotes for a calm market using a hybrid approach.
        It respects a minimum spread while trying to improve on the market price.
        """
        mid_price = order_book.mid_price
        
        minimum_spread = 2 * TICK_SIZE
        ideal_bid = mid_price - (minimum_spread / 2)
        ideal_ask = mid_price + (minimum_spread / 2)

        opportunistic_bid = order_book.best_bid + TICK_SIZE
        opportunistic_ask = order_book.best_ask - TICK_SIZE

        final_bid = max(ideal_bid, opportunistic_bid)
        final_ask = min(ideal_ask, opportunistic_ask)
        
        if final_bid >= final_ask:
            return self._round_to_tick(ideal_bid), self._round_to_tick(ideal_ask)

        return self._round_to_tick(final_bid), self._round_to_tick(final_ask)
    
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

        #if self.volatility == 0 or pd.isna(self.volatility):
        #    logger.info("Zero volatility detected. Using hybrid calm market quotes.")
        #    return self._get_calm_market_quotes(order_book)

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