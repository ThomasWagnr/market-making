import math
import numpy as np
import pandas as pd
from collections import deque
from scipy.stats import linregress
from typing import List, Dict, Any, Tuple
import logging

from .base_strategy import BaseStrategy
from order_book import OrderBook

logger = logging.getLogger(__name__)
MINIMUM_VOLATILITY = 0.0001

class AvellanedaStoikovStrategy(BaseStrategy):
    """
    Implements a highly configurable Avellaneda-Stoikov market making model with dynamic 
    liquidity estimation, EWMA volatility, optional trend skew, and optional quote layering.
    """

    def __init__(self, 
                gamma: float = 10, 
                lookback_period: int = 100, 
                ewma_span: int = 20, 
                enable_trend_skew: bool = True,
                enable_layering : bool = True,
                trend_window: int = 20,
                max_skew: float = 0.005,
                k_scaling_factor: float = 10.0,
                layer_price_step : int = 1,
                layer_size_ratio: float = 1.5,
                max_layers: int = 5):

        self.gamma = gamma
        self.lookback_period = lookback_period
        self.enable_trend_skew = enable_trend_skew
        self.enable_layering = enable_layering
        self.ewma_span = ewma_span
        self.trend_window = trend_window
        self.max_skew = max_skew
        self.k_scaling_factor = k_scaling_factor
        self.layer_price_step = layer_price_step
        self.layer_size_ratio = layer_size_ratio
        self.max_layers = max_layers
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

    def _create_quote_ladder(self, total_size: float, best_price: float, side: str, tick_size: float) -> List[Dict[str, Any]]:
        """Builds a ladder of quotes with progressively larger size."""
        quotes = []
        remaining_size = total_size
        current_price = best_price
        layer_num = 1
        
        if self.layer_size_ratio <= 1.0 or self.max_layers <= 1:
            base_size = total_size / self.max_layers if self.max_layers > 0 else total_size
        else:
            r_n = self.layer_size_ratio ** self.max_layers
            if (1 - r_n) == 0:
                base_size = total_size / self.max_layers
            else:
                base_size = total_size * (1 - self.layer_size_ratio) / (1 - r_n)

        current_layer_size = base_size
        
        while remaining_size > 1.0 and layer_num <= self.max_layers:
            size_for_this_layer = min(remaining_size, current_layer_size)
            
            quotes.append({'price': self._round_to_tick(current_price, tick_size), 'size': round(size_for_this_layer, 2)})
            
            remaining_size -= size_for_this_layer
            current_layer_size *= self.layer_size_ratio
            
            if side == "BUY":
                current_price -= self.layer_price_step * tick_size
            else: # SELL
                current_price += self.layer_price_step * tick_size
            
            layer_num += 1
            
        return quotes
    
    def calculate_quotes(self, order_book: OrderBook, inventory_position: float, 
                         time_horizon: float, total_bid_size: float, total_ask_size: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        
        mid_price = order_book.mid_price
        if mid_price is None:
            return [], []

        self.mid_price_history.append(mid_price)

        if len(self.mid_price_history) < self.lookback_period:
            logger.info(f"Strategy warming up. Data collected: {len(self.mid_price_history)}/{self.lookback_period}")
            return [], []

        price_series = pd.Series(list(self.mid_price_history))
        price_changes = price_series.diff().dropna()
        measured_volatility = price_changes.ewm(span=self.ewma_span).std().iloc[-1]
        effective_volatility = max(measured_volatility, MINIMUM_VOLATILITY) if pd.notna(measured_volatility) else MINIMUM_VOLATILITY

        k = self._estimate_liquidity(order_book)
        skew = self._calculate_trend_skew() if self.enable_trend_skew else 0.0

        inventory_term = inventory_position * self.gamma * effective_volatility**2 * time_horizon
        reservation_price = mid_price - inventory_term + skew

        liquidity_term = math.log(1 + self.gamma / k)
        spread = self.gamma * effective_volatility**2 * time_horizon + (2 / self.gamma) * liquidity_term

        best_bid_price = reservation_price - (spread / 2)
        best_ask_price = reservation_price + (spread / 2)

        if best_bid_price >= best_ask_price or best_bid_price < 0 or best_ask_price > 1:
            return [], []

        if self.enable_layering:
            bid_quotes = self._create_quote_ladder(total_bid_size, best_bid_price, "BUY", order_book.tick_size)
            ask_quotes = self._create_quote_ladder(total_ask_size, best_ask_price, "SELL", order_book.tick_size)
        else:
            bid_quotes = [{'price': self._round_to_tick(best_bid_price, order_book.tick_size), 'size': total_bid_size}]
            ask_quotes = [{'price': self._round_to_tick(best_ask_price, order_book.tick_size), 'size': total_ask_size}]
        
        return bid_quotes, ask_quotes

    def _round_to_tick(self, price: float, tick_size: float) -> float:
        """Rounds a price to the nearest valid tick size for the market."""
        if tick_size <= 0: return price
        return round(price / tick_size) * tick_size