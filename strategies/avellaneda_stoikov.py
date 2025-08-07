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
                max_layers: int = 3,
                max_size_tolerance_pct: float = 0.80,
                min_size_tolerance_pct: float = 0.20,
                patience_depth_factor: float = 0.8,
                book_depth_ma_window: int = 100,
                liquidity_fraction: float = 0.7):

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
        self.max_size_tolerance_pct = max_size_tolerance_pct
        self.min_size_tolerance_pct = min_size_tolerance_pct
        self.patience_depth_factor = patience_depth_factor
        self.liquidity_fraction = liquidity_fraction
        self.book_depth_history = deque(maxlen=book_depth_ma_window)
        self.mid_price_history = deque(maxlen=lookback_period)
        self.inventory_skew_intensity = 0.0

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

    def _update_dynamic_parameters(self, order_book: OrderBook) -> float:
        """
        Calculates and returns the max_patience_depth based on a moving average of book depth.
        """
        top_bids_volume = sum(size for _, size in list(order_book.bids.items())[:5])
        top_asks_volume = sum(size for _, size in list(order_book.asks.items())[:5])
        self.book_depth_history.append(top_bids_volume + top_asks_volume)
        
        if len(self.book_depth_history) < self.book_depth_history.maxlen:
            return 500.0 # Return a safe default during warm-up

        average_depth = np.mean(self.book_depth_history)
        return average_depth * self.patience_depth_factor

    def get_size_tolerance(self, order_book: OrderBook, active_order: Dict[str, Any]) -> float:
        """Calculates a dynamic tolerance based on the order's queue position."""
        max_patience_depth = self._update_dynamic_parameters(order_book)
        volume_ahead = active_order.get('volume_ahead', max_patience_depth)
        
        queue_depth_ratio = min(1.0, volume_ahead / (max_patience_depth + 1e-9))
        
        # Linearly interpolate between the max and min tolerance based on queue depth
        dynamic_tolerance = self.max_size_tolerance_pct - \
                            (queue_depth_ratio * (self.max_size_tolerance_pct - self.min_size_tolerance_pct))
        return dynamic_tolerance

    def _calculate_trend_skew(self) -> float:
        """Calculates trend by fitting a regression line to recent mid-prices."""
        if len(self.mid_price_history) < self.trend_window:
            return 0.0

        recent_prices = list(self.mid_price_history)[-self.trend_window:]

        time_steps = np.arange(self.trend_window)
        regression = linregress(time_steps, recent_prices)
        slope = regression.slope
        
        return np.clip(slope, -self.max_skew, self.max_skew)

    def _create_quote_ladder(self, total_size: float, best_price: float, side: str, 
                             tick_size: float, order_book: OrderBook) -> List[Dict[str, Any]]:
        """
        Builds an opportunistic ladder. It maximizes the top layer based on liquidity,
        then distributes any remaining size to deeper layers.
        """
        quotes = []
        if total_size < 1.0:
            return quotes

        # --- Layer 1: The Opportunistic Top Layer ---
        if side == "BUY":
            # Liquidity for a new bid is capped by the existing best bid's size
            available_liquidity = order_book.bids.get(-order_book.best_bid, 0) if order_book.best_bid else 0
        else: # SELL
            available_liquidity = order_book.asks.get(order_book.best_ask, 0) if order_book.best_ask else 0

        # The size of our top layer is the smaller of our total desired size or the liquidity cap
        size_layer_1 = min(total_size, available_liquidity * self.liquidity_fraction)
        
        # Place the first layer if its size is meaningful
        if size_layer_1 >= 1e-9:
            p = self._round_bid(best_price, tick_size) if side == "BUY" else self._round_ask(best_price, tick_size)
            if side == "BUY" and order_book.best_ask is not None and p >= order_book.best_ask:
                p = self._round_bid(order_book.best_ask - tick_size, tick_size)
                p = max(0.0, p)
            elif side == "SELL" and order_book.best_bid is not None and p <= order_book.best_bid:
                p = self._round_ask(order_book.best_bid + tick_size, tick_size)
                p = min(1.0, p)
            quotes.append({'price': p, 'size': round(size_layer_1, 2)})

        # --- Deeper Layers: Distribute the Remainder ---
        remaining_size = total_size - size_layer_1
        if remaining_size < 1.0 or self.max_layers <= 1:
            return quotes

        current_price = best_price
        
        # Distribute the remaining size evenly across the deeper layers
        num_deeper_layers = self.max_layers - 1
        if num_deeper_layers > 0:
            size_per_deeper_layer = remaining_size / num_deeper_layers

            for _ in range(num_deeper_layers):
                if remaining_size < 1.0: break

                if side == "BUY":
                    current_price -= self.layer_price_step * tick_size
                else: # SELL
                    current_price += self.layer_price_step * tick_size

                size_for_this_layer = min(remaining_size, size_per_deeper_layer)
                if size_for_this_layer >=1e-9:
                    p = self._round_bid(current_price, tick_size) if side == "BUY" else self._round_ask(current_price, tick_size)
                    if side == "BUY":
                        if order_book.best_ask is not None and p >= order_book.best_ask:
                            p = self._round_bid(order_book.best_ask - tick_size, tick_size)
                        p = max(0.0, p)
                    else:
                        if order_book.best_bid is not None and p <= order_book.best_bid:
                            p = self._round_ask(order_book.best_bid + tick_size, tick_size)
                        p = min(1.0, p)
                    quotes.append({'price': p, 'size': round(size_for_this_layer, 2)})
                remaining_size -= size_for_this_layer
        
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
        reservation_price = mid_price - inventory_term - (inventory_position * getattr(self, 'inventory_skew_intensity', 0.0)) + skew
        liquidity_term = math.log(1 + self.gamma / k)
        spread = self.gamma * effective_volatility**2 * time_horizon + (2 / self.gamma) * liquidity_term
        
        # Consolidated rounding and cross-prevention logic
        ideal_bid = reservation_price - (spread / 2)
        ideal_ask = reservation_price + (spread / 2)
        final_bid = self._round_bid(ideal_bid, order_book.tick_size)
        final_ask = self._round_ask(ideal_ask, order_book.tick_size)
        if order_book.best_ask is not None and final_bid >= order_book.best_ask:
            final_bid = self._round_bid(order_book.best_ask - order_book.tick_size, order_book.tick_size)
        if order_book.best_bid is not None and final_ask <= order_book.best_bid:
            final_ask = self._round_ask(order_book.best_bid + order_book.tick_size, order_book.tick_size)
        if final_bid >= final_ask or final_bid < 0 or final_ask > 1:
            return [], []

        if self.enable_layering:
            bid_quotes = self._create_quote_ladder(total_bid_size, final_bid, "BUY", order_book.tick_size, order_book)
            ask_quotes = self._create_quote_ladder(total_ask_size, final_ask, "SELL", order_book.tick_size, order_book)
        else:
            bid_quotes = [{'price': final_bid, 'size': total_bid_size}]
            ask_quotes = [{'price': final_ask, 'size': total_ask_size}]

        return bid_quotes, ask_quotes

    def _round_bid(self, price: float, tick_size: float) -> float:
        if tick_size <= 0: return price
        return math.floor(price / tick_size) * tick_size

    def _round_ask(self, price: float, tick_size: float) -> float:
        if tick_size <= 0: return price
        return math.ceil(price / tick_size) * tick_size    