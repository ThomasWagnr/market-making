import asyncio
import json
import logging
import itertools 
import aiohttp
import websockets
import csv
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from order_book import OrderBook
from trade_history import TradeHistory
from event_dispatcher import EventDispatcher
from utils import get_market_details 
from strategies.base_strategy import BaseStrategy
from execution_client import ExecutionClient
from backtesting.simulation_client import SimulatedExchange

logger = logging.getLogger(__name__)

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

class MarketMakerBot:
    """A generic bot runner that operates using a provided strategy module."""

    def __init__(self,
                 market_id: str,
                 strategy: BaseStrategy,
                 execution_client: ExecutionClient,
                 total_capital: float = 2000.0,
                 minting_capital_fraction: float = 0.5,
                 order_value_percentage: float = 0.05,
                 simulated_fills: List[Dict[str, Any]] = None,
                 simulated_taker_trades: List[Dict[str, Any]] = None):

        # --- Configuration ---
        self.market_id = market_id
        self.min_order_size: float = 1.0
        self.yes_token_id: Optional[str] = None
        self.no_token_id: Optional[str] = None
        self.strategy = strategy
        self.execution_client = execution_client
        self.is_running = True

        # --- NEW: Capital Allocation ---
        self.total_capital = total_capital
        self.order_value_percentage = order_value_percentage
        self.target_order_value = self.total_capital * self.order_value_percentage # The desired value of each quote
        self.starting_shares = self.total_capital * minting_capital_fraction # e.g., $1000 = 1000 YES and 1000 NO shares
        self.cash = self.total_capital - self.starting_shares # The other $1000 is for trading

        # --- Reporting & Simulation ---
        self.simulated_fills = simulated_fills if simulated_fills is not None else []
        self.simulated_taker_trades = simulated_taker_trades if simulated_taker_trades is not None else []

        # --- Performance Tracking ---
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_value = self.total_capital
        self.average_entry_price = 0.0
        
        # --- Live State ---
        self.inventory_position = 0.0
        self.active_bids : Dict[int, Dict[str, Any]] = {}
        self.active_asks : Dict[int, Dict[str, Any]] = {}

        # --- Market State ---
        self.time_horizon = 1.0
        self.market_start_time: Optional[datetime] = None
        self.market_close_time: Optional[datetime] = None
        self.session_duration_seconds: float = 0.0
        self._last_sim_time_epoch: Optional[float] = None

        # --- Component Initialization ---
        self.order_book: Optional[OrderBook] = None
        self.trade_history: Optional[TradeHistory] = None
        self.dispatcher: Optional[EventDispatcher] = None

        # --- Retry/Backoff Configuration ---
        self.initial_retry_delay = 1
        self.max_retry_delay = 32
        self.retry_multiplier = 2.0

        # --- Queue tracking caches (optimized incremental refresh) ---
        # Book last-seen sizes per exact price
        self._book_bids_size: Dict[float, float] = {}
        self._book_asks_size: Dict[float, float] = {}
        # Our own size aggregated per price
        self._our_bids_by_price: Dict[float, float] = {}
        self._our_asks_by_price: Dict[float, float] = {}
        # Better volume caches per our active price
        self._better_vol_at_bid_price: Dict[float, float] = {}
        self._better_vol_at_ask_price: Dict[float, float] = {}

    def _on_update(self, event_type: str, data: dict):
        """Core trigger, called by the dispatcher after any data update."""

        if not self.is_running:
            return

        # In backtests, the SimulatedExchange drives fills; avoid double-counting here.
        if event_type == "last_trade_price":
            if not isinstance(self.execution_client, SimulatedExchange):
                self._check_fills(data)

        # Maintain queue positions/cache on book updates
        if event_type == "price_change":
            self._refresh_queue_positions(data.get("changes", []))
        elif event_type == "book":
            self._rebuild_queue_caches()

        # In backtests, update time_horizon using the recorded sim time
        try:
            sim_time = data.get('sim_time') if isinstance(data, dict) else None
            if sim_time is not None and self.market_close_time and self.session_duration_seconds > 0:
                now_dt = datetime.fromtimestamp(float(sim_time), tz=timezone.utc)
                time_remaining = (self.market_close_time - now_dt).total_seconds()
                self.time_horizon = max(0.0, time_remaining / self.session_duration_seconds)
                # keep for shutdown timestamping
                self._last_sim_time_epoch = float(sim_time)
        except Exception:
            pass

        self._update_pnl()

        total_bid_size, total_ask_size = self._calculate_order_sizes()

        # Delegate quote calculation to the strategy module
        new_bids, new_asks = self.strategy.calculate_quotes(
            order_book=self.order_book,
            inventory_position=self.inventory_position,
            time_horizon=self.time_horizon,
            total_bid_size=total_bid_size,
            total_ask_size=total_ask_size,
            current_time=data.get('sim_time') if isinstance(data, dict) else None
        )

        state_str = (f"{self.order_book} | Inv: {self.inventory_position:+.1f} | "
                     f"P&L: ${self.realized_pnl:+.2f} | Equity: ${self.total_value:,.2f}")
        print(f"\r{state_str}", end="", flush=True)

        if new_bids or new_asks:
            self._update_orders(new_bids, new_asks)
        else:
            self._cancel_all_orders()
            

    def _update_pnl(self):
        """Calculates unrealized P&L and total equity."""
        mid_price = self.order_book.mid_price
        if not mid_price:
            return
        
        # 1. Calculate the market value of our TRADED inventory
        market_value_of_traded_inventory = self.inventory_position * mid_price
        
        # 2. The capital used to mint our starting shares is always worth its original value
        value_of_minted_assets = self.starting_shares

        # 3. The correct total equity is the sum of our cash, our minted assets, and our traded assets
        self.total_value = self.cash + value_of_minted_assets + market_value_of_traded_inventory
        
        # 4. Unrealized P&L is the gain/loss on the TRADED portion of our inventory
        if self.inventory_position != 0:
            cost_basis = self.inventory_position * self.average_entry_price
            self.unrealized_pnl = market_value_of_traded_inventory - cost_basis
        else:
            self.unrealized_pnl = 0.0

    def _calculate_order_sizes(self) -> tuple[float, float]:
        """
        Calculates the total target order sizes for each side, constrained by
        the bot's actual share inventory and available cash.
        """
        mid_price = self.order_book.mid_price
        if not mid_price or mid_price <= 1e-9:
            return 0.0, 0.0

        # 1. Determine the ideal size based on our target capital per quote
        ideal_size = self.target_order_value / mid_price

        # 2. Determine the HARD LIMITS based on our actual capital
        # Max we can sell is the number of YES shares we currently own
        # Inventory is long YES shares, so positive inventory adds to what we can sell.
        max_sellable_shares = self.starting_shares - self.inventory_position
        
        # Max we can buy is what our cash can afford
        max_buyable_shares = self.cash / mid_price

        # 3. Calculate final sizes, respecting inventory skew AND hard limits
        # The old inventory factors now represent our desire to trade, not the limit itself.
        inventory_factor_buy = 1.0 - (self.inventory_position / self.starting_shares)
        inventory_factor_sell = 1.0 + (self.inventory_position / self.starting_shares)
        
        # The total size we quote is our ideal size, skewed by inventory,
        # but capped by our absolute capital limits.
        total_bid_size = min(ideal_size * inventory_factor_buy, max_buyable_shares)
        total_ask_size = min(ideal_size * inventory_factor_sell, max_sellable_shares)

        # 4. Ensure sizes are not below the market minimum
        final_bid_size = total_bid_size if total_bid_size >= self.min_order_size else 0.0
        final_ask_size = total_ask_size if total_ask_size >= self.min_order_size else 0.0
        
        return final_bid_size, final_ask_size

    def _update_orders(self, new_bids: List[Dict[str, Any]], new_asks: List[Dict[str, Any]]):
        """
        Reconciles active orders with the new target quote ladder from the strategy,
        using a dynamic tolerance based on queue position to decide whether to re-quote.
        """
        # Create a map of target quotes for efficient lookup: {price: size}
        target_bids_map = {q['price']: q['size'] for q in new_bids}
        target_asks_map = {q['price']: q['size'] for q in new_asks}

        # --- Reconcile Bids ---
        # First, check existing orders to see if they should be kept or canceled.
        for order_id, order in list(self.active_bids.items()):
            should_cancel = False
            
            # Check if the order's price is still in our target ladder
            if order['price'] in target_bids_map:
                # If the price is correct, check if the size is significantly different.
                # To do this, we ask the strategy for the appropriate tolerance.
                tolerance = self.strategy.get_size_tolerance(self.order_book, order)
                target_size = target_bids_map[order['price']]
                
                if order['size'] > 0: # Avoid division by zero
                    size_diff_pct = abs(order['size'] - target_size) / order['size']
                    if size_diff_pct > tolerance:
                        # The size discrepancy is too large, so we should re-quote.
                        should_cancel = True
                # If the size difference is within tolerance, we do nothing to keep our queue priority.
            else:
                # The price is no longer optimal, so we must cancel.
                should_cancel = True
            
            if should_cancel:
                # Adjust our per-price aggregate before deletion
                if order_id in self.active_bids:
                    price = self.active_bids[order_id]['price']
                    remaining = self.active_bids[order_id]['size']
                    self._our_bids_by_price[price] = max(0.0, self._our_bids_by_price.get(price, 0.0) - remaining)
                self._cancel_order(order_id)
                if order_id in self.active_bids: del self.active_bids[order_id]
        
        # Now, place new bids for any target price levels where we don't have an order.
        active_bid_prices = {o['price'] for o in self.active_bids.values()}
        for quote in new_bids:
            if quote['price'] not in active_bid_prices and quote['size'] >= self.min_order_size:
                # Compute baseline components at placement time
                price = quote['price']
                size = quote['size']
                better_vol = sum(s for p, s in self.order_book.bids.items() if -p > price)
                book_at_price = self.order_book.bids.get(-price, 0.0)
                our_total_at_price = sum(o['size'] for o in self.active_bids.values() if o['price'] == price)
                equal_other_vol = max(0.0, book_at_price - our_total_at_price)
                our_prior_vol = our_total_at_price  # orders already present at this price are prior

                # Place order via execution client
                order_id = self._place_order("BUY", price, size)
                if order_id:
                    volume_ahead = better_vol + equal_other_vol + our_prior_vol
                    self.active_bids[order_id] = {
                        'price': price,
                        'size': size,
                        'volume_ahead': volume_ahead,
                        'baseline_equal_other_vol': equal_other_vol,
                        'baseline_our_prior_vol': our_prior_vol,
                    }
                    # Update caches for incremental refresh
                    self._our_bids_by_price[price] = self._our_bids_by_price.get(price, 0.0) + size
                    if price not in self._better_vol_at_bid_price:
                        self._better_vol_at_bid_price[price] = better_vol
                    self._book_bids_size[price] = book_at_price

        # --- Symmetrical Logic for Asks ---
        for order_id, order in list(self.active_asks.items()):
            should_cancel = False
            if order['price'] in target_asks_map:
                tolerance = self.strategy.get_size_tolerance(self.order_book, order)
                target_size = target_asks_map[order['price']]
                if order['size'] > 0:
                    size_diff_pct = abs(order['size'] - target_size) / order['size']
                    if size_diff_pct > tolerance:
                        should_cancel = True
            else:
                should_cancel = True
            
            if should_cancel:
                if order_id in self.active_asks:
                    price = self.active_asks[order_id]['price']
                    remaining = self.active_asks[order_id]['size']
                    self._our_asks_by_price[price] = max(0.0, self._our_asks_by_price.get(price, 0.0) - remaining)
                self._cancel_order(order_id)
                if order_id in self.active_asks: del self.active_asks[order_id]

        active_ask_prices = {o['price'] for o in self.active_asks.values()}
        for quote in new_asks:
            if quote['price'] not in active_ask_prices and quote['size'] >= self.min_order_size:
                price = quote['price']
                size = quote['size']
                better_vol = sum(s for p, s in self.order_book.asks.items() if p < price)
                book_at_price = self.order_book.asks.get(price, 0.0)
                our_total_at_price = sum(o['size'] for o in self.active_asks.values() if o['price'] == price)
                equal_other_vol = max(0.0, book_at_price - our_total_at_price)
                our_prior_vol = our_total_at_price

                order_id = self._place_order("SELL", price, size)
                if order_id:
                    volume_ahead = better_vol + equal_other_vol + our_prior_vol
                    self.active_asks[order_id] = {
                        'price': price,
                        'size': size,
                        'volume_ahead': volume_ahead,
                        'baseline_equal_other_vol': equal_other_vol,
                        'baseline_our_prior_vol': our_prior_vol,
                    }
                    self._our_asks_by_price[price] = self._our_asks_by_price.get(price, 0.0) + size
                    if price not in self._better_vol_at_ask_price:
                        self._better_vol_at_ask_price[price] = better_vol
                    self._book_asks_size[price] = book_at_price

    def _check_fills(self, trade: dict):
        """
        Checks for fills, respecting the simulated queue 
        position and updates inventory, cash, and P&L.
        """
        asset_id = trade.get('asset_id')
        if not asset_id: return
        trade_price = float(trade['price'])
        remaining_trade_size = float(trade['size'])
        # Pass along sim_time if present for consistent timestamps
        sim_time = trade.get('sim_time')

        # Prefer pre-normalized price if provided
        if 'equivalent_yes_price' in trade:
            equivalent_yes_price = float(trade['equivalent_yes_price'])
        else:
            if asset_id == self.yes_token_id:
                equivalent_yes_price = trade_price
            elif asset_id == self.no_token_id:
                equivalent_yes_price = 1.0 - trade_price
            else:
                return

        # --- Check for Bid Fills (Our Buy Orders) ---
        # Sort by price to process the best-priced orders first
        sorted_bid_ids = sorted(self.active_bids, key=lambda oid: self.active_bids[oid]['price'], reverse=True)
        for order_id in sorted_bid_ids:
            if remaining_trade_size <= 1e-9: break
            order = self.active_bids.get(order_id)
            if not order: continue

            if equivalent_yes_price <= order['price']:
                # This trade first fills orders ahead of us in the queue
                eats_ahead = min(remaining_trade_size, order['volume_ahead'])
                order['volume_ahead'] -= eats_ahead
                remaining_trade_size -= eats_ahead

                if remaining_trade_size <= 1e-9: continue

                # If we are at the front of the queue, we can get filled
                if order['volume_ahead'] <= 1e-9:
                    fill_size = min(remaining_trade_size, order['size'])
                    self._process_fill('BUY', order['price'], fill_size, order_id, sim_time)

        # --- Symmetrical Logic for Ask Fills (Our Sell Orders) ---
        sorted_ask_ids = sorted(self.active_asks, key=lambda oid: self.active_asks[oid]['price'])
        for order_id in sorted_ask_ids:
            if remaining_trade_size <= 1e-9: break
            order = self.active_asks.get(order_id)
            if not order: continue

            if equivalent_yes_price >= order['price']:
                eats_ahead = min(remaining_trade_size, order['volume_ahead'])
                order['volume_ahead'] -= eats_ahead
                remaining_trade_size -= eats_ahead

                if remaining_trade_size <= 1e-9: continue
                
                if order['volume_ahead'] <= 1e-9:
                    fill_size = min(remaining_trade_size, order['size'])
                    self._process_fill('SELL', order['price'], fill_size, order_id, sim_time)

    def _process_fill(self, side: str, price: float, size: float, order_id: int, sim_time=None):
        """Processes a single fill, updating inventory, P&L, and active orders."""
        # Normalize timestamp: prefer simulated time from data, else now
        if isinstance(sim_time, (int, float)):
            ts_iso = datetime.fromtimestamp(sim_time, tz=timezone.utc).isoformat()
        elif isinstance(sim_time, str):
            ts_iso = sim_time
        else:
            ts_iso = datetime.now(timezone.utc).isoformat()

        if side == 'BUY':
            active_order_book = self.active_bids
            
            if self.inventory_position < 0:
                self.realized_pnl += (self.average_entry_price - price) * size
            else:
                new_total_cost = (self.average_entry_price * self.inventory_position) + (price * size)
                new_inventory = self.inventory_position + size
                if abs(new_inventory) > 1e-9:
                    self.average_entry_price = new_total_cost / new_inventory
            
            self.inventory_position += size
            self.cash -= size * price
        else: # SELL
            active_order_book = self.active_asks

            if self.inventory_position > 0:
                self.realized_pnl += (price - self.average_entry_price) * size
            else:
                new_total_cost = (self.average_entry_price * abs(self.inventory_position)) + (price * size)
                new_inventory = abs(self.inventory_position) + size
                if abs(new_inventory) > 1e-9:
                    self.average_entry_price = new_total_cost / new_inventory
            
            self.inventory_position -= size
            self.cash += size * price
        
        # Update active order state
        if order_id in active_order_book:
            active_order_book[order_id]['size'] -= size
            active_order_book[order_id]['volume_ahead'] = 0.0
            if active_order_book[order_id]['size'] <= 1e-9:
                del active_order_book[order_id]
        
        if abs(self.inventory_position) < 1e-9:
            self.average_entry_price = 0.0

        # Update PnL after state changes and record enriched fill log
        self._update_pnl()
        self.simulated_fills.append({
            'timestamp': ts_iso,
            'side': side,
            'price': price,
            'size': size,
            'order_id': order_id,
            'equity': self.total_value,
            'inventory': self.inventory_position,
            'cash': self.cash
        })

        logger.info(f"FILL: {side} {size:.2f} @ {price:.3f}. New Inventory: {self.inventory_position:+.1f}")

    def _place_order(self, side: str, price: float, size: float) -> int | None:
        """Delegates order placement to the execution client."""
        # For a simulation, we pass the current order book for queue analysis
        if isinstance(self.execution_client, SimulatedExchange):
            return self.execution_client.place_order(side, price, size, self.order_book)
        else: # For a live client
            return self.execution_client.place_order(side, price, size)

    def _cancel_order(self, order_id: int):
        """Delegates order cancellation to the execution client."""
        return self.execution_client.cancel_order(order_id)

    def _cancel_all_orders(self):
        """Cancels all currently active orders on both sides."""
        for order_id in list(self.active_bids.keys()):
            self._cancel_order(order_id)
        self.active_bids.clear()
        
        for order_id in list(self.active_asks.keys()):
            self._cancel_order(order_id)
        self.active_asks.clear()

    def _refresh_queue_positions(self, changes: List[Dict[str, Any]]):
        """Incrementally updates queue positions using deltas from price_change events.

        - Maintains book size maps and better-volume caches per our active prices
        - Recomputes only affected price buckets
        - Does not count new arrivals at our price as ahead (caps equal-price others at baseline)
        """
        if not self.order_book or not changes:
            return

        touched_bid_prices: set[float] = set()
        touched_ask_prices: set[float] = set()

        # 1) Update last-seen book size maps and better-volume caches via deltas
        for ch in changes:
            side = ch.get('side', '').upper()
            price = float(ch.get('price'))
            new_size = float(ch.get('size'))
            if side == 'BUY':
                prev = self._book_bids_size.get(price, 0.0)
                delta = new_size - prev
                if delta != 0.0:
                    # update better volume cache for all our bid prices lower than this price
                    for p in list(self._better_vol_at_bid_price.keys()):
                        if p < price:
                            self._better_vol_at_bid_price[p] = max(0.0, self._better_vol_at_bid_price[p] + delta)
                    self._book_bids_size[price] = new_size
                    # any order at this price or below may be affected
                    touched_bid_prices.update([p for p in self._better_vol_at_bid_price.keys() if p <= price])
                    touched_bid_prices.add(price)
            elif side == 'SELL':
                prev = self._book_asks_size.get(price, 0.0)
                delta = new_size - prev
                if delta != 0.0:
                    for p in list(self._better_vol_at_ask_price.keys()):
                        if p > price:
                            self._better_vol_at_ask_price[p] = max(0.0, self._better_vol_at_ask_price[p] + delta)
                    self._book_asks_size[price] = new_size
                    touched_ask_prices.update([p for p in self._better_vol_at_ask_price.keys() if p >= price])
                    touched_ask_prices.add(price)

        # 2) Recompute per-order volume_ahead only for touched prices
        # Bids
        if touched_bid_prices:
            bids_by_price: Dict[float, List[tuple[int, Dict[str, Any]]]] = {}
            for oid, order in self.active_bids.items():
                if order['price'] in touched_bid_prices:
                    bids_by_price.setdefault(order['price'], []).append((oid, order))
            for price, orders in bids_by_price.items():
                orders.sort(key=lambda x: x[0])
                better_vol = self._better_vol_at_bid_price.get(price)
                if better_vol is None:
                    # initialize from book if missing
                    better_vol = sum(s for p, s in self.order_book.bids.items() if -p > price)
                    self._better_vol_at_bid_price[price] = better_vol
                book_at_price = self._book_bids_size.get(price, 0.0)
                our_total_at_price = self._our_bids_by_price.get(price, 0.0)
                baseline_equal_other = max(0.0, book_at_price - our_total_at_price)
                cumulative_our_prior = 0.0
                for oid, order in orders:
                    equal_other_vol = min(baseline_equal_other, order.get('baseline_equal_other_vol', baseline_equal_other))
                    order['volume_ahead'] = max(0.0, better_vol) + equal_other_vol + cumulative_our_prior
                    cumulative_our_prior += max(0.0, order['size'])

        # Asks
        if touched_ask_prices:
            asks_by_price: Dict[float, List[tuple[int, Dict[str, Any]]]] = {}
            for oid, order in self.active_asks.items():
                if order['price'] in touched_ask_prices:
                    asks_by_price.setdefault(order['price'], []).append((oid, order))
            for price, orders in asks_by_price.items():
                orders.sort(key=lambda x: x[0])
                better_vol = self._better_vol_at_ask_price.get(price)
                if better_vol is None:
                    better_vol = sum(s for p, s in self.order_book.asks.items() if p < price)
                    self._better_vol_at_ask_price[price] = better_vol
                book_at_price = self._book_asks_size.get(price, 0.0)
                our_total_at_price = self._our_asks_by_price.get(price, 0.0)
                baseline_equal_other = max(0.0, book_at_price - our_total_at_price)
                cumulative_our_prior = 0.0
                for oid, order in orders:
                    equal_other_vol = min(baseline_equal_other, order.get('baseline_equal_other_vol', baseline_equal_other))
                    order['volume_ahead'] = max(0.0, better_vol) + equal_other_vol + cumulative_our_prior
                    cumulative_our_prior += max(0.0, order['size'])

    def _rebuild_queue_caches(self):
        """Rebuilds book and better-volume caches from a full snapshot."""
        if not self.order_book:
            return
        # Rebuild book maps from order_book structures
        self._book_bids_size = { -p: s for p, s in self.order_book.bids.items() }
        self._book_asks_size = { p: s for p, s in self.order_book.asks.items() }
        # Rebuild our per-price aggregates
        self._our_bids_by_price.clear()
        for o in self.active_bids.values():
            self._our_bids_by_price[o['price']] = self._our_bids_by_price.get(o['price'], 0.0) + o['size']
        self._our_asks_by_price.clear()
        for o in self.active_asks.values():
            self._our_asks_by_price[o['price']] = self._our_asks_by_price.get(o['price'], 0.0) + o['size']
        # Recompute better volume cache for our active prices only
        self._better_vol_at_bid_price = {
            price: sum(s for p, s in self.order_book.bids.items() if -p > price)
            for price in {o['price'] for o in self.active_bids.values()}
        }
        self._better_vol_at_ask_price = {
            price: sum(s for p, s in self.order_book.asks.items() if p < price)
            for price in {o['price'] for o in self.active_asks.values()}
        }


    async def _decrement_time(self):
        """Dynamically calculates the time horizon based on the market's actual close time."""
        if not self.market_start_time or not self.market_close_time or self.session_duration_seconds <= 0:
            logger.error("Cannot start time decrement task: market times not initialized.")
            return

        logger.info("Time horizon tracking started.")
        while self.time_horizon > 0 and self.is_running:
            now = datetime.now(timezone.utc)
            time_remaining = (self.market_close_time - now).total_seconds()
            self.time_horizon = max(0, time_remaining / self.session_duration_seconds)
            await asyncio.sleep(1) # Update every second
        
        logger.info("Time horizon reached zero. Market has likely closed.")

    async def run(self):
        """The main run loop that connects to the WebSocket and processes messages."""
        logger.info("Starting bot for market: %s", self.market_id)

        current_retry_delay = self.initial_retry_delay

        async with aiohttp.ClientSession() as session:
            # --- Fetch market info on startup ---
            market_details = await get_market_details(session, self.market_id)

            if not market_details:
                logger.critical("Failed to get market details. Shutting down.")
                return

            # Store all the fetched details
            self.yes_token_id = market_details['yes_token_id']
            self.no_token_id = market_details['no_token_id']
            self.market_close_time = market_details['close_time']
            self.min_order_size = market_details['min_order_size']
            tick_size = market_details['min_tick_size']

            self.order_book = OrderBook(self.market_id, tick_size)
            self.trade_history = TradeHistory(self.market_id)
            self.dispatcher = EventDispatcher(
                primary_order_book=self.order_book,
                primary_asset_id=self.yes_token_id,
                trade_history=self.trade_history,
                update_callback=self._on_update
            )
            
            self.market_close_time = market_details['close_time']
            self.market_start_time = datetime.now(timezone.utc)
            self.session_duration_seconds = \
                (self.market_close_time - self.market_start_time).total_seconds()

            if self.session_duration_seconds <= 0:
                logger.critical("Market has already closed. Shutting down.")
                return
            # --- End startup fetching ---

            # Start the background task to decrement the time horizon
            asyncio.create_task(self._decrement_time())

            while self.is_running:
                try:
                    logger.info("Attempting to connect to WebSocket...")
                    async with websockets.connect(WS_URL, ping_interval=20) as ws:
                        logger.info("WebSocket connected for market %s.", self.market_id)
                        current_retry_delay = self.initial_retry_delay
                        sub_msg = {"assets_ids": [self.yes_token_id, self.no_token_id], "type": "market"}
                        await ws.send(json.dumps(sub_msg))
                        logger.info("Subscribed to market updates of both YES and NO tokens. Listening...")

                        async for message in ws:
                            if not self.is_running:
                                break
                            self.dispatcher.dispatch(message)

                except Exception as e:
                    logger.error("An unexpected error occurred in the run loop: %s", e, exc_info=True)
                    logger.info("Retrying in %d s...", current_retry_delay)
                    await asyncio.sleep(current_retry_delay)
                    current_retry_delay = min(self.max_retry_delay, current_retry_delay * self.retry_multiplier)

    def stop(self):
        """Signals the bot's main run loop to terminate."""
        if self.is_running:
            logger.info("Stop signal received. Shutting down bot for market %s...", self.market_id)
            self.is_running = False

    async def shutdown(self):
        """
        Shuts down the bot, liquidating any open positions with aggressive market orders.
        """
        self.stop() # Ensure the main loop is signaled to stop
        print() # Move to a new line to preserve the last status update
        logger.info("--- Starting Aggressive Shutdown ---")

        logger.info("Canceling all active orders...")
        self._cancel_all_orders()
        await asyncio.sleep(0.1) # Brief pause to allow cancellations to process

        if abs(self.inventory_position) > 1e-9:
            logger.warning("Bot has remaining inventory of %+.2f. Liquidating position...", self.inventory_position)

            # Snapshot book at submit
            best_bid = self.order_book.best_bid
            best_ask = self.order_book.best_ask
            mid_at_submit = self.order_book.mid_price

            if self.inventory_position > 0:
                # SELL aggressively into bids
                target = self.inventory_position
                filled, notional, levels = self.order_book.consume_from_bids(target)
                if filled > 0:
                    vwap = notional / filled
                    slippage_best = (vwap - best_bid) if best_bid is not None else None
                    slippage_mid = (vwap - mid_at_submit) if mid_at_submit is not None else None

                    # Update state as if we sold 'filled' at vwap
                    realized = 0.0
                    if self.inventory_position > 0:
                        close_sz = min(self.inventory_position, filled)
                        realized += (vwap - self.average_entry_price) * close_sz
                        self.inventory_position -= close_sz
                        self.cash += close_sz * vwap
                        if abs(self.inventory_position) < 1e-9:
                            self.average_entry_price = 0.0
                    # If any residual (should not happen here), ignore for now
                    self.realized_pnl += realized
                    self._update_pnl()

                    # Log aggressive trade artifact and append synthetic fill
                    ts_iso = (
                        datetime.fromtimestamp(self._last_sim_time_epoch, tz=timezone.utc).isoformat()
                        if isinstance(self._last_sim_time_epoch, (int, float)) else
                        datetime.now(timezone.utc).isoformat()
                    )
                    self.simulated_taker_trades.append({
                        'timestamp': ts_iso,
                        'side': 'SELL',
                        'requested_size': float(target),
                        'filled_size': float(filled),
                        'vwap': float(vwap),
                        'slippage_vs_best': float(slippage_best) if slippage_best is not None else None,
                        'slippage_vs_mid': float(slippage_mid) if slippage_mid is not None else None,
                        'levels_touched': int(len(levels))
                    })
                    self.simulated_fills.append({
                        'timestamp': ts_iso,
                        'side': 'SELL',
                        'price': float(vwap),
                        'size': float(filled),
                        'order_id': None,
                        'equity': self.total_value,
                        'inventory': self.inventory_position,
                        'cash': self.cash,
                        'exec_type': 'AGGRESSIVE_SWEEP'
                    })

                    logger.info(
                        "Aggressive SELL: filled %.2f (VWAP %.4f) | slip_best=%.5f slip_mid=%.5f | levels=%d",
                        filled, vwap,
                        slippage_best if slippage_best is not None else float('nan'),
                        slippage_mid if slippage_mid is not None else float('nan'),
                        len(levels)
                    )
                else:
                    logger.error("Cannot liquidate long position: No buyers in the order book.")

            else:
                # BUY aggressively from asks to cover short
                target = abs(self.inventory_position)
                filled, notional, levels = self.order_book.consume_from_asks(target)
                if filled > 0:
                    vwap = notional / filled
                    slippage_best = (best_ask - vwap) if best_ask is not None else None
                    slippage_mid = (mid_at_submit - vwap) if mid_at_submit is not None else None

                    realized = 0.0
                    if self.inventory_position < 0:
                        close_sz = min(abs(self.inventory_position), filled)
                        realized += (self.average_entry_price - vwap) * close_sz
                        self.inventory_position += close_sz
                        self.cash -= close_sz * vwap
                        if abs(self.inventory_position) < 1e-9:
                            self.average_entry_price = 0.0
                    self.realized_pnl += realized
                    self._update_pnl()

                    ts_iso = (
                        datetime.fromtimestamp(self._last_sim_time_epoch, tz=timezone.utc).isoformat()
                        if isinstance(self._last_sim_time_epoch, (int, float)) else
                        datetime.now(timezone.utc).isoformat()
                    )
                    self.simulated_taker_trades.append({
                        'timestamp': ts_iso,
                        'side': 'BUY',
                        'requested_size': float(target),
                        'filled_size': float(filled),
                        'vwap': float(vwap),
                        'slippage_vs_best': float(slippage_best) if slippage_best is not None else None,
                        'slippage_vs_mid': float(slippage_mid) if slippage_mid is not None else None,
                        'levels_touched': int(len(levels))
                    })
                    self.simulated_fills.append({
                        'timestamp': ts_iso,
                        'side': 'BUY',
                        'price': float(vwap),
                        'size': float(filled),
                        'order_id': None,
                        'equity': self.total_value,
                        'inventory': self.inventory_position,
                        'cash': self.cash,
                        'exec_type': 'AGGRESSIVE_SWEEP'
                    })

                    logger.info(
                        "Aggressive BUY: filled %.2f (VWAP %.4f) | slip_best=%.5f slip_mid=%.5f | levels=%d",
                        filled, vwap,
                        slippage_best if slippage_best is not None else float('nan'),
                        slippage_mid if slippage_mid is not None else float('nan'),
                        len(levels)
                    )
                else:
                    logger.error("Cannot liquidate short position: No sellers in the order book.")
        else:
            logger.info("Inventory is flat. No liquidation needed.")

        logger.info("--- Shutdown Complete ---")
