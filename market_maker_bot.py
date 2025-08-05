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
from utils import get_market_tokens, get_market_close_time
from strategies.base_strategy import BaseStrategy
from execution_client import ExecutionClient

logger = logging.getLogger(__name__)

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
POSITION_LIMIT = 1500.0

class MarketMakerBot:
    """A generic bot runner that operates using a provided strategy module."""

    def __init__(self,
                 market_id: str,
                 strategy: BaseStrategy,
                 execution_client: ExecutionClient,
                 base_order_value: float = 100.0,
                 simulated_fills: List[Dict[str, Any]] = None):

        # --- Configuration ---
        self.market_id = market_id
        self.yes_token_id: Optional[str] = None
        self.no_token_id: Optional[str] = None
        self.strategy = strategy
        self.base_order_value = base_order_value
        self.execution_client = execution_client
        self.is_running = True

        # --- Reporting & Simulation ---
        self.simulated_fills = simulated_fills if simulated_fills is not None else []

        # --- Performance Tracking ---
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.cash = 10000.0
        self.total_value = self.cash
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

        # --- Component Initialization ---
        self.order_book = OrderBook(self.market_id)
        self.trade_history = TradeHistory(self.market_id)
        self.dispatcher : Optional[EventDispatcher] = None

    def _on_update(self, event_type: str, data: dict):
        """Core trigger, called by the dispatcher after any data update."""

        if not self.is_running:
            return

        if event_type == "last_trade_price":
            self._check_fills(data)

        self._update_pnl()

        total_bid_size, total_ask_size = self._calculate_order_sizes()

        # 2. Delegate quote calculation to the strategy module
        new_bids, new_asks = self.strategy.calculate_quotes(
            order_book=self.order_book,
            inventory_position=self.inventory_position,
            time_horizon=self.time_horizon,
            total_bid_size=total_bid_size,
            total_ask_size=total_ask_size
        )

        # 3. Print current state to the console
        state_str = (f"{self.order_book} | Inv: {self.inventory_position:+.1f} | "
                     f"P&L: ${self.realized_pnl:+.2f} | Equity: ${self.total_value:,.2f}")
        print(f"\r{state_str}", end="", flush=True)

        # 4. Update orders on the exchange based on the strategy's decision
        if new_bids or new_asks:
            self._update_orders(new_bids, new_asks)
        else:
            self._cancel_all_orders()

    def _update_pnl(self):
        """Calculates unrealized P&L and total equity."""
        if not self.order_book.best_bid or not self.order_book.best_ask:
            return

        mid_price = (self.order_book.best_bid + self.order_book.best_ask) / 2
        
        # Calculate the market value of our current inventory
        market_value = self.inventory_position * mid_price
        
        # Calculate unrealized P&L if we hold a position
        if self.inventory_position != 0:
            cost_basis = self.inventory_position * self.average_entry_price
            self.unrealized_pnl = market_value - cost_basis
        else:
            self.unrealized_pnl = 0.0
        
        # Total equity is our cash plus the current market value of our assets
        self.total_value = self.cash + market_value

    def _calculate_order_sizes(self) -> tuple[float, float]:
        """
        Calculates dynamic, symmetric order sizes that preserve the inventory
        skew ratio even when constrained by liquidity.
        """
        mid_price = self.order_book.mid_price
        if not mid_price or mid_price <= 1e-9: # Safety check
            return 1.0, 1.0

        base_size = self.base_order_value / mid_price

        inventory_factor_buy = 1.0 - (self.inventory_position / POSITION_LIMIT)
        inventory_factor_sell = 1.0 + (self.inventory_position / POSITION_LIMIT)

        initial_bid_size = base_size * inventory_factor_buy
        initial_ask_size = base_size * inventory_factor_sell
        
        # 3. Liquidity adjustment
        # Don't place orders larger than a fraction of the visible volume at the best price
        liquidity_fraction = 0.72
        max_bid_size_by_liquidity = float('inf')
        max_ask_size_by_liquidity = float('inf')

        if self.order_book.best_ask and self.order_book.best_ask in self.order_book.asks:
            max_ask_size_by_liquidity = self.order_book.asks[self.order_book.best_ask] * liquidity_fraction
        if self.order_book.best_bid and -self.order_book.best_bid in self.order_book.bids:
            max_bid_size_by_liquidity = self.order_book.bids[-self.order_book.best_bid] * liquidity_fraction

        bid_scaling_factor = max_bid_size_by_liquidity / (initial_bid_size + 1e-9)
        ask_scaling_factor = max_ask_size_by_liquidity / (initial_ask_size + 1e-9)

        final_scaling_factor = min(1.0, bid_scaling_factor, ask_scaling_factor)

        # Calculate the actual sizes that preserve the inventory skew
        bid_size = initial_bid_size * final_scaling_factor
        ask_size = initial_ask_size * final_scaling_factor

        return max(1.0, bid_size), max(1.0, ask_size)

    def _update_orders(self, new_bids: List[Dict[str, Any]], new_asks: List[Dict[str, Any]]):
        """Manages order cancellations and placements to match the strategy's target."""

        target_bid_prices = {q['price'] for q in new_bids}

        # Cancel any active bids whose price is no longer in the target ladder
        for order_id, order in list(self.active_bids.items()):
            if order['price'] not in target_bid_prices:
                self._cancel_order(order_id)
                del self.active_bids[order_id]

        # Place new bids for price levels where we don't have an order yet
        active_bid_prices = {o['price'] for o in self.active_bids.values()}
        for quote in new_bids:
            if quote['price'] not in active_bid_prices and self.inventory_position < POSITION_LIMIT:
                order_id = self._place_order("BUY", quote['price'], quote['size'])
                if order_id:
                    self.active_bids[order_id] = quote

        # --- Symmetrical Logic for Asks ---
        target_ask_prices = {q['price'] for q in new_asks}
        for order_id, order in list(self.active_asks.items()):
            if order['price'] not in target_ask_prices:
                self._cancel_order(order_id)
                del self.active_asks[order_id]

        active_ask_prices = {o['price'] for o in self.active_asks.values()}
        for quote in new_asks:
            if quote['price'] not in active_ask_prices and self.inventory_position > -POSITION_LIMIT:
                order_id = self._place_order("SELL", quote['price'], quote['size'])
                if order_id:
                    self.active_asks[order_id] = quote

    def _check_fills(self, trade: dict):
        """
        Checks for fills from either the YES or NO token feed
        and updates inventory, cash, and P&L.
        """
        asset_id = trade.get('asset_id')
        if not asset_id: return
        trade_price = float(trade['price'])
        trade_size = float(trade['size'])

        if asset_id == self.yes_token_id:
            equivalent_yes_price = trade_price
        elif asset_id == self.no_token_id:
            equivalent_yes_price = 1.0 - trade_price
        else:
            return 

        # Check for bid fills
        for order_id, order in list(self.active_bids.items()):
            if abs(equivalent_yes_price - order['price']) <= 1e-9:
                self._process_fill('BUY', order['price'], min(trade_size, order['size']), order_id)
        
        # Check for ask fills
        for order_id, order in list(self.active_asks.items()):
            if abs(equivalent_yes_price - order['price']) <= 1e-9:
                self._process_fill('SELL', order['price'], min(trade_size, order['size']), order_id)

    def _process_fill(self, side: str, price: float, size: float, order_id: int):
        """Processes a single fill, updating inventory, P&L, and active orders."""
        self.simulated_fills.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'side': side, 'price': price, 'size': size
        })

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
            if active_order_book[order_id]['size'] <= 1e-9:
                del active_order_book[order_id]
        
        if abs(self.inventory_position) < 1e-9:
            self.average_entry_price = 0.0

        logger.info(f"FILL: {side} {size:.2f} @ {price:.3f}. New Inventory: {self.inventory_position:+.1f}")

    def _place_order(self, side: str, price: float, size: float) -> int | None:
        """Delegates order placement to the execution client."""
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

        async with aiohttp.ClientSession() as session:
            # --- Fetch market info on startup ---
            tokens, close_time = await asyncio.gather(
                get_market_tokens(session, self.market_id),
                get_market_close_time(session, self.market_id)
            )

            if not tokens or not close_time:
                logger.critical("Failed to get market token or close time. Shutting down.")
                return

            self.yes_token_id, self.no_token_id = tokens
            self.dispatcher = EventDispatcher(
                primary_order_book=self.order_book,
                primary_asset_id=self.yes_token_id,
                trade_history=self.trade_history,
                update_callback=self._on_update
            )
            
            self.market_close_time = close_time
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
                        sub_msg = {"assets_ids": [self.yes_token_id, self.no_token_id], "type": "market"}
                        await ws.send(json.dumps(sub_msg))
                        logger.info("Subscribed to market updates of both YES and NO tokens. Listening...")

                        async for message in ws:
                            if not self.is_running:
                                break
                            self.dispatcher.dispatch(message)

                except Exception as e:
                    logger.error("An unexpected error occurred in the run loop: %s", e, exc_info=True)
                    logger.info("Retrying in 5s...")
                    await asyncio.sleep(5)

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

            # If we are long, we need to sell immediately
            if self.inventory_position > 0:
                if self.order_book.best_bid:
                    price = self.order_book.best_bid
                    size = self.inventory_position
                    logger.info("Placing aggressive SELL order for %.2f @ %.3f", size, price)
                    self._place_order("SELL", price, size)
                else:
                    logger.error("Cannot liquidate long position: No buyers in the order book.")
            
            # If we are short, we need to buy back immediately
            else: # self.inventory_position < 0
                if self.order_book.best_ask:
                    price = self.order_book.best_ask
                    size = abs(self.inventory_position)
                    logger.info("Placing aggressive BUY order for %.2f @ %.3f", size, price)
                    self._place_order("BUY", price, size)
                else:
                    logger.error("Cannot liquidate short position: No sellers in the order book.")
        else:
            logger.info("Inventory is flat. No liquidation needed.")

        self.save_report()
        logger.info("--- Shutdown Complete ---")

    def save_report(self):
        """Saves the simulated orders and fills to CSV files."""
        logger.info("Saving simulation report to CSV files...")
        
        if self.simulated_orders:
            with open('simulated_orders.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.simulated_orders[0].keys())
                writer.writeheader()
                writer.writerows(self.simulated_orders)
            logger.info("Saved simulated_orders.csv")

        if self.simulated_fills:
            with open('simulated_fills.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.simulated_fills[0].keys())
                writer.writeheader()
                writer.writerows(self.simulated_fills)
            logger.info("Saved simulated_fills.csv")
