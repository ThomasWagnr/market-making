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
from utils import get_yes_token_for_market, get_market_close_time
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
                 base_order_size: float = 250.0,
                 simulated_orders: List[Dict[str, Any]] = None,
                 simulated_fills: List[Dict[str, Any]] = None):
                 
        # --- Configuration ---
        self.market_id = market_id
        self.strategy = strategy
        self.base_order_size = base_order_size
        self.execution_client = execution_client
        self.is_running = True

        # --- Reporting & Simulation ---
        self.simulated_orders = simulated_orders if simulated_orders is not None else []
        self.simulated_fills = simulated_fills if simulated_fills is not None else []

        # --- Performance Tracking ---
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.cash = 10000.0
        self.average_entry_price = 0.0
        
        # --- Live State ---
        self.inventory_position = 0.0
        self.active_bid = {'id': None, 'price': 0.0, 'size': 0.0}
        self.active_ask = {'id': None, 'price': 0.0, 'size': 0.0}

        # --- Market State ---
        self.time_horizon = 1.0
        self.market_start_time: Optional[datetime] = None
        self.market_close_time: Optional[datetime] = None
        self.session_duration_seconds: float = 0.0

        # --- Component Initialization ---
        self.order_book = OrderBook(self.market_id)
        self.trade_history = TradeHistory(self.market_id)
        self.dispatcher = EventDispatcher(
            order_book=self.order_book,
            trade_history=self.trade_history,
            update_callback=self._on_update
        )

    def _on_update(self, event_type: str, data: dict):
        """Core trigger, called by the dispatcher after any data update."""

        if not self.is_running:
            return

        if event_type == "last_trade_price":
            self._check_fills(data)

        self._update_pnl()

        # 2. Delegate quote calculation to the strategy module
        new_bid, new_ask = self.strategy.calculate_quotes(
            order_book=self.order_book,
            inventory_position=self.inventory_position,
            time_horizon=self.time_horizon
        )

        # 3. Print current state to the console
        state_str = (f"{self.order_book} | Inv: {self.inventory_position:+.1f} | "
                     f"P&L: ${self.realized_pnl:+.2f} | Equity: ${self.total_value:,.2f}")
        print(f"\r{state_str}", end="", flush=True)

        # 4. Update orders on the exchange based on the strategy's decision
        if new_bid is not None and new_ask is not None:
            self._update_orders(new_bid, new_ask)
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
        """Calculates dynamic order sizes based on base size, inventory, and liquidity."""
        mid_price = self.order_book.mid_price
        if not mid_price: # Safety check
            return 1.0, 1.0

        inventory_factor_buy = 1.0 - (self.inventory_position / POSITION_LIMIT)
        inventory_factor_sell = 1.0 + (self.inventory_position / POSITION_LIMIT)
        
        bid_size = self.base_order_size * inventory_factor_buy
        ask_size = self.base_order_size * inventory_factor_sell
        
        # 3. Liquidity adjustment
        # Don't place orders larger than a fraction of the visible volume at the best price
        liquidity_fraction = 0.72
        if self.order_book.best_ask and self.order_book.best_ask in self.order_book.asks:
            ask_size = min(ask_size, self.order_book.asks[self.order_book.best_ask] * liquidity_fraction)
        if self.order_book.best_bid and -self.order_book.best_bid in self.order_book.bids:
            bid_size = min(bid_size, self.order_book.bids[-self.order_book.best_bid] * liquidity_fraction)

        return max(1.0, bid_size), max(1.0, ask_size)

    def _update_orders(self, new_bid: float, new_ask: float):
        """Manages order cancellations and placements to match the strategy's target."""

        bid_size, ask_size = self._calculate_order_sizes()

        # Cancel ask if price needs to change
        if self.active_ask['id'] is not None and self.active_ask['price'] != new_ask:
            self._cancel_order(self.active_ask['id'])
            self.active_ask = {'id': None, 'price': 0.0, 'size': 0.0}

        # Cancel bid if price needs to change
        if self.active_bid['id'] is not None and self.active_bid['price'] != new_bid:
            self._cancel_order(self.active_bid['id'])
            self.active_bid = {'id': None, 'price': 0.0, 'size': 0.0}

        # Place new bid if we don't have one
        if self.active_bid['id'] is None and self.inventory_position < POSITION_LIMIT:
            order_id = self._place_order("BUY", new_bid, bid_size)
            if order_id:
                self.active_bid = {'id': order_id, 'price': new_bid, 'size': bid_size}

        # Place new ask if we don't have one
        if self.active_ask['id'] is None and self.inventory_position > -POSITION_LIMIT:
            order_id = self._place_order("SELL", new_ask, ask_size)
            if order_id:
                self.active_ask = {'id': order_id, 'price': new_ask, 'size': ask_size}

    def _check_fills(self, trade: dict):
        """Checks for fills and updates inventory, cash, and P&L."""
        trade_price = float(trade['price'])
        trade_size = float(trade['size'])

        side = None
        fill_amount = 0.0

        # Check for a bid fill
        if self.active_bid['id'] and trade_price == self.active_bid['price']:
            side = 'BUY'
            fill_amount = min(trade_size, self.active_bid['size'])
            is_full_fill = (self.active_bid['size'] - fill_amount) <= 1e-9

            self.active_bid['size'] -= fill_amount
            if is_full_fill:
                self.active_bid = {'id': None, 'price': 0.0, 'size': 0.0}
        
        elif self.active_ask['id'] and trade_price == self.active_ask['price']:
            side = 'SELL'
            fill_amount = min(trade_size, self.active_ask['size'])
            is_full_fill = (self.active_ask['size'] - fill_amount) <= 1e-9
            
            self.active_ask['size'] -= fill_amount
            if is_full_fill:
                self.active_ask = {'id': None, 'price': 0.0, 'size': 0.0}
        
        if side:
            self.simulated_fills.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'side': side, 'price': trade_price, 'size': fill_amount
            })

            # Update cash and inventory position
            if side == 'BUY':
                # --- Update Average Entry Price ---
                # If reducing a short position, realize P&L
                if self.inventory_position < 0:
                    self.realized_pnl += (self.average_entry_price - trade_price) * fill_amount
                # If opening or adding to a long position, update avg price
                else: 
                    new_total_cost = (self.average_entry_price * self.inventory_position) + (trade_price * fill_amount)
                    self.average_entry_price = new_total_cost / (self.inventory_position + fill_amount)
                
                self.inventory_position += fill_amount
                self.cash -= fill_amount * trade_price

            else: # SELL
                # --- Update Average Entry Price ---
                # If reducing a long position, realize P&L
                if self.inventory_position > 0:
                    self.realized_pnl += (trade_price - self.average_entry_price) * fill_amount
                # If opening or adding to a short position, update avg price
                else:
                    new_total_cost = (self.average_entry_price * abs(self.inventory_position)) + (trade_price * fill_amount)
                    self.average_entry_price = new_total_cost / (abs(self.inventory_position) + fill_amount)
                
                self.inventory_position -= fill_amount
                self.cash += fill_amount * trade_price

            # Reset average entry price if position is now flat
            if abs(self.inventory_position) < 1e-9:
                self.average_entry_price = 0.0

            logger.info(f"FILL: {side} {fill_amount} @ {trade_price:.3f}. New Inventory: {self.inventory_position:+.1f}")

    def _place_order(self, side: str, price: float, size: float) -> int | None:
        """Delegates order placement to the execution client."""
        return self.execution_client.place_order(side, price, size)

    def _cancel_order(self, order_id: int):
        """Delegates order cancellation to the execution client."""
        return self.execution_client.cancel_order(order_id)

    def _cancel_all_orders(self):
        """Cancels all currently active orders."""
        if self.active_bid.get('id'):
            self._cancel_order(self.active_bid['id'])
            self.active_bid = {'id': None, 'price': 0.0, 'size': 0.0}
        if self.active_ask.get('id'):
            self._cancel_order(self.active_ask['id'])
            self.active_ask = {'id': None, 'price': 0.0, 'size': 0.0}


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
            yes_token_id, close_time = await asyncio.gather(
                get_yes_token_for_market(session, self.market_id),
                get_market_close_time(session, self.market_id)
            )

            if not yes_token_id or not close_time:
                logger.critical("Failed to get market token or close time. Shutting down.")
                return

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
                        sub_msg = {"assets_ids": [yes_token_id], "type": "market"}
                        await ws.send(json.dumps(sub_msg))
                        logger.info("Subscribed to market updates. Listening...")

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
