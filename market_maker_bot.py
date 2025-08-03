import asyncio
import json
import logging
import aiohttp
import websockets

from order_book import OrderBook
from trade_history import TradeHistory
from event_dispatcher import EventDispatcher
from utils import get_yes_token_for_market
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
POSITION_LIMIT = 500.0

class MarketMakerBot:
    """A generic bot runner that operates using a provided strategy module."""

    def __init__(self, market_id: str, strategy: BaseStrategy, lot_size: float = 10.0):
        # --- Configuration ---
        self.market_id = market_id
        self.strategy = strategy
        self.lot_size = lot_size
        self.is_running = True

        # --- Live State ---
        self.inventory_position = 0.0
        self.time_horizon = 1.0  # Represents the fraction of the trading session remaining
        self.active_bid = {'id': None, 'price': 0.0, 'size': 0.0}
        self.active_ask = {'id': None, 'price': 0.0, 'size': 0.0}

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
        # 1. Check for potential fills from the last trade
        if event_type == "last_trade_price":
            self._check_fills(data)

        # 2. Delegate quote calculation to the strategy module
        new_bid, new_ask = self.strategy.calculate_quotes(
            order_book=self.order_book,
            inventory_position=self.inventory_position,
            time_horizon=self.time_horizon
        )

        # 3. Print current state to the console
        state_str = f"{self.order_book} | {self.trade_history} | Inv: {self.inventory_position:+.1f}"
        print(f"\r{state_str}", end="", flush=True)

        # 4. Update orders on the exchange based on the strategy's decision
        if new_bid is not None and new_ask is not None:
            self._update_orders(new_bid, new_ask)
        else:
            self._cancel_all_orders()

    def _update_orders(self, new_bid: float, new_ask: float):
        """Manages order cancellations and placements to match the strategy's target."""
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
            order_id = self._place_order("BUY", new_bid, self.lot_size)
            if order_id:
                self.active_bid = {'id': order_id, 'price': new_bid, 'size': self.lot_size}

        # Place new ask if we don't have one
        if self.active_ask['id'] is None and self.inventory_position > -POSITION_LIMIT:
            order_id = self._place_order("SELL", new_ask, self.lot_size)
            if order_id:
                self.active_ask = {'id': order_id, 'price': new_ask, 'size': self.lot_size}

    def _check_fills(self, trade: dict):
        """Checks for fills and updates the inventory position."""
        trade_price = float(trade['price'])
        trade_size = float(trade['size'])

        # Check for a bid fill
        if self.active_bid['id'] and trade_price == self.active_bid['price']:
            # Determine the actual amount filled in this trade
            fill_amount = min(trade_size, self.active_bid['size'])
            is_full_fill = (self.active_bid['size'] - fill_amount) <= 1e-9

            # Log the specific outcome
            if is_full_fill:
                logger.info(f"Our BID was FULLY filled! Amount: {fill_amount} at {trade_price:.3f}")
            else:
                logger.info(f"Our BID was PARTIALLY filled! Amount: {fill_amount} at {trade_price:.3f}")

            # Update state
            self.inventory_position += fill_amount
            self.active_bid['size'] -= fill_amount

            # Reset the order if it's now fully filled
            if is_full_fill:
                self.active_bid = {'id': None, 'price': 0.0, 'size': 0.0}

        # Check for an ask fill
        elif self.active_ask['id'] and trade_price == self.active_ask['price']:
            # Determine the actual amount filled in this trade
            fill_amount = min(trade_size, self.active_ask['size'])
            is_full_fill = (self.active_ask['size'] - fill_amount) <= 1e-9

            # Log the specific outcome
            if is_full_fill:
                logger.info(f"Our ASK was FULLY filled! Amount: {fill_amount} at {trade_price:.3f}")
            else:
                logger.info(f"Our ASK was PARTIALLY filled! Amount: {fill_amount} at {trade_price:.3f}")

            # Update state
            self.inventory_position -= fill_amount
            self.active_ask['size'] -= fill_amount

            # Reset the order if it's now fully filled
            if is_full_fill:
                self.active_ask = {'id': None, 'price': 0.0, 'size': 0.0}

    def _place_order(self, side: str, price: float, size: float) -> int:
        """Placeholder for order placement logic. Returns a dummy ID."""
        order_id = abs(hash(f"{side}{price}{size}{asyncio.run(asyncio.sleep(0))}"))
        logger.info(f"PLACING {side} order for {size} @ {price:.3f} (ID: {order_id})")
        # To-Do: Implement order placement logic
        return order_id

    def _cancel_order(self, order_id: int):
        """Placeholder for canceling an order."""
        logger.info(f"CANCELING order {order_id}")
        # To-Do: Implement order cancellation logic

    def _cancel_all_orders(self):
        """Cancels all currently active orders."""
        if self.active_bid.get('id'):
            self._cancel_order(self.active_bid['id'])
        if self.active_ask.get('id'):
            self._cancel_order(self.active_ask['id'])

    async def _decrement_time(self):
        """Simulates the passage of time for the strategy's T parameter."""
        # To-Do: Link this to the market's closing time.
        session_duration_seconds = 300 # e.g., a 5-minute session
        while self.time_horizon > 0 and self.is_running:
            await asyncio.sleep(1)
            self.time_horizon = max(0, self.time_horizon - (1 / session_duration_seconds))

    async def run(self):
        """The main run loop that connects to the WebSocket and processes messages."""
        logger.info("Starting bot for market: %s", self.market_id)

        async with aiohttp.ClientSession() as session:
            yes_token_id = await get_yes_token_for_market(session, self.market_id)
            if not yes_token_id:
                logger.critical("Could not get token for market %s. Shutting down.", self.market_id)
                return

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
        """Signals the bot to gracefully shut down."""
        logger.info("Stop signal received. Shutting down bot for market %s...", self.market_id)
        self.is_running = False