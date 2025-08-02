# market_maker_bot.py

import asyncio
import json
import logging
import aiohttp
import websockets

from order_book import OrderBook
from trade_history import TradeHistory
from event_dispatcher import EventDispatcher
from utils import get_yes_token_for_market

logger = logging.getLogger(__name__)

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

class MarketMakerBot:
    """
    Orchestrates all components and logic for a market-making strategy.
    """

    def __init__(self, market_id: str, order_size: float, desired_spread: float):
        self.market_id = market_id
        self.order_size = order_size
        self.desired_spread = desired_spread
        self.is_running = True

        self.order_book = OrderBook(self.market_id)
        self.trade_history = TradeHistory(self.market_id)
        self.dispatcher = EventDispatcher(
            order_book=self.order_book,
            trade_history=self.trade_history,
            update_callback=self._on_update
        )

    def _on_update(self):
        """The core strategy trigger, called after any data update."""
        state_str = f"{self.order_book} | {self.trade_history}"
        print(f"\r{state_str}", end="", flush=True)

        # To-Do: Trading logic would go here ---

    async def run(self):
        """
        The main run loop that connects to the WebSocket and processes messages.
        """
        logger.info("Starting bot for market: %s", self.market_id)

        async with aiohttp.ClientSession() as session:
            yes_token_id = await get_yes_token_for_market(session, self.market_id)
            if not yes_token_id:
                logger.critical("Could not get token for market %s. Shutting down.", self.market_id)
                return

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

                except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError) as e:
                    logger.warning("Connection error for %s: %s. Retrying in 5s...", self.market_id, e)
                    await asyncio.sleep(5)
                except Exception as e:
                    logger.error("An unexpected error occurred in the run loop: %s", e, exc_info=True)
                    logger.info("Retrying in 5s...")
                    await asyncio.sleep(5)

    def stop(self):
        """
        Signals the bot to gracefully shut down.
        """
        logger.info("Stop signal received. Shutting down bot for market %s...", self.market_id)
        self.is_running = False