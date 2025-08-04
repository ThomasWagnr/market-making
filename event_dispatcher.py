import orjson
from typing import Callable
from order_book import OrderBook
from trade_history import TradeHistory

class EventDispatcher:
    """Parses WebSocket messages, dispatches them, and triggers a callback."""

    def __init__(self, order_book: OrderBook, trade_history: TradeHistory, update_callback: Callable):
        self.order_book = order_book
        self.trade_history = trade_history
        self.market_id = order_book.market_id
        self.update_callback = update_callback

    def dispatch(self, raw_message: str):
        """Parses a raw message and routes each event."""
        events = orjson.loads(raw_message)
        if not isinstance(events, list):
            events = [events]

        for event in events:
            # Only process events for the dispatcher's market
            if event.get("market") != self.market_id:
                continue

            event_type = event.get("event_type")
            
            if event_type == "book":
                self.order_book.update_from_snapshot(event)
            elif event_type == "price_change":
                self.order_book.update_from_price_change(event)
            elif event_type == "last_trade_price":
                self.trade_history.add_trade(event)
                
            if self.update_callback:
                self.update_callback(event_type, event)