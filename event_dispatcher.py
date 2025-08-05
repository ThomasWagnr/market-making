import orjson
from typing import Callable
from order_book import OrderBook
from trade_history import TradeHistory

class EventDispatcher:
    """
    Parses messages and dispatches them. It updates one primary order book
    but processes trades from all subscribed feeds.
    """

    def __init__(self, primary_order_book: OrderBook, primary_asset_id: str, 
                 trade_history: TradeHistory, update_callback: Callable):
        self.primary_order_book = primary_order_book
        self.primary_asset_id = primary_asset_id
        self.trade_history = trade_history
        self.market_id = primary_order_book.market_id
        self.update_callback = update_callback

    def dispatch(self, raw_message: str):
        """Parses a raw message and routes each event."""
        events = orjson.loads(raw_message)
        if not isinstance(events, list):
            events = [events]

        for event in events:
            event_type = event.get("event_type")
            asset_id = event.get("asset_id")

            # --- MODIFIED: Simplified Routing Logic ---
            # Only update the book if the event is for our primary asset
            if event_type in ["book", "price_change"] and asset_id == self.primary_asset_id:
                if event_type == "book":
                    self.primary_order_book.update_from_snapshot(event)
                else: # price_change
                    self.primary_order_book.update_from_price_change(event)

            # Always process trades to capture all market activity
            if event_type == "last_trade_price":
                self.trade_history.add_trade(event)
            
            # Notify the bot of any update
            self.update_callback(event_type, event)