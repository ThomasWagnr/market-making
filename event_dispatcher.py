import orjson
import logging
from typing import Callable, Dict, Optional
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
        self._last_trade_price_by_asset: Dict[str, float] = {}
        self._logger = logging.getLogger(__name__)

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
                # Normalize trade price to YES-equivalent for consistent analytics
                try:
                    raw_price = float(event.get("price"))
                except Exception:
                    raw_price = None

                if raw_price is not None:
                    if asset_id == self.primary_asset_id:
                        yes_equiv_price = raw_price
                    else:
                        yes_equiv_price = 1.0 - raw_price

                    # Attach normalized price for downstream consumers
                    event["equivalent_yes_price"] = yes_equiv_price

                    # Parity check with the last seen opposite token trade
                    self._parity_check(asset_id, raw_price)

                    # Record in trade history using YES-equivalent price
                    event_for_history = dict(event)
                    event_for_history["price"] = yes_equiv_price
                    self.trade_history.add_trade(event_for_history)
                else:
                    # Fallback: record as-is if price not parseable
                    self.trade_history.add_trade(event)
            
            # Notify the bot of any update
            self.update_callback(event_type, event)

    def _parity_check(self, asset_id: Optional[str], price: Optional[float]):
        """Logs a warning if YES/NO prices deviate beyond tolerance."""
        if asset_id is None or price is None:
            return
        self._last_trade_price_by_asset[asset_id] = price

        # If we have both sides, check parity: YES ~= 1 - NO
        try:
            tol = max(1e-6, float(self.primary_order_book.tick_size))
        except Exception:
            tol = 1e-6

        for other_asset, other_price in list(self._last_trade_price_by_asset.items()):
            if other_asset == asset_id:
                continue
            if asset_id == self.primary_asset_id:
                yes_price, no_price = float(price), float(other_price)
            else:
                yes_price, no_price = float(other_price), float(price)
            diff = abs(yes_price - (1.0 - no_price))
            if diff > tol:
                self._logger.warning(
                    "Trade parity mismatch for market %s: YES %.5f vs 1-NO %.5f (diff %.5f > tol %.5f)",
                    self.market_id, yes_price, 1.0 - no_price, diff, tol
                )