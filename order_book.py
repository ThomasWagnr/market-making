import orjson
from sortedcontainers import SortedDict

class OrderBook:
    """
    Manages the state of the order book for a single market.
    """
    def __init__(self, market_id: str):
        self.market_id = market_id
        self.bids = SortedDict()
        self.asks = SortedDict()

    def update_from_snapshot(self, data: dict):
        """
        Rebuilds the order book from a full snapshot ('book' event).
        """
        self.bids.clear()
        self.asks.clear()

        for level in data.get('bids', []):
            price, size = float(level['price']), float(level['size'])
            if size > 0: self.bids[-price] = size

        for level in data.get('asks', []):
            price, size = float(level['price']), float(level['size'])
            if size > 0: self.asks[price] = size

    def update_from_price_change(self, data: dict):
        """
        Updates the book from a 'price_change' event.
        """
        for change in data.get('changes', []):
            price, size = float(change['price']), float(change['size'])
            side = change['side'].upper()

            book = self.bids if side == 'BUY' else self.asks
            price_key = -price if side == 'BUY' else price
            
            if size > 0:
                book[price_key] = size
            elif price_key in book:
                del book[price_key]

    @property
    def best_bid(self) -> float | None:
        """
        Returns the best bid price (highest price).
        """
        if not self.bids:
            return None
        return -self.bids.peekitem(0)[0] 

    @property
    def best_ask(self) -> float | None:
        """
        Returns the best ask price (lowest price).
        """
        if not self.asks:
            return None
        return self.asks.peekitem(0)[0]

    @property
    def mid_price(self) -> float | None:
        """
        Returns the mid price (average of the best bid and ask).
        """
        bid, ask = self.best_bid, self.best_ask
        return (bid + ask) / 2.0 if bid is not None and ask is not None else None

    @property
    def spread(self) -> float | None:
        """
        Returns the difference between the best ask and best bid.
        """
        bid, ask = self.best_bid, self.best_ask
        return ask - bid if bid is not None and ask is not None else None

    def __str__(self):
        """
        Provides a clean, single-line string representation of the book's state.
        """
        bid_str = f"{self.best_bid:.3f}" if self.best_bid is not None else "N/A"
        ask_str = f"{self.best_ask:.3f}" if self.best_ask is not None else "N/A"
        mid_price = self.mid_price
        mid_price_str = f"{mid_price:.4f}" if mid_price is not None else "N/A"
        spread = self.spread
        spread_str = f"{spread:.4f}" if spread is not None else "N/A"
        
        return f"Book State | Best Bid: {bid_str} | Best Ask: {ask_str} | Mid Price: {mid_price_str} | Spread: {spread_str}"