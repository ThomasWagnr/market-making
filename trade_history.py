import collections

class TradeHistory:
    """
    Tracks recent trades for a single market.
    """
    def __init__(self, market_id: str, max_trades: int = 50000):
        self.market_id = market_id
        self.trades = collections.deque(maxlen=max_trades)
        self.last_trade = None

    def add_trade(self, data: dict):
        """
        Adds a new trade from a 'last_trade_price' event.
        """
        trade = {
            'price': float(data['price']),
            'size': float(data['size']),
            'side': data['side'].upper(),
            'timestamp': int(data['timestamp'])
        }
        self.trades.append(trade)
        self.last_trade = trade

    def __str__(self):
        if not self.last_trade:
            return "Trades: N/A"
        
        side = self.last_trade['side']
        price = self.last_trade['price']
        size = self.last_trade['size']
        
        return f"Last Trade: {side} {size} @ {price:.3f}"