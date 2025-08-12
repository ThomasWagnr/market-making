import collections
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

class TradeHistory:
    """
    Tracks recent trades and maintains a complete log for a single market.
    """
    def __init__(self, market_id: str, max_recent_trades: int = 5000):
        self.market_id = market_id
        self.recent_trades = collections.deque(maxlen=max_recent_trades)
        self.last_trade = None

        self.trade_log: List[Dict[str, Any]] = []

    def add_trade(self, data: dict):
        """
        Adds a new trade to the recent trades deque and the permanent log.
        """
        # Prefer recorded sim_time if present for backtests
        ts = data.get('sim_time') if isinstance(data, dict) else None
        if isinstance(ts, (int, float)):
            ts_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            ts_dt = datetime.now(timezone.utc)

        trade_record = {
            'price': float(data['price']),
            'size': float(data['size']),
            'side': data['side'].upper(),
            'timestamp': ts_dt
        }

        self.recent_trades.append(trade_record)
        self.trade_log.append(trade_record)
        
        self.last_trade = trade_record

    def get_recent_trades(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Returns the most recent trades.
        """
        return list(self.recent_trades)[-count:]

    def get_volume_in_last_seconds(self, seconds: int) -> float:
        """Calculates the total trade volume in the last N seconds."""
        now = datetime.now(timezone.utc)
        time_threshold = now - timedelta(seconds=seconds)
        
        recent_volume = sum(trade['size'] for trade in self.recent_trades 
                            if trade['timestamp'] > time_threshold)
        return recent_volume

    def __str__(self):
        if not self.last_trade:
            return "Last Trade: N/A"
        
        trade_record_str = {**self.last_trade, 'timestamp': self.last_trade['timestamp'].isoformat()}
        
        side = trade_record_str['side']
        price = f"{trade_record_str['price']:.3f}"
        size = trade_record_str['size']
        
        return f"Last Trade: {side} {size} @ {price}"