import logging
import aiohttp
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

MARKET_INFO_URL = "https://clob.polymarket.com/markets/{market_id}"

async def get_market_details(session: aiohttp.ClientSession, market_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetches all key details for a market.
    """
    url = MARKET_INFO_URL.format(market_id=market_id)
    logger.info("Fetching market details for: %s", market_id)
    try:
        async with session.get(url, timeout=10) as resp:
            resp.raise_for_status()
            market_data = await resp.json()
            
            yes_token = next((t.get('token_id') for t in market_data.get('tokens', []) if t.get('outcome') == 'Yes'), None)
            no_token = next((t.get('token_id') for t in market_data.get('tokens', []) if t.get('outcome') == 'No'), None)
            close_time_str = market_data.get('end_date_iso')
            
            min_tick_size = market_data.get('minimum_tick_size')
            min_order_size = market_data.get('minimum_order_size')

            # Validate that we got everything
            if not all([yes_token, no_token, close_time_str, min_tick_size, min_order_size]):
                logger.error("API response for market %s was missing required data.", market_id)
                return None

            if close_time_str.endswith('Z'):
                close_time_str = close_time_str[:-1] + '+00:00'
            close_time = datetime.fromisoformat(close_time_str)

            return {
                "yes_token_id": yes_token,
                "no_token_id": no_token,
                "close_time": close_time,
                "min_tick_size": float(min_tick_size),
                "min_order_size": float(min_order_size)
            }
            
    except Exception as e:
        logger.error("Error fetching market details for %s: %s", market_id, e, exc_info=True)
        return None