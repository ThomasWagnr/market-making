import logging
import aiohttp
from datetime import datetime, timezone
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

MARKET_INFO_URL = "https://clob.polymarket.com/markets/{market_id}"

async def get_market_tokens(session: aiohttp.ClientSession, market_id: str) -> Optional[Tuple[str, str]]:
    """
    Fetches both the YES and NO token IDs for a given market ID.

    Args:
        session: An active aiohttp.ClientSession.
        market_id: The string ID of the market.

    Returns:
        A tuple containing (yes_token_id, no_token_id), or None if either is not found.
    """
    url = MARKET_INFO_URL.format(market_id=market_id)
    logger.debug("Fetching tokens for market ID: %s", market_id)

    try:
        async with session.get(url, timeout=10) as resp:
            resp.raise_for_status()
            info = await resp.json()
            
            yes_token = next((t.get('token_id') for t in info.get('tokens', []) if t.get('outcome') == 'Yes'), None)
            no_token = next((t.get('token_id') for t in info.get('tokens', []) if t.get('outcome') == 'No'), None)

            if not yes_token or not no_token:
                logger.warning("Could not find both YES and NO tokens for market %s", market_id)
                return None
            
            logger.info("Found YES token (%s) and NO token (%s)", yes_token, no_token)
            return yes_token, no_token
            
    except aiohttp.ClientError as e:
        logger.error("Aiohttp client error fetching tokens for market %s: %s", market_id, e)
        return None
    except Exception as e:
        logger.error("Unexpected error fetching tokens for market %s: %s", market_id, e, exc_info=True)
        return None

async def get_market_close_time(session: aiohttp.ClientSession, market_id: str) -> Optional[datetime]:
    """
    Fetches the closing time for a given market ID and returns it as a timezone-aware datetime object.

    Args:
        session: An active aiohttp.ClientSession.
        market_id: The hex string ID of the market.

    Returns:
        A timezone-aware datetime object representing the market's closing time, or None on failure.
    """
    url = MARKET_INFO_URL.format(market_id=market_id)
    logger.debug("Fetching close time for market ID: %s", market_id)

    try:
        async with session.get(url, timeout=10) as resp:
            resp.raise_for_status()
            info = await resp.json()

            close_time_str = info.get('end_date_iso')
            if not close_time_str:
                logger.warning("Market close time not found in API response for market %s", market_id)
                return None

            if close_time_str.endswith('Z'):
                close_time_str = close_time_str[:-1] + '+00:00'

            return datetime.fromisoformat(close_time_str)

    except aiohttp.ClientError as e:
        logger.error("Aiohttp client error fetching close time for market %s: %s", market_id, e)
        return None
    except (ValueError, TypeError) as e:
        logger.error("Error parsing close time for market %s: %s", market_id, e)
        return None
    except Exception as e:
        logger.error("Unexpected error fetching close time for market %s: %s", market_id, e, exc_info=True)
        return None