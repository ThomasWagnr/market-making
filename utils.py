import logging
import aiohttp
from typing import Optional

logger = logging.getLogger(__name__)

MARKET_INFO_URL = "https://clob.polymarket.com/markets/{market_id}"

async def get_yes_token_for_market(session: aiohttp.ClientSession, market_id: str) -> Optional[str]:
    """
    Fetches the YES token ID for a given market ID using an existing aiohttp session.

    Args:
        session: An active aiohttp.ClientSession.
        market_id: The hex string ID of the market.

    Returns:
        The string token ID for the 'Yes' outcome, or None if not found or an error occurs.
    """
    url = MARKET_INFO_URL.format(market_id=market_id)
    logger.debug("Fetching token for market ID: %s", market_id)

    try:
        async with session.get(url, timeout=10) as resp:
            # Raises an HTTPError for bad responses (4xx or 5xx)
            resp.raise_for_status()
            info = await resp.json()

            token_id = next(
                (t.get('token_id') for t in info.get('tokens', []) if t.get('outcome') == 'Yes'),
                None
            )

            if not token_id:
                logger.warning("'Yes' token not found in API response for market %s", market_id)

            return token_id

    except aiohttp.ClientError as e:
        logger.error("Aiohttp client error fetching token for market %s: %s", market_id, e)
        return None
    except Exception as e:
        # Catch any other unexpected errors
        logger.error("Unexpected error fetching token for market %s: %s", market_id, e, exc_info=True)
        return None