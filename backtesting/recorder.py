import asyncio
import sys
import os
import json
import gzip
import time
import logging
import aiohttp
import websockets
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import get_yes_token_for_market

# --- Configuration & Logging Setup ---
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)-28s - %(levelname)-8s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ShutdownManager:
    """A helper to manage shutdown signals for a clean exit."""
    def __init__(self):
        self.running = True
    
    async def shutdown(self, signal_name: str):
        """Initiates the shutdown process."""
        if self.running:
            logger.warning("Received shutdown signal: %s. Shutting down gracefully.", signal_name)
            self.running = False
            # Give any running tasks a moment to finish
            await asyncio.sleep(1)

async def record_market(market_id: str, token_id: str, shutdown_manager: ShutdownManager):
    """
    Connects to a market's WebSocket feed and saves all messages to a
    compressed line-delimited JSON file.
    """

    data_dir = os.path.join('backtesting', 'data')

    os.makedirs(data_dir, exist_ok=True)

    output_filename = os.path.join(data_dir, f'market_data_{market_id}.jsonl.gz')

    logger.info(f"Starting to record data for market '{market_id}' into '{output_filename}'")
    logger.info("Press Ctrl+C to stop recording.")

    messages_recorded = 0
    while shutdown_manager.running:
        try:
            with gzip.open(output_filename, 'at') as f:
                async with websockets.connect(WS_URL, ping_interval=20) as ws:
                    subscribe_msg = {"assets_ids": [token_id], "type": "market"}
                    await ws.send(json.dumps(subscribe_msg))
                    logger.info("Connection successful. Subscribed to market feed.")

                    # This loop will run until the outer while loop is broken by the shutdown manager
                    while shutdown_manager.running:
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue # No message received, check if we should shut down

                        try:
                            log_entry = {
                                'timestamp': time.time(),
                                'data': json.loads(message)
                            }
                            f.write(json.dumps(log_entry) + '\n')
                            messages_recorded += 1
                            
                            # Flush data to disk every 100 messages for safety
                            if messages_recorded % 100 == 0:
                                f.flush()
                            
                            print(f"\rMessages Recorded: {messages_recorded}", end="", flush=True)

                        except json.JSONDecodeError:
                            logger.warning("Received a non-JSON message: %s", message)

        except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError) as e:
            if shutdown_manager.running:
                logger.warning("Connection closed: %s. Retrying in 10 seconds...", e)
                await asyncio.sleep(10)
        except Exception as e:
            if shutdown_manager.running:
                logger.error("An unexpected error occurred: %s. Retrying in 10 seconds...", e, exc_info=True)
                await asyncio.sleep(10)

async def main():
    """Main function to orchestrate the recorder."""
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <market_id>")
        sys.exit(1)
    
    market_id = sys.argv[1]

    shutdown_manager = ShutdownManager()
    loop = asyncio.get_running_loop()
    
    if sys.platform != "win32":
        import signal
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown_manager.shutdown(s.name)))
    
    async with aiohttp.ClientSession() as session:
        token_id = await get_yes_token_for_market(session, market_id)
        if not token_id:
            logger.critical("Could not get token ID. Aborting.")
            return

    await record_market(market_id, token_id, shutdown_manager)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        print("\nRecorder has shut down.")