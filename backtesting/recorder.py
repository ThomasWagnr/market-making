import asyncio
import sys
import os
import json
import gzip
import time
import logging
from datetime import datetime
from typing import Optional, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import aiohttp
import websockets
from utils import get_market_details

# --- Configuration & Logging Setup ---
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
CONFIG_FILENAME = "markets_to_record.txt"
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(name)-28s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class MarketRecorder:
    """Manages the recording process for a single market."""
    def __init__(self, market_id: str, session: aiohttp.ClientSession, run_timestamp: str):
        self.market_id = market_id
        self.session = session
        self.token_ids: Optional[List[str]] = None
        self.run_timestamp = run_timestamp
        
        partial_filename = f'market_data_{self.market_id}_{self.run_timestamp}.part.jsonl.gz'
        self.partial_filepath = os.path.join(DATA_DIR, partial_filename)
        
        self.messages_recorded = 0

        self.initial_retry_delay = 1   # Start with a 1-second delay
        self.max_retry_delay = 64      # Cap the delay at 64 seconds
        self.retry_multiplier = 2.0    # Double the delay on each failure

    async def start(self, shutdown_event: asyncio.Event):
        """Starts the recording process and runs until shutdown is signaled."""
        logger.info(f"[{self.market_id}] Initializing recorder task...")
        
        details = await get_market_details(self.session, self.market_id)
        if not details:
            logger.error(f"[{self.market_id}] Could not get token IDs. Aborting this recorder task.")
            return
        self.token_ids = [details['yes_token_id'], details['no_token_id']]

        current_retry_delay = self.initial_retry_delay

        while not shutdown_event.is_set():
            try:
                with gzip.open(self.partial_filepath, 'at') as f:
                    async with websockets.connect(WS_URL, ping_interval=20) as ws:
                        subscribe_msg = {"assets_ids": self.token_ids, "type": "market"}
                        await ws.send(json.dumps(subscribe_msg))
                        logger.info(f"[{self.market_id}] Connection successful. Subscribed and recording.")

                        current_retry_delay = self.initial_retry_delay

                        while not shutdown_event.is_set():
                            try:
                                message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            except asyncio.TimeoutError:
                                continue

                            try:
                                log_entry = {
                                    'timestamp': time.time(),
                                    'data': json.loads(message)
                                }
                                f.write(json.dumps(log_entry) + '\n')
                                self.messages_recorded += 1
                                
                                if self.messages_recorded % 100 == 0:
                                    f.flush()
                            except json.JSONDecodeError:
                                logger.warning(f"[{self.market_id}] Received a non-JSON message.")
            
            except asyncio.CancelledError:
                logger.info(f"[{self.market_id}] Recorder task cancelled.")
                break
            except Exception as e:
                if not shutdown_event.is_set():
                    logger.error("An error occurred: %s. Retrying in %d s...", e, current_retry_delay)
                    await asyncio.sleep(current_retry_delay)
                    current_retry_delay = min(self.max_retry_delay, current_retry_delay * self.retry_multiplier)
        
        logger.info(f"[{self.market_id}] Recording stopped. Total messages: {self.messages_recorded}")

    def finalize(self):
        """Renames the partial file to its final name upon shutdown."""
        if not os.path.exists(self.partial_filepath) or self.messages_recorded == 0:
            logger.warning(f"[{self.market_id}] No data recorded, no file to finalize.")
            if os.path.exists(self.partial_filepath):
                os.remove(self.partial_filepath)
            return
            
        end_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_filename = f'market_data_{self.market_id}_{self.run_timestamp}_to_{end_timestamp}.jsonl.gz'
        final_filepath = os.path.join(DATA_DIR, final_filename)
        
        try:
            os.rename(self.partial_filepath, final_filepath)
            logger.info(f"[{self.market_id}] Finalized recording. File saved as: {final_filename}")
        except Exception as e:
            logger.error(f"[{self.market_id}] Could not finalize/rename data file: {e}")


async def monitor_progress(recorders: List[MarketRecorder], shutdown_event: asyncio.Event):
    """Periodically prints the total number of messages recorded."""
    while not shutdown_event.is_set():
        total_messages = sum(rec.messages_recorded for rec in recorders)
        print(f"\rTotal Messages Recorded: {total_messages}", end="", flush=True)
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            break


async def main():
    """Main function to read the config and launch all recorders."""
    logger.info("--- Starting Multi-Market Recorder ---")
    os.makedirs(DATA_DIR, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_filepath = os.path.join(script_dir, CONFIG_FILENAME)

    try:
        with open(config_filepath, 'r') as f:
            market_ids = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        logger.critical(f"Configuration file not found at: {config_filepath}. Please create it.")
        return

    if not market_ids:
        logger.warning("No market IDs found in the configuration file. Exiting.")
        return

    logger.info(f"Found {len(market_ids)} markets to record from '{os.path.basename(config_filepath)}'.")
    
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def signal_handler(sig):
        logger.warning("Shutdown signal received: %s. Draining connections...", sig.name)
        shutdown_event.set()

    if sys.platform != "win32":
        import signal
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler, sig)

    async with aiohttp.ClientSession() as session:
        recorders = [MarketRecorder(market_id, session, run_timestamp) for market_id in market_ids]
        
        recorder_tasks = [asyncio.create_task(rec.start(shutdown_event)) for rec in recorders]
        monitor_task = asyncio.create_task(monitor_progress(recorders, shutdown_event))
        
        all_tasks = recorder_tasks + [monitor_task]
        await asyncio.gather(*all_tasks)

    logger.info("All recorder tasks have stopped. Finalizing files...")
    for rec in recorders:
        rec.finalize()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        print("\nRecorder has shut down.")