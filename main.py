import asyncio
import sys
import logging
import os
import csv
from datetime import datetime
from typing import List, Dict, Any

from market_maker_bot import MarketMakerBot
from strategies.avellaneda_stoikov import AvellanedaStoikovStrategy
from execution_client import DryRunExecutionClient
from config import load_config

logger = logging.getLogger(__name__)

def save_report_to_csv(filename: str, data: List[Dict[str, Any]]):
    """Saves a list of dictionaries to a CSV file."""
    if not data:
        logger.info(f"No data to save for {filename}.")
        return
        
    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        logger.info(f"Successfully saved report to {filename}")
    except Exception as e:
        logger.error(f"Failed to save report to {filename}: {e}", exc_info=True)

def main():
    """
    Configures and runs the market maker bot.
    """
    if len(sys.argv) != 2:
        print("Usage: python main.py <market_id>")
        sys.exit(1)

    market_id = sys.argv[1]

    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f"bot_run_{market_id}_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)-24s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename), # Log to a file
            logging.StreamHandler()            # Log to the console
        ]
    )
    logging.getLogger('websockets').setLevel(logging.WARNING)

    # --- 1. Load configuration and configure the Strategy/Bot ---
    config = load_config()
    strategy = AvellanedaStoikovStrategy(**config.get('strategy', {}))

    simulated_orders_list = []
    simulated_fills_list = []

    execution_client = DryRunExecutionClient(
        orders_list=simulated_orders_list
    )

    bot = MarketMakerBot(
        market_id=market_id,
        strategy=strategy,
        execution_client=execution_client,
        simulated_fills=simulated_fills_list,
        **config.get('bot', {})
    )

    try:
        logger.info("Starting bot run...")
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Shutdown signal received by user.")
    finally:
        logger.info("Initiating graceful shutdown...")
        asyncio.run(bot.shutdown())

        logger.info("Saving final reports...")
        if bot.simulated_fills:
            save_report_to_csv(f'simulated_fills_{timestamp}.csv', bot.simulated_fills)
        if simulated_orders_list:
            save_report_to_csv(f'simulated_orders_{timestamp}.csv', simulated_orders_list)
        if bot.trade_history:
            save_report_to_csv(f'market_trades_{timestamp}.csv', bot.trade_history.trade_log)
        
        logger.info("Bot shutdown complete.")

if __name__ == "__main__":
    main()