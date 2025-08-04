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

    # --- 1. Choose and Configure the Strategy ---
    strategy = AvellanedaStoikovStrategy(
        gamma=10.0,
        lookback_period=20,
        ewma_span=20,
        trend_skew=True,
        k_scaling_factor=10.0,
        max_skew=0.005
    )

    simulated_orders_list = []
    simulated_fills_list = []

    execution_client = DryRunExecutionClient(
        orders_list=simulated_orders_list
    )

    bot = MarketMakerBot(
        market_id=market_id,
        strategy=strategy,
        execution_client=execution_client,
        base_order_size=100.0,
        simulated_fills=simulated_fills_list
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
        save_report_to_csv('simulated_orders.csv', simulated_orders_list)
        save_report_to_csv('simulated_fills.csv', bot.simulated_fills)
        save_report_to_csv('trade_history.csv', bot.trade_history.trade_log)

        logger.info("Bot shutdown complete.")

if __name__ == "__main__":
    main()