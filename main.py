import asyncio
import sys
import logging
import os
from datetime import datetime

from market_maker_bot import MarketMakerBot
from strategies.avellaneda_stoikov import AvellanedaStoikovStrategy
from execution_client import DryRunExecutionClient

logger = logging.getLogger(__name__)

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

    # --- Configure logging to output to both console and file ---
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
        orders_list=simulated_orders_list,
        fills_list=simulated_fills_list
    )

    bot = MarketMakerBot(
        market_id=market_id,
        strategy=strategy,
        execution_client=execution_client,
        base_order_size=100.0,
        simulated_orders=simulated_orders_list, # Pass lists for reporting
        simulated_fills=simulated_fills_list
    )

    try:
        logger.info("Starting bot run...")
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Shutdown signal received by user.")
    finally:
        logger.info("Initiating graceful shutdown...")
        # The bot's own shutdown method handles liquidation and saving reports
        asyncio.run(bot.shutdown())
        logger.info("Bot shutdown complete.")

if __name__ == "__main__":
    main()