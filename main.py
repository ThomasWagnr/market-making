import asyncio
import sys
import logging
import os
from datetime import datetime
from market_maker_bot import MarketMakerBot
from strategies.avellaneda_stoikov import AvellanedaStoikovStrategy

def main():
    """
    Configures and runs the market maker bot.
    """
    if len(sys.argv) != 2:
        print("Usage: python main.py <market_id>")
        sys.exit(1)

    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"{log_dir}/bot_run_{timestamp}.log"

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

    market_id = sys.argv[1]

    # --- 1. Choose and Configure the Strategy ---
    strategy = AvellanedaStoikovStrategy(
        gamma=10.0,
        lookback_period=20,
        ewma_span=20,
        trend_skew=True
    )

    # --- 2. Configure the Bot with the Chosen Strategy ---
    bot = MarketMakerBot(
        market_id=market_id,
        strategy=strategy,
        risk_fraction=0.1,
        dry_run=True
    )

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\nBot stopped manually.")
    finally:
        bot.stop()
        bot.save_report()
        print("Bot shutdown gracefully.")

if __name__ == "__main__":
    main()