import asyncio
import sys
import logging
from market_maker_bot import MarketMakerBot

def main():
    """
    Configures and runs the market maker bot.
    """
    if len(sys.argv) != 2:
        print("Usage: python main.py <market_id>")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO, # Change to logging.DEBUG for more verbose output
        format='%(asctime)s - %(name)-24s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger('websockets').setLevel(logging.WARNING)

    market_id = sys.argv[1]

    # --- Bot Configuration ---
    bot = MarketMakerBot(
        market_id=market_id,
        order_size=10.0,
        desired_spread=0.02
    )

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")
    finally:
        bot.stop()
        print("Bot shutdown gracefully.")

if __name__ == "__main__":
    main()