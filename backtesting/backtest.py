# backtesting/backtest.py

import sys
import os
import json
import gzip
import logging
from datetime import datetime, timezone
import asyncio
import aiohttp
from typing import Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from market_maker_bot import MarketMakerBot
from strategies.avellaneda_stoikov import AvellanedaStoikovStrategy
from backtesting.simulation_client import SimulatedExchange
from backtesting.analytics import generate_performance_report
from utils import get_market_details
from event_dispatcher import EventDispatcher
from order_book import OrderBook
from trade_history import TradeHistory
from config import load_config

# Configure logging for the backtest
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)-28s - %(levelname)-8s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


async def run_backtest(data_filepath: str, market_id: str, strategy_params: Dict, bot_params: Dict):
    """Orchestrates the backtest from a historical data file."""
    
    logger.info(f"--- Starting Backtest for Market: {market_id} ---")
    
    # 1. Initialize all components for the simulation
    simulated_orders = []
    simulated_fills = []
    book_snapshots = []
    order_state_snapshots = []
    
    strategy = AvellanedaStoikovStrategy(**strategy_params)
    sim_exchange = SimulatedExchange(orders_list=simulated_orders)

    bot = MarketMakerBot(
        market_id=market_id,
        strategy=strategy,
        execution_client=sim_exchange,
        simulated_fills=simulated_fills,
        simulated_taker_trades=[],
        **bot_params
    )
    
    # 2. Perform the bot's async startup logic to get token IDs and set up the dispatcher
    logger.info("Fetching token IDs for backtest setup...")
    async with aiohttp.ClientSession() as session:
        market_details = await get_market_details(session, market_id)
        if not market_details:
            logger.critical("Could not fetch token IDs for backtest. Aborting.")
            return
        
        bot.yes_token_id = market_details['yes_token_id']
        bot.no_token_id = market_details['no_token_id']
        bot.market_close_time = market_details['close_time']
        bot.min_order_size = market_details['min_order_size']
        tick_size = market_details['min_tick_size']

        bot.order_book = OrderBook(bot.market_id, tick_size=tick_size)
        bot.trade_history = TradeHistory(bot.market_id)
        bot.dispatcher = EventDispatcher(
            primary_order_book=bot.order_book,
            primary_asset_id=bot.yes_token_id,
            trade_history=bot.trade_history,
            update_callback=bot._on_update
        )
    
    # 3. The Simulation Engine (Event Loop)
    logger.info(f"Loading and processing data from {data_filepath}...")
    message_count = 0
    first_event_ts: float | None = None
    try:
        with gzip.open(data_filepath, 'rt') as f:
            for line in f:
                log_entry = json.loads(line)
                message_data = log_entry['data']
                timestamp = log_entry['timestamp']

                # Initialize backtest session timing on first event
                if first_event_ts is None:
                    first_event_ts = float(timestamp)
                    bot.market_start_time = datetime.fromtimestamp(first_event_ts, tz=timezone.utc)
                    session_secs = (bot.market_close_time - bot.market_start_time).total_seconds() if bot.market_close_time else 0.0
                    if session_secs <= 0:
                        logger.warning("Computed non-positive session duration (%.2f s). Using fallback of 1.0s.", session_secs)
                        session_secs = 1.0
                    bot.session_duration_seconds = session_secs
                    bot.time_horizon = 1.0
                
                # Update the simulated exchange's internal clock
                sim_exchange.set_time(timestamp)

                events = message_data if isinstance(message_data, list) else [message_data]
                
                # Before the bot reacts, the exchange must process any trades from the message
                for event in events:
                # Attach simulated time to every event
                    event["sim_time"] = timestamp
                    if event.get("event_type") == "last_trade_price":
                        sim_exchange.check_for_fills(event, bot._check_fills)
                
                # Now, dispatch the message to update the bot's view of the world (order book)
                # and trigger its strategy logic (_on_update).
                bot.dispatcher.dispatch(json.dumps(events))

                # After bot updates, snapshot the current book and our active orders
                try:
                    book_snapshots.append({
                        'timestamp': timestamp,
                        'best_bid': bot.order_book.best_bid if bot.order_book else None,
                        'best_ask': bot.order_book.best_ask if bot.order_book else None,
                        'mid_price': bot.order_book.mid_price if bot.order_book else None,
                        'spread': bot.order_book.spread if bot.order_book else None,
                    })
                    # Active bids
                    for oid, o in bot.active_bids.items():
                        order_state_snapshots.append({
                            'timestamp': timestamp,
                            'order_id': oid,
                            'side': 'BUY',
                            'price': o.get('price'),
                            'size': o.get('size'),
                            'volume_ahead': o.get('volume_ahead'),
                        })
                    # Active asks
                    for oid, o in bot.active_asks.items():
                        order_state_snapshots.append({
                            'timestamp': timestamp,
                            'order_id': oid,
                            'side': 'SELL',
                            'price': o.get('price'),
                            'size': o.get('size'),
                            'volume_ahead': o.get('volume_ahead'),
                        })
                except Exception:
                    pass
                
                message_count += 1
                if message_count % 10000 == 0:
                    logger.info(f"Processed {message_count} messages...")

    except FileNotFoundError:
        logger.critical(f"Data file not found at {data_filepath}. Please run the recorder first.")
        return
    except Exception as e:
        logger.critical(f"An error occurred during the simulation loop: {e}", exc_info=True)
        return

    # 4. Shutdown and liquidate any remaining inventory aggressively
    logger.info(f"Simulation loop complete. Processed {message_count} total messages.")
    await bot.shutdown()
    bot._update_pnl() # Final P&L calculation after shutdown
    bot_final_state = {
        'realized_pnl': bot.realized_pnl,
        'unrealized_pnl': bot.unrealized_pnl,
        'total_value': bot.total_value,
        'inventory_position': bot.inventory_position
    }
    # Create a default output directory name next to data file
    output_dir = os.path.join(os.path.dirname(data_filepath), f"report_{os.path.basename(data_filepath).replace('.jsonl.gz','')}")
    generate_performance_report(
        bot_final_state,
        simulated_fills,
        orders_log=simulated_orders,
        output_dir=output_dir,
        book_snapshots=book_snapshots,
        order_state_snapshots=order_state_snapshots,
        taker_trades=getattr(bot, 'simulated_taker_trades', None)
    )


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path_to_data_file>")
        print(f"Example: python backtesting/backtest.py backtesting/data/market_data_0xabc...jsonl.gz")
        sys.exit(1)
        
    data_file = sys.argv[1]
    
    try:
        filename = os.path.basename(data_file)
        # 1. Find the part of the filename that starts with '0x'
        market_id_part = next((part for part in filename.split('_') if part.startswith('0x')), None)
        
        if not market_id_part:
            raise ValueError("Market ID starting with '0x' not found in filename.")
            
        # 2. Extract the first 66 characters (0x + 64 hex chars)
        market_id_from_filename = market_id_part[:66]
        
        # 3. Final check to ensure it looks like a valid ID
        if len(market_id_from_filename) != 66:
             raise ValueError("Parsed Market ID is not the correct length.")

    except (IndexError, ValueError) as e:
        print(f"Error parsing market_id from filename: {e}")
        print("Please ensure filename follows the 'market_data_MARKETID_...' format.")
        sys.exit(1)

    # --- Load parameters from config file ---
    config = load_config()
    strategy_config = config.get('strategy', {})
    bot_config = config.get('bot', {})
    
    # Run the backtest
    asyncio.run(run_backtest(
        data_filepath=data_file,
        market_id=market_id_from_filename,
        strategy_params=strategy_config,
        bot_params=bot_config
    ))