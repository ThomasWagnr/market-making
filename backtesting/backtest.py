import sys
import os
import json
import gzip
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from market_maker_bot import MarketMakerBot
from strategies.avellaneda_stoikov import AvellanedaStoikovStrategy
from backtesting.simulation_client import SimulatedExchange
from backtesting.analytics import generate_performance_report

# Configure logging for the backtest
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)-28s - %(levelname)-8s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def run_backtest(data_filepath: str, market_id: str, strategy_params: dict, bot_params: dict):
    """Orchestrates the backtest from a historical data file."""
    
    logger.info(f"--- Starting Backtest for Market: {market_id} ---")
    
    # 1. Initialize all components for the simulation
    simulated_orders = []
    simulated_fills = []
    
    strategy = AvellanedaStoikovStrategy(**strategy_params)
    sim_exchange = SimulatedExchange(orders_list=simulated_orders)

    bot = MarketMakerBot(
        market_id=market_id,
        strategy=strategy,
        execution_client=sim_exchange,
        simulated_fills=simulated_fills,
        **bot_params
    )
    
    # 2. The Simulation Engine (Event Loop)
    logger.info(f"Loading and processing data from {data_filepath}...")
    try:
        with gzip.open(data_filepath, 'rt') as f:
            for i, line in enumerate(f):
                log_entry = json.loads(line)
                message_data = log_entry['data']
                timestamp = log_entry['timestamp']
                
                # Update the simulated exchange's internal clock
                sim_exchange.set_time(timestamp)

                # This is a list of events, just like from the live feed
                events = message_data if isinstance(message_data, list) else [message_data]
                
                # Before the bot reacts, the exchange must process any trades from the message
                for event in events:
                    if event.get("event_type") == "last_trade_price":
                        sim_exchange.check_for_fills(event, bot._check_fills)
                
                # Now, dispatch the message to update the bot's view of the world (order book)
                # and trigger its strategy logic (_on_update).
                bot.dispatcher.dispatch(json.dumps(events))
                
                if (i + 1) % 5000 == 0:
                    logger.info(f"Processed {i+1} messages...")

    except FileNotFoundError:
        logger.critical(f"Data file not found at {data_filepath}. Please run the recorder first.")
        return
    except Exception as e:
        logger.critical(f"An error occurred during the simulation loop: {e}", exc_info=True)
        return

    # 3. Generate the final performance report
    bot._update_pnl() # Final P&L calculation
    bot_final_state = {
        'realized_pnl': bot.realized_pnl,
        'unrealized_pnl': bot.unrealized_pnl,
        'total_value': bot.total_value,
        'inventory_position': bot.inventory_position
    }
    generate_performance_report(bot_final_state, simulated_fills)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path_to_data_file>")
        print(f"Example: python backtesting/backtest.py backtesting/market_data_0xabc... .jsonl.gz")
        sys.exit(1)
        
    data_file = sys.argv[1]
    # Infer market_id from filename, e.g., "market_data_MARKET_ID.jsonl.gz"
    market_id = os.path.basename(data_file).split('_')[2].replace('.jsonl.gz', '')

    # --- Define the parameters for this specific backtest run ---
    strategy_config = {
        'gamma': 10.0,
        'lookback_period': 100,
        'ewma_span': 50,
        'enable_trend_skew': True,
        'k_scaling_factor': 10.0,
        'max_skew': 0.005,
        'trend_window': 20
    }
    bot_config = {
        'base_order_size': 100.0
    }
    
    run_backtest(
        data_filepath=data_file,
        market_id=market_id,
        strategy_params=strategy_config,
        bot_params=bot_config
    )