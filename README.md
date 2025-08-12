## Market-Making Bot (Polymarket CLOB)

A configurable market-making bot implementing an Avellaneda–Stoikov-style strategy for prediction markets that trade in the 0–1 price range. It supports:

- Live run (dry-run execution by default)
- Historical backtesting on recorded WebSocket data
- Data recording utilities to build your own backtest datasets


### Features
- **Strategy**: Avellaneda–Stoikov with EWMA volatility, dynamic liquidity estimation, optional trend skew, and layered quoting.
- **Queue-aware simulation**: FIFO queue modeling for fills during backtests.
- **Event pipeline**: WebSocket ingestion → order book maintenance → strategy quotes → execution client.
- **Reporting**: Console P&L/position updates and CSV outputs in live runs.


## Quickstart

### 1) Setup
```bash
python3 -m venv venv
source venv/bin/activate  # on macOS/Linux
pip install -r requirements.txt
```

Tested with Python 3.11+. macOS users may need Xcode CLT for scientific packages.


### 2) Live Run (Dry-Run)
Runs the bot against the live WebSocket feed, using a simulated execution client (no real orders).

```bash
python main.py <market_id>
# example
python main.py 0xab6faa3e66abacc484bbb4bd31ae5e2a56d6f6252b5023631f1bd9e5299fa2f8
```

Outputs
- Logs: `logs/bot_run_<market_id>_<timestamp>.log`
- CSVs at shutdown: `simulated_orders_<ts>.csv`, `simulated_fills_<ts>.csv`, `market_trades_<ts>.csv`

Configuration is loaded from `config.json` (see Configuration below). If a key is omitted or set to null, the code's constructor defaults are used.

### Configuration

- **Initialize** a local config from the template (kept out of Git):
```bash
cp config.example.json config.json
```
- **Edit** `config.json` and fill values under `strategy` and `bot`. Any missing or `null` values will fall back to the in-code defaults in `AvellanedaStoikovStrategy` and `MarketMakerBot`.
- **Optional custom path** using an environment variable:
```bash
BOT_CONFIG_PATH=/absolute/path/to/my-config.json python main.py <market_id>
# or for backtests
BOT_CONFIG_PATH=/absolute/path/to/my-config.json python backtesting/backtest.py backtesting/data/<file>.jsonl.gz
```

### 3) Record Historical Data (Optional)
Record multiple markets from the live feed to build JSONL.GZ datasets for backtesting.

1) Add market IDs to `backtesting/markets_to_record.txt` (one per line).
2) Run:
```bash
python backtesting/recorder.py
```
Data will be written to `backtesting/data/market_data_<market>_<start>_to_<end>.jsonl.gz`.


### 4) Backtest on Recorded Data
Run the strategy offline on a recorded file (sample files already in `backtesting/data/`).

```bash
python backtesting/backtest.py backtesting/data/<file>.jsonl.gz

# example
python backtesting/backtest.py backtesting/data/market_data_0xab6faa3e66abacc484bbb4bd31ae5e2a56d6f6252b5023631f1bd9e5299fa2f8_20250805_200856_to_20250805_201342.jsonl.gz
```

Notes
- The script parses the `market_id` automatically from the filename (first `0x...` token).
- Backtests also read configuration via `config.json` (or `BOT_CONFIG_PATH`). No code edits required to tune params.
- A performance report (trades, P&L, inventory) prints to the console at the end.


## Project Structure
```text
market-making/
  main.py                        # Entry point for live (dry-run) operation
  market_maker_bot.py            # Core bot: order book/events → strategy → orders/P&L
  execution_client.py            # Abstract client + DryRun client (live), used by bot
  order_book.py                  # Order book with best bid/ask/mid/spread utilities
  event_dispatcher.py            # Routes WebSocket events, updates book/trades
  trade_history.py               # Recent trades + full trade log
  strategies/
    base_strategy.py             # Strategy interface
    avellaneda_stoikov.py        # Avellaneda–Stoikov implementation with layering/skew
  backtesting/
    recorder.py                  # Multi-market recorder (JSONL.GZ output)
    backtest.py                  # Offline simulation runner + report
    simulation_client.py         # Queue-aware simulated exchange (FIFO)
    analytics.py                 # Performance report utilities
  analysis/                      # Exploratory analysis tools and plots
  logs/                          # Run logs
```


## How It Works
- **Ingestion**: WebSocket subscription to both Yes/No asset IDs of a market.
- **State**: `EventDispatcher` updates a single `OrderBook` (primary asset) and logs all trades.
- **Strategy**: `AvellanedaStoikovStrategy.calculate_quotes(...)` computes target bid/ask (optionally layered), factoring in:
  - EWMA volatility over a time window
  - Liquidity proxy via VWAS for top-of-book depth (scales `k`)
  - Inventory and time-to-close terms
  - Optional trend skew and dynamic size tolerances based on queue depth
- **Execution**: Orders are placed/canceled via an `ExecutionClient`:
  - Live: `DryRunExecutionClient` (no real orders, CSV/log only)
  - Backtest: `SimulatedExchange` (models queue priority, partial fills)
- **Accounting**: Tracks `cash`, `inventory_position`, `average_entry_price`, realized/unrealized P&L, and `total_value`.


## Configuration Reference

### Strategy (see `strategies/avellaneda_stoikov.py`)
- **gamma**: Risk aversion. Higher widens spreads and increases inventory penalty.
- **lookback_period / ewma_span**: Warmup and smoothing for volatility.
- **enable_trend_skew / trend_window / max_skew**: Optional directional skew.
- **enable_layering / max_layers / layer_price_step / layer_size_ratio**: Ladder construction.
- **k_scaling_factor**: Scales liquidity parameter inferred from book depth.
- **liquidity_fraction**: Fraction of top-of-book liquidity used for the first layer.
- **max_size_tolerance_pct / min_size_tolerance_pct / patience_depth_factor**: Re-quote tolerance vs queue depth.

### Bot (see `main.py` or `backtesting/backtest.py`)
- **total_capital**: Starting equity.
- **minting_capital_fraction**: Portion allocated to initial Yes/No shares.
- **order_value_percentage**: Target notional per quote.

