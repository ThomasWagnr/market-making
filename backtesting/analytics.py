import pandas as pd
from typing import List, Dict, Any

def generate_performance_report(bot_final_state: dict, fills_log: List[Dict[str, Any]]):
    """Calculates and prints key performance metrics from a backtest run."""
    
    print("\n" + "="*50)
    print("--- Backtest Performance Report ---")
    print("="*50)

    if not fills_log:
        print("No trades were executed during the simulation.")
        print("---------------------------------")
        return

    # Use pandas for powerful and easy analysis
    fills_df = pd.DataFrame(fills_log)
    fills_df['price'] = fills_df['price'].astype(float)
    fills_df['size'] = fills_df['size'].astype(float)
    fills_df['timestamp'] = pd.to_datetime(fills_df['timestamp'])
    
    # --- P&L Metrics ---
    realized_pnl = bot_final_state['realized_pnl']
    unrealized_pnl = bot_final_state['unrealized_pnl']
    final_equity = bot_final_state['total_value']
    
    print(f"\n--- P&L ---")
    print(f"Final Equity:           ${final_equity:,.2f}")
    print(f"Realized P&L:           ${realized_pnl:+.2f}")
    print(f"Unrealized P&L:         ${unrealized_pnl:+.2f}")
    print(f"Ending Inventory:       {bot_final_state['inventory_position']:+.2f} shares")

    # --- Trade Stats ---
    total_trades = len(fills_df)
    buy_trades = fills_df[fills_df['side'] == 'BUY']
    sell_trades = fills_df[fills_df['side'] == 'SELL']
    total_volume = fills_df['size'].sum()
    
    print("\n--- Trade Stats ---")
    print(f"Total Trades:           {total_trades}")
    print(f"Buy Trades:             {len(buy_trades)}")
    print(f"Sell Trades:            {len(sell_trades)}")
    print(f"Total Volume Traded:    {total_volume:,.2f}")

    if not buy_trades.empty:
        avg_buy_price = (buy_trades['price'] * buy_trades['size']).sum() / buy_trades['size'].sum()
        print(f"Average Buy Price:      ${avg_buy_price:.4f}")

    if not sell_trades.empty:
        avg_sell_price = (sell_trades['price'] * sell_trades['size']).sum() / sell_trades['size'].sum()
        print(f"Average Sell Price:     ${avg_sell_price:.4f}")
    
    print("="*50)