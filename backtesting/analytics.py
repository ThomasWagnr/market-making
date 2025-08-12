import os
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any, Optional

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _max_drawdown(equity_series: pd.Series) -> float:
    if equity_series.empty:
        return 0.0
    running_max = equity_series.cummax()
    drawdowns = (equity_series - running_max) / running_max
    return float(drawdowns.min())


def generate_performance_report(
    bot_final_state: dict,
    fills_log: List[Dict[str, Any]],
    orders_log: Optional[List[Dict[str, Any]]] = None,
    output_dir: Optional[str] = None,
    book_snapshots: Optional[List[Dict[str, Any]]] = None,
    order_state_snapshots: Optional[List[Dict[str, Any]]] = None,
):
    """Calculates, prints, and optionally saves key performance metrics from a backtest run."""
    
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
    fills_df.sort_values('timestamp', inplace=True)
    
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
    
    # --- Risk Metrics ---
    if 'equity' in fills_df.columns:
        equity_curve = fills_df.set_index('timestamp')['equity']
        mdd = _max_drawdown(equity_curve)
        print("\n--- Risk ---")
        print(f"Max Drawdown:           {mdd:.2%}")
    else:
        mdd = None

    # --- Execution Quality ---
    # Round-trip hit ratio approximation
    round_trips = 0
    profitable_round_trips = 0
    inventory = 0.0
    avg_entry = 0.0
    for _, row in fills_df.iterrows():
        side, px, sz = row['side'], float(row['price']), float(row['size'])
        if side == 'BUY':
            new_cost = avg_entry * max(inventory, 0.0) + px * sz
            inventory += sz
            if inventory > 1e-9:
                avg_entry = new_cost / inventory
        else:  # SELL
            if inventory > 0:
                close_sz = min(inventory, sz)
                pnl = (px - avg_entry) * close_sz
                round_trips += 1
                if pnl > 0:
                    profitable_round_trips += 1
                inventory -= close_sz
            else:
                # Short-side bookkeeping (not typical here, but handled for completeness)
                new_cost = avg_entry * max(-inventory, 0.0) + px * sz
                inventory -= sz
                if -inventory > 1e-9:
                    avg_entry = new_cost / (-inventory)
    hit_ratio = (profitable_round_trips / round_trips) if round_trips > 0 else None
    if hit_ratio is not None:
        print("\n--- Execution Quality ---")
        print(f"Round-trip Hit Ratio:   {hit_ratio:.2%} ({profitable_round_trips}/{round_trips})")

    # Turnover
    turnover = (fills_df['price'] * fills_df['size']).sum()
    print(f"Turnover (Notional):    ${turnover:,.2f}")

    # Time to first fill per order (if orders available)
    time_to_first_fill = None
    if orders_log:
        orders_df = pd.DataFrame(orders_log)
        if 'timestamp' in orders_df.columns:
            orders_df['timestamp'] = pd.to_datetime(orders_df['timestamp'])
        first_fills = (
            fills_df.dropna(subset=['order_id'])
                    .sort_values('timestamp')
                    .groupby('order_id')['timestamp']
                    .first()
        )
        placements = orders_df[orders_df['action'] == 'PLACE']
        placements = placements.set_index('order_id')['timestamp'] if 'order_id' in placements.columns else pd.Series(dtype='datetime64[ns]')
        joined = first_fills.to_frame('fill_time').join(placements.to_frame('place_time'), how='inner')
        if not joined.empty:
            deltas = (joined['fill_time'] - joined['place_time']).dt.total_seconds()
            time_to_first_fill = deltas.mean()
            print(f"Avg Time-to-First-Fill: {time_to_first_fill:.2f}s over {len(deltas)} orders")
    
    print("="*50)

    # --- Artifacts ---
    if output_dir:
        _ensure_dir(output_dir)
        # Save fills
        fills_path = os.path.join(output_dir, 'fills.csv')
        fills_df.to_csv(fills_path, index=False)
        # Save orders, if provided
        if orders_log:
            orders_df = pd.DataFrame(orders_log)
            orders_df.to_csv(os.path.join(output_dir, 'orders.csv'), index=False)
        # Save order book snapshots
        if book_snapshots:
            book_df = pd.DataFrame(book_snapshots)
            book_df['timestamp'] = pd.to_datetime(book_df['timestamp'], unit='s', errors='coerce')
            book_df.sort_values('timestamp', inplace=True)
            book_df.to_csv(os.path.join(output_dir, 'book_snapshots.csv'), index=False)
        # Save order state snapshots
        if order_state_snapshots:
            oss_df = pd.DataFrame(order_state_snapshots)
            oss_df['timestamp'] = pd.to_datetime(oss_df['timestamp'], unit='s', errors='coerce')
            oss_df.sort_values('timestamp', inplace=True)
            oss_df.to_csv(os.path.join(output_dir, 'order_state_snapshots.csv'), index=False)

        # Derive and save aggressive trade summary if fills include shutdown sweeps
        if 'equity' in fills_df.columns and 'order_id' in fills_df.columns:
            # Heuristic placeholder: in future, tag aggressive events explicitly
            # For now, export last snapshot of equity and inventory for reference
            summary_tail = fills_df.tail(1)
            summary_tail.to_csv(os.path.join(output_dir, 'end_state.csv'), index=False)
        # Save summary
        summary = {
            'final_equity': float(final_equity),
            'realized_pnl': float(realized_pnl),
            'unrealized_pnl': float(unrealized_pnl),
            'ending_inventory': float(bot_final_state['inventory_position']),
            'total_trades': int(total_trades),
            'total_volume': float(total_volume),
            'turnover': float(turnover),
            'max_drawdown': float(mdd) if mdd is not None else None,
            'profitable_round_trips': int(profitable_round_trips),
            'round_trips': int(round_trips),
            'hit_ratio': float(hit_ratio) if hit_ratio is not None else None,
            'avg_time_to_first_fill_seconds': float(time_to_first_fill) if time_to_first_fill is not None else None,
            'generated_at': datetime.utcnow().isoformat() + 'Z',
        }
        pd.Series(summary).to_json(os.path.join(output_dir, 'summary.json'))