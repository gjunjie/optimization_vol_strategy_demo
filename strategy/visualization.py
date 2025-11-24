"""
Visualization utilities for volatility arbitrage strategy.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any


def plot_iv_vs_moneyness(moneyness_clean, iv_clean):
    """
    Plot Implied Volatility vs Moneyness.
    
    Parameters
    ----------
    moneyness_clean : array-like
        Clean moneyness values
    iv_clean : array-like
        Clean implied volatility values
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(moneyness_clean, iv_clean, alpha=0.5, s=10)
    plt.xlabel('Moneyness')
    plt.ylabel('Implied Volatility')
    plt.title('Moneyness vs. Implied Volatility')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_snapshot(snapshot_idx: int, rolling_snapshots: List[Dict[str, Any]]):
    """
    Plot a single snapshot with signals.
    
    Parameters
    ----------
    snapshot_idx : int
        Index of the snapshot to plot
    rolling_snapshots : List[Dict]
        List of snapshot dictionaries
    """
    selected_snapshot = rolling_snapshots[snapshot_idx]
    
    # Extract data
    train_data = selected_snapshot['train_data']
    test_data = selected_snapshot['test_data']
    poly_model = selected_snapshot['poly_model']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot training data (gray background)
    ax.scatter(train_data['Moneyness'], train_data['ImpliedVol'], 
              alpha=0.2, s=30, c='lightgray', label='Training Data (10 min window)')
    
    # Plot fitted curve
    moneyness_range = np.linspace(train_data['Moneyness'].min(), 
                                  train_data['Moneyness'].max(), 200)
    fitted_iv = poly_model(moneyness_range)
    ax.plot(moneyness_range, fitted_iv, 'b-', linewidth=2.5, 
           label='Fitted Polynomial (degree 3)', zorder=5)
    
    # Plot test data with signals
    buy_mask = test_data['Signal'] == 'Buy'
    sell_mask = test_data['Signal'] == 'Sell'
    hold_mask = test_data['Signal'] == 'Hold'
    
    if hold_mask.any():
        ax.scatter(test_data.loc[hold_mask, 'Moneyness'], 
                  test_data.loc[hold_mask, 'ImpliedVol'], 
                  alpha=0.5, s=80, c='yellow', edgecolors='orange',
                  linewidths=1.5, label='Hold (Test)', zorder=6)
    
    if buy_mask.any():
        ax.scatter(test_data.loc[buy_mask, 'Moneyness'], 
                  test_data.loc[buy_mask, 'ImpliedVol'], 
                  alpha=0.9, s=200, c='green', marker='^', edgecolors='darkgreen', 
                  linewidths=2, label=f'BUY Signal ({buy_mask.sum()})', zorder=7)
        
        # Draw lines from points to fitted curve
        for idx in test_data[buy_mask].index:
            m = test_data.loc[idx, 'Moneyness']
            actual_iv = test_data.loc[idx, 'ImpliedVol']
            pred_iv = poly_model(m)
            ax.plot([m, m], [pred_iv, actual_iv], 'g--', alpha=0.5, linewidth=1.5)
    
    if sell_mask.any():
        ax.scatter(test_data.loc[sell_mask, 'Moneyness'], 
                  test_data.loc[sell_mask, 'ImpliedVol'], 
                  alpha=0.9, s=200, c='red', marker='v', edgecolors='darkred', 
                  linewidths=2, label=f'SELL Signal ({sell_mask.sum()})', zorder=7)
        
        # Draw lines from points to fitted curve
        for idx in test_data[sell_mask].index:
            m = test_data.loc[idx, 'Moneyness']
            actual_iv = test_data.loc[idx, 'ImpliedVol']
            pred_iv = poly_model(m)
            ax.plot([m, m], [pred_iv, actual_iv], 'r--', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('Moneyness (Strike / Forward Price)', fontsize=13)
    ax.set_ylabel('Implied Volatility', fontsize=13)
    ax.set_title(f'Rolling Window Strategy Snapshot (Index: {snapshot_idx}/{len(rolling_snapshots)-1})\n'
                f'Time: {selected_snapshot["time"]}\n'
                f'Training: {selected_snapshot["train_start"]} to {selected_snapshot["train_end"]}',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print details about this snapshot
    print(f"\n=== Snapshot Details ===")
    print(f"Snapshot Index: {snapshot_idx} / {len(rolling_snapshots)-1}")
    print(f"Time: {selected_snapshot['time']}")
    print(f"Training window: {selected_snapshot['train_start']} to {selected_snapshot['train_end']}")
    print(f"Training data points: {len(train_data)}")
    print(f"Test data points: {len(test_data)}")
    print(f"\nSignals in this window:")
    print(f"  Buy: {buy_mask.sum()}")
    print(f"  Sell: {sell_mask.sum()}")
    print(f"  Hold: {hold_mask.sum()}")
    
    if buy_mask.any():
        print(f"\n📈 Buy Signal Details:")
        print(test_data.loc[buy_mask, ['Strike', 'CallPut', 'Moneyness', 'ImpliedVol', 'PredictedIV', 'Z_Score']])
    
    if sell_mask.any():
        print(f"\n📉 Sell Signal Details:")
        print(test_data.loc[sell_mask, ['Strike', 'CallPut', 'Moneyness', 'ImpliedVol', 'PredictedIV', 'Z_Score']])

