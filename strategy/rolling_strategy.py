"""
Rolling window strategy for volatility arbitrage.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any


def build_rolling_snapshots(
    df_rolling: pd.DataFrame,
    train_window_minutes: int = 10,
    test_window_minutes: int = 1,
    min_train_points: int = 10,
    moneyness_min: float = 0.3,
    moneyness_max: float = 1.7,
    z_threshold: float = 2.0
) -> List[Dict[str, Any]]:
    """
    Build rolling window snapshots with fitted models and trading signals.
    
    Parameters
    ----------
    df_rolling : pd.DataFrame
        DataFrame with Timestamp, Moneyness, and ImpliedVol columns
    train_window_minutes : int
        Number of minutes for training window
    test_window_minutes : int
        Number of minutes for test window
    min_train_points : int
        Minimum number of training points required
    moneyness_min : float
        Minimum moneyness threshold for signals
    moneyness_max : float
        Maximum moneyness threshold for signals
    z_threshold : float
        Z-score threshold for buy/sell signals
        
    Returns
    -------
    List[Dict]
        List of snapshot dictionaries containing:
        - time: current timestamp
        - train_data: training data
        - test_data: test data with signals
        - poly_model: fitted polynomial model
        - train_start: training window start time
        - train_end: training window end time
    """
    train_window = pd.Timedelta(minutes=train_window_minutes)
    test_window = pd.Timedelta(minutes=test_window_minutes)
    start_time = df_rolling['Timestamp'].min()
    end_time = df_rolling['Timestamp'].max()
    
    rolling_snapshots = []
    current_time = start_time + train_window
    
    print(f"Building rolling window snapshots...")
    
    while current_time < end_time:
        train_start = current_time - train_window
        train_end = current_time
        test_end = current_time + test_window
        
        # Get training and test data
        train_mask = (df_rolling['Timestamp'] >= train_start) & (df_rolling['Timestamp'] < train_end)
        test_mask = (df_rolling['Timestamp'] >= train_end) & (df_rolling['Timestamp'] < test_end)
        
        train_data = df_rolling[train_mask]
        test_data = df_rolling[test_mask].copy()
        
        if len(test_data) == 0:
            current_time += test_window
            continue
        
        # Clean training data
        train_clean_mask = ~(np.isnan(train_data['Moneyness']) | np.isnan(train_data['ImpliedVol']))
        train_clean = train_data[train_clean_mask]
        
        if len(train_clean) < min_train_points:
            current_time += test_window
            continue
        
        try:
            # Fit polynomial
            poly_coeffs = np.polyfit(train_clean['Moneyness'], train_clean['ImpliedVol'], 3)
            poly_model = np.poly1d(poly_coeffs)
            
            # Predict on test data
            test_data['PredictedIV'] = poly_model(test_data['Moneyness'])
            test_data['Residual'] = test_data['ImpliedVol'] - test_data['PredictedIV']
            
            # Calculate Z-scores
            train_preds = poly_model(train_clean['Moneyness'])
            train_residuals = train_clean['ImpliedVol'] - train_preds
            train_mean = train_residuals.mean()
            train_std = train_residuals.std()
            
            if train_std > 1e-6:
                test_data['Z_Score'] = (test_data['Residual'] - train_mean) / train_std
            else:
                test_data['Z_Score'] = 0
            
            # Determine signals
            test_data['Signal'] = 'Hold'
            test_data.loc[test_data['Z_Score'] <= -z_threshold, 'Signal'] = 'Buy'
            test_data.loc[test_data['Z_Score'] >= z_threshold, 'Signal'] = 'Sell'
            
            # Apply moneyness threshold to signal points only
            outside_moneyness = (test_data['Moneyness'] < moneyness_min) | (test_data['Moneyness'] > moneyness_max)
            test_data.loc[outside_moneyness, 'Signal'] = 'Hold'
            
            # Save snapshot
            rolling_snapshots.append({
                'time': current_time,
                'train_data': train_clean.copy(),
                'test_data': test_data.copy(),
                'poly_model': poly_model,
                'train_start': train_start,
                'train_end': train_end
            })
            
        except Exception as e:
            pass
        
        current_time += test_window
    
    print(f"Created {len(rolling_snapshots)} rolling window snapshots")
    
    if len(rolling_snapshots) > 0:
        # Count signals
        total_buy = sum(len(s['test_data'][s['test_data']['Signal'] == 'Buy']) for s in rolling_snapshots)
        total_sell = sum(len(s['test_data'][s['test_data']['Signal'] == 'Sell']) for s in rolling_snapshots)
        print(f"Total Buy signals: {total_buy}")
        print(f"Total Sell signals: {total_sell}")
    
    return rolling_snapshots


def filter_snapshots_by_signal(rolling_snapshots: List[Dict]) -> Dict[str, List[int]]:
    """
    Create indices for snapshots with different signal types.
    
    Parameters
    ----------
    rolling_snapshots : List[Dict]
        List of snapshot dictionaries
        
    Returns
    -------
    Dict[str, List[int]]
        Dictionary with keys: 'buy', 'sell', 'both', 'any'
        Each containing list of snapshot indices
    """
    snapshots_with_buy = [i for i, s in enumerate(rolling_snapshots) 
                          if len(s['test_data'][s['test_data']['Signal'] == 'Buy']) > 0]
    snapshots_with_sell = [i for i, s in enumerate(rolling_snapshots) 
                           if len(s['test_data'][s['test_data']['Signal'] == 'Sell']) > 0]
    snapshots_with_both = [i for i, s in enumerate(rolling_snapshots) 
                           if (len(s['test_data'][s['test_data']['Signal'] == 'Buy']) > 0 and
                               len(s['test_data'][s['test_data']['Signal'] == 'Sell']) > 0)]
    snapshots_with_any = [i for i, s in enumerate(rolling_snapshots) 
                          if len(s['test_data'][s['test_data']['Signal'] != 'Hold']) > 0]
    
    print(f"Snapshots with Buy signals: {len(snapshots_with_buy)}")
    print(f"Snapshots with Sell signals: {len(snapshots_with_sell)}")
    print(f"Snapshots with Both signals: {len(snapshots_with_both)}")
    print(f"Snapshots with Any signals: {len(snapshots_with_any)}")
    
    return {
        'buy': snapshots_with_buy,
        'sell': snapshots_with_sell,
        'both': snapshots_with_both,
        'any': snapshots_with_any
    }

