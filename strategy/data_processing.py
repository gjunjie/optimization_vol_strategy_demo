"""
Data processing utilities for volatility arbitrage strategy.
"""
import pandas as pd
import numpy as np
from optimization.optimized_model import calculate_optimized


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV data and calculate optimized values.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing options data
        
    Returns
    -------
    pd.DataFrame
        DataFrame with optimized values
    """
    df = pd.read_csv(csv_path)
    df_optimized = calculate_optimized(df.copy())
    return df_optimized


def calculate_moneyness(df: pd.DataFrame) -> pd.Series:
    """
    Calculate moneyness (Strike / Forward Price) for each option.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing underlying prices and strikes
        
    Returns
    -------
    pd.Series
        Moneyness values
    """
    under_open_mid = (df['UnderOpenBidPrice'] + df['UnderOpenAskPrice']) / 2
    under_close_mid = (df['UnderCloseBidPrice'] + df['UnderCloseAskPrice']) / 2
    forward_price = (under_open_mid + under_close_mid) / 2
    moneyness = df['Strike'] / forward_price
    return moneyness


def get_clean_data_for_plotting(df: pd.DataFrame) -> tuple:
    """
    Filter out NaN values and return clean data for plotting.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Moneyness and ImpliedVol columns
        
    Returns
    -------
    tuple
        (moneyness_clean, iv_clean) - cleaned arrays for plotting
    """
    moneyness = calculate_moneyness(df)
    mask = ~(np.isnan(moneyness) | np.isnan(df['ImpliedVol']))
    moneyness_clean = moneyness[mask]
    iv_clean = df['ImpliedVol'][mask]
    return moneyness_clean, iv_clean


def prepare_rolling_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for rolling window analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with date and time information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with Timestamp and Moneyness columns added
    """
    df_rolling = df.copy()
    df_rolling['Timestamp'] = pd.to_datetime(
        df_rolling['Date'].astype(str) + ' ' + df_rolling['TimeBarStart']
    )
    df_rolling = df_rolling.sort_values('Timestamp')
    df_rolling['Moneyness'] = calculate_moneyness(df_rolling)
    return df_rolling

