"""
Optimized model execution logic.
"""

import pandas as pd
import numpy as np
from optimization.black76_model import (
    calculate_implied_volatility,
    calculate_all_greeks,
)

# Days per year (accounting for leap years)
DAYS_PER_YEAR = 365.25


def calculate_optimized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Implied Volatility and Greeks for the DataFrame using Black-76
    with vectorized operations for performance.

    Args:
        df: Pandas DataFrame with options data.
            Must contain: 'VWAP', 'Strike', 'ExpirationDate', 'Date',
            'UnderOpenBidPrice', 'UnderOpenAskPrice', 'UnderCloseBidPrice',
            'UnderCloseAskPrice', 'CallPut'

    Returns:
        DataFrame with added columns: 'ImpliedVol', 'Delta', 'Gamma', 'Vega',
        'Theta', 'Rho'
    """
    df_out = df.copy()

    # Pre-allocate result arrays
    n = len(df_out)
    iv_arr = np.full(n, np.nan)
    delta_arr = np.full(n, np.nan)
    gamma_arr = np.full(n, np.nan)
    vega_arr = np.full(n, np.nan)
    theta_arr = np.full(n, np.nan)
    rho_arr = np.full(n, np.nan)

    # 1. Date Processing & Time to Expiry
    if not np.issubdtype(df_out["Date"].dtype, np.datetime64):
        dates = pd.to_datetime(df_out["Date"], format="%Y%m%d", errors="coerce")
    else:
        dates = df_out["Date"]

    if not np.issubdtype(df_out["ExpirationDate"].dtype, np.datetime64):
        expiries = pd.to_datetime(
            df_out["ExpirationDate"], format="%Y%m%d", errors="coerce"
        )
    else:
        expiries = df_out["ExpirationDate"]

    days_to_expiry = (expiries - dates).dt.days.values
    # Handle potential NaNs if dates were invalid
    days_to_expiry = np.nan_to_num(days_to_expiry, nan=0.0)
    T = days_to_expiry / DAYS_PER_YEAR

    # 2. Underlying Forward Price (F)
    under_open_mid = (
        df_out["UnderOpenBidPrice"].values + df_out["UnderOpenAskPrice"].values
    ) / 2
    under_close_mid = (
        df_out["UnderCloseBidPrice"].values + df_out["UnderCloseAskPrice"].values
    ) / 2
    F = (under_open_mid + under_close_mid) / 2

    # 3. Strike (K) & Price
    K = df_out["Strike"].values.astype(float)
    price = df_out["VWAP"].values.astype(float)

    # 4. Risk-Free Rate (r) - Cached Interpolation
    # Pre-calculate for unique days_to_expiry values
    # Match benchmark's edge case handling: clamp to boundaries
    curve_days = np.array([0, 30, 90, 180, 365, 730])
    curve_rates = np.array([0.039, 0.039, 0.038, 0.036, 0.035, 0.033])
    
    # Handle edge cases like benchmark: clamp to boundaries
    def interpolate_rate(days):
        if days <= curve_days[0]:
            return curve_rates[0]
        if days >= curve_days[-1]:
            return curve_rates[-1]
        return np.interp(days, curve_days, curve_rates)
    
    unique_days = np.unique(days_to_expiry)
    r_cache = {days: interpolate_rate(days) for days in unique_days}
    r = np.array([r_cache[days] for days in days_to_expiry])

    # 5. Calculation Mask (T > 0)
    mask_valid = T > 0

    # 6. Optimize CallPut Processing - Ultra-fast numpy-based approach
    # 0 for Call, 1 for Put
    # Convert to numpy array once and use simple character comparison (no regex, no np.char overhead)
    callput_arr = df_out["CallPut"].values.astype("U1")  # Unicode string array, single char
    # Check for 'P' or 'p' directly using vectorized comparison (fastest approach, no string ops)
    option_type_int = ((callput_arr == "P") | (callput_arr == "p")).astype(np.int8)

    # Refine masks with validity
    mask_calls = mask_valid & (option_type_int == 0)
    mask_puts = mask_valid & (option_type_int == 1)

    # 7. Process Calls
    if np.any(mask_calls):
        call_iv = calculate_implied_volatility(
            price[mask_calls],
            F[mask_calls],
            K[mask_calls],
            T[mask_calls],
            r[mask_calls],
            "C",
        )

        call_greeks = calculate_all_greeks(
            F[mask_calls],
            K[mask_calls],
            T[mask_calls],
            r[mask_calls],
            call_iv,
            "C",
        )

        # Assign to result arrays
        iv_arr[mask_calls] = call_iv
        delta_arr[mask_calls] = call_greeks["delta"]
        gamma_arr[mask_calls] = call_greeks["gamma"]
        vega_arr[mask_calls] = call_greeks["vega"]
        theta_arr[mask_calls] = call_greeks["theta"]
        rho_arr[mask_calls] = call_greeks["rho"]

    # 8. Process Puts
    if np.any(mask_puts):
        put_iv = calculate_implied_volatility(
            price[mask_puts],
            F[mask_puts],
            K[mask_puts],
            T[mask_puts],
            r[mask_puts],
            "P",
        )

        put_greeks = calculate_all_greeks(
            F[mask_puts],
            K[mask_puts],
            T[mask_puts],
            r[mask_puts],
            put_iv,
            "P",
        )

        # Assign to result arrays
        iv_arr[mask_puts] = put_iv
        delta_arr[mask_puts] = put_greeks["delta"]
        gamma_arr[mask_puts] = put_greeks["gamma"]
        vega_arr[mask_puts] = put_greeks["vega"]
        theta_arr[mask_puts] = put_greeks["theta"]
        rho_arr[mask_puts] = put_greeks["rho"]

    # Assign results back to DataFrame
    df_out["ImpliedVol"] = iv_arr
    df_out["Delta"] = delta_arr
    df_out["Gamma"] = gamma_arr
    df_out["Vega"] = vega_arr
    df_out["Theta"] = theta_arr
    df_out["Rho"] = rho_arr

    return df_out
