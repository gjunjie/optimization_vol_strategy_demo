"""
Main benchmark model execution logic.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

from utils.rates import interpolate_risk_free_rate
from benchmark.black76_model import (
    calculate_implied_volatility,
    calculate_delta,
    calculate_gamma,
    calculate_vega,
    calculate_theta,
    calculate_rho,
)

# Days per year (accounting for leap years)
DAYS_PER_YEAR = 365.25


def calculate_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Implied Volatility and Greeks for the DataFrame using Black-76.

    Processes the DataFrame row-by-row.

    Args:
        df: Pandas DataFrame with options data.
            Must contain: 'VWAP', 'Strike', 'ExpirationDate', 'Date',
            'UnderOpenBidPrice', 'UnderOpenAskPrice', 'UnderCloseBidPrice',
            'UnderCloseAskPrice'

    Returns:
        DataFrame with added columns: 'ImpliedVol', 'Delta', 'Gamma', 'Vega',
        'Theta', 'Rho'
    """
    df_out = df.copy()

    # Initialize result lists
    implied_vols = []
    deltas = []
    gammas = []
    vegas = []
    thetas = []
    rhos = []

    # Iterate row by row
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Calculating Risks"):
        try:
            # Extract data
            price = float(row["VWAP"])
            strike = float(row["Strike"])
            option_type = str(row["CallPut"])

            # Date parsing (assuming integer YYYYMMDD or similar if not datetime)
            date_val = row["Date"]
            expiry_val = row["ExpirationDate"]

            if isinstance(date_val, (int, float, str)):
                date_dt = pd.to_datetime(str(date_val), format="%Y%m%d")
            else:
                date_dt = date_val

            if isinstance(expiry_val, (int, float, str)):
                expiry_dt = pd.to_datetime(str(expiry_val), format="%Y%m%d")
            else:
                expiry_dt = expiry_val

            # Calculate time to expiry in years
            days_to_expiry = (expiry_dt - date_dt).days
            T = days_to_expiry / DAYS_PER_YEAR

            # Calculate Underlying (Forward) Price
            under_open_mid = (row["UnderOpenBidPrice"] + row["UnderOpenAskPrice"]) / 2
            under_close_mid = (
                row["UnderCloseBidPrice"] + row["UnderCloseAskPrice"]
            ) / 2
            F = (under_open_mid + under_close_mid) / 2

            # Interpolate Risk-Free Rate
            r = interpolate_risk_free_rate(days_to_expiry)

            # Calculate Metrics
            if T <= 0:
                iv = np.nan
                delta = np.nan
                gamma = np.nan
                vega = np.nan
                theta = np.nan
                rho = np.nan
            else:
                iv = calculate_implied_volatility(price, F, strike, T, r, option_type)

                if np.isnan(iv):
                    delta = np.nan
                    gamma = np.nan
                    vega = np.nan
                    theta = np.nan
                    rho = np.nan
                else:
                    delta = calculate_delta(F, strike, T, r, iv, option_type)
                    gamma = calculate_gamma(F, strike, T, r, iv, option_type)
                    vega = calculate_vega(F, strike, T, r, iv, option_type)
                    theta = calculate_theta(F, strike, T, r, iv, option_type)
                    rho = calculate_rho(F, strike, T, r, iv, option_type)

            implied_vols.append(iv)
            deltas.append(delta)
            gammas.append(gamma)
            vegas.append(vega)
            thetas.append(theta)
            rhos.append(rho)

        except Exception:
            # Handle errors gracefully for individual rows
            # print(f"Error processing row {index}: {e}")
            implied_vols.append(np.nan)
            deltas.append(np.nan)
            gammas.append(np.nan)
            vegas.append(np.nan)
            thetas.append(np.nan)
            rhos.append(np.nan)

    # Assign new columns
    df_out["ImpliedVol"] = implied_vols
    df_out["Delta"] = deltas
    df_out["Gamma"] = gammas
    df_out["Vega"] = vegas
    df_out["Theta"] = thetas
    df_out["Rho"] = rhos

    return df_out
