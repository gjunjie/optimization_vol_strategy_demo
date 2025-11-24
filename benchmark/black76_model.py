"""
Black-76 Option Pricing Model, Implied Volatility, and Greeks.
"""

import numpy as np
from scipy.stats import norm


IV_TOLERANCE = 1e-5
IV_MAX_ITERATIONS = 100
IV_LOW_BOUND = 0.001
IV_HIGH_BOUND = 5.0


def black76_price(
    F: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> float:
    """
    Calculate Black-76 option price.

    Args:
        F: Forward price of the underlying asset
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Volatility
        option_type: 'C' for Call, 'P' for Put

    Returns:
        Option price
    """
    if T <= 0:
        return max(0, F - K) if option_type == "C" else max(0, K - F)

    d1 = (np.log(F / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    discount_factor = np.exp(-r * T)

    if option_type == "C":
        price = discount_factor * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = discount_factor * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    return price


def calculate_implied_volatility(
    price: float, F: float, K: float, T: float, r: float, option_type: str
) -> float:
    """
    Calculate Implied Volatility using the Bisection Method.

    Args:
        price: Market price of the option
        F: Forward price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free rate
        option_type: 'C' for Call, 'P' for Put

    Returns:
        Implied volatility (sigma)
    """
    # Check for intrinsic value violation (arbitrage condition)
    intrinsic = max(0, F - K) if option_type == "C" else max(0, K - F)
    intrinsic_pv = intrinsic * np.exp(-r * T)
    if price < intrinsic_pv:
        return np.nan  # Arbitrage violation or bad data

    low = IV_LOW_BOUND
    high = IV_HIGH_BOUND

    for _ in range(IV_MAX_ITERATIONS):
        mid = (low + high) / 2
        estimated_price = black76_price(F, K, T, r, mid, option_type)

        if abs(estimated_price - price) < IV_TOLERANCE:
            return mid

        if estimated_price < price:
            low = mid
        else:
            high = mid

    return mid


def calculate_delta(
    F: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> float:
    """Calculate Black-76 Delta."""
    if T <= 0 or sigma <= 0:
        return np.nan

    d1 = (np.log(F / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    discount = np.exp(-r * T)

    if option_type == "C":
        return discount * norm.cdf(d1)
    else:
        return discount * (norm.cdf(d1) - 1)


def calculate_gamma(
    F: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> float:
    """Calculate Black-76 Gamma."""
    if T <= 0 or sigma <= 0:
        return np.nan

    d1 = (np.log(F / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    discount = np.exp(-r * T)

    return (discount * norm.pdf(d1)) / (F * sigma * np.sqrt(T))


def calculate_vega(
    F: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> float:
    """Calculate Black-76 Vega."""
    if T <= 0 or sigma <= 0:
        return np.nan

    d1 = (np.log(F / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    discount = np.exp(-r * T)

    return F * discount * norm.pdf(d1) * np.sqrt(T)


def calculate_theta(
    F: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> float:
    """Calculate Black-76 Theta."""
    if T <= 0 or sigma <= 0:
        return np.nan

    d1 = (np.log(F / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    discount = np.exp(-r * T)

    if option_type == "C":
        return (
            -(F * discount * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            + r * K * discount * norm.cdf(d2)
            - r * F * discount * norm.cdf(d1)
        )
    else:
        return (
            -(F * discount * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            + r * F * discount * norm.cdf(-d1)
            - r * K * discount * norm.cdf(-d2)
        )


def calculate_rho(
    F: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> float:
    """Calculate Black-76 Rho."""
    if T <= 0 or sigma <= 0:
        return np.nan

    price = black76_price(F, K, T, r, sigma, option_type)
    return -T * price
