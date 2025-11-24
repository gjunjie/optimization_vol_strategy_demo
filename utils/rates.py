"""
Risk-free rate interpolation module.
"""

import numpy as np


def interpolate_risk_free_rate(days_to_expiry: float) -> float:
    """
    Interpolate risk-free rate based on days to expiration.

    Uses a simple yield curve assumption for demonstration:
    - 1 day: 3.9%
    - 30 days: 3.9%
    - 90 days: 3.8%
    - 180 days: 3.6%
    - 365 days: 3.5%
    - 730 days: 3.3%

    Args:
        days_to_expiry: Number of days to expiration.

    Returns:
        Interpolated risk-free rate (annualized, e.g., 0.02 for 2%).
    """
    # Defined yield curve points (days, rate)
    curve_days = np.array([0, 30, 90, 180, 365, 730])
    curve_rates = np.array([0.039, 0.039, 0.038, 0.036, 0.035, 0.033])

    # Handle edge cases
    if days_to_expiry <= curve_days[0]:
        return curve_rates[0]
    if days_to_expiry >= curve_days[-1]:
        return curve_rates[-1]

    # Linear interpolation
    return float(np.interp(days_to_expiry, curve_days, curve_rates))
