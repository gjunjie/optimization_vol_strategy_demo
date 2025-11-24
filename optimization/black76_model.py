"""
Black-76 Option Pricing Model, Implied Volatility, and Greeks.
"""

from typing import Union, Dict
import numpy as np
from scipy.special import ndtr

# Constants
IV_TOLERANCE = 1e-5
IV_MAX_ITERATIONS = 100
IV_LOW_BOUND = 0.001
IV_HIGH_BOUND = 5.0
INV_SQRT_2PI = 1.0 / np.sqrt(2 * np.pi)


def _norm_pdf(x):
    """Fast standard normal PDF."""
    return INV_SQRT_2PI * np.exp(-0.5 * x * x)


def _norm_cdf(x):
    """Fast standard normal CDF."""
    return ndtr(x)


def _brenner_subrahmanyam(
    F: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    price: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    option_type: str,
) -> Union[float, np.ndarray]:
    """
    Brenner-Subrahmanyam approximation for implied volatility.
    Simple approximation, best for ATM options.
    """
    # Convert to forward price (future value of option)
    price_fwd = price * np.exp(r * T)

    if option_type == "P":
        # Use Call-Put parity to get equivalent Call price
        price_fwd = price_fwd + (F - K)

    return np.sqrt(2 * np.pi / T) * (price_fwd / F)


def _corrado_miller(
    F: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    price: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    option_type: str,
) -> Union[float, np.ndarray]:
    """
    Corrado-Miller approximation for implied volatility.
    More accurate for OTM/ITM options.
    """
    # Convert to forward price
    price_fwd = price * np.exp(r * T)

    if option_type == "P":
        price_fwd = price_fwd + (F - K)

    diff = F - K
    term1 = price_fwd - diff / 2

    # Safeguard for square root
    radicand = term1**2 - diff**2 / np.pi

    # Vectorized check
    is_scalar = np.isscalar(radicand)
    if is_scalar:
        if radicand < 0:
            return IV_LOW_BOUND
        sqrt_term = np.sqrt(radicand)
    else:
        radicand = np.maximum(0, radicand)
        sqrt_term = np.sqrt(radicand)

    sigma = (np.sqrt(2 * np.pi / T) / (F + K)) * (term1 + sqrt_term)

    if not is_scalar:
        sigma = np.where(radicand <= 0, IV_LOW_BOUND, sigma)

    return sigma


def _get_initial_guess(
    F: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    price: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    option_type: str,
) -> Union[float, np.ndarray]:
    """
    Select best initial guess based on moneyness.
    """
    # Handling scalar/vector inputs
    F = np.asarray(F)
    K = np.asarray(K)
    T = np.asarray(T)

    # Avoid division by zero
    # Vectorized safe divide or mask

    moneyness = np.log(F / K)

    # Use Brenner-Subrahmanyam for ATM (within 10%)
    mask_atm = np.abs(moneyness) < 0.1

    # If all inputs are scalar, np.where works fine too
    bs_guess = _brenner_subrahmanyam(F, K, T, price, r, option_type)
    cm_guess = _corrado_miller(F, K, T, price, r, option_type)

    # If Corrado-Miller failed (returned lower bound), fall back to Brenner-Subrahmanyam
    # This handles cases where CM approximation is invalid (negative radicand)
    cm_failed = cm_guess <= IV_LOW_BOUND + 1e-6  # Check if CM returned the lower bound
    guess = np.where(mask_atm | cm_failed, bs_guess, cm_guess)

    # For deep OTM options, ensure minimum initial guess to avoid Newton-Raphson instability
    # Deep OTM calls (F << K) or deep OTM puts (F >> K) need higher volatility
    # Use a heuristic: if price is significant but guess is very low, use a higher starting point
    price_fwd = np.asarray(price) * np.exp(np.asarray(r) * T)
    # For deep OTM, use a minimum guess based on price relative to forward
    # This is a heuristic to prevent Newton-Raphson from starting too low
    min_guess_otm = np.sqrt(2 * np.pi / T) * (price_fwd / F) * 2.0  # Scale up for OTM
    # Only apply this if the original guess is very low and we have a meaningful price
    mask_low_guess = (guess < 0.1) & (price_fwd > 1e-6)
    guess = np.where(mask_low_guess, np.maximum(guess, min_guess_otm), guess)

    # Clamp result
    return np.clip(guess, IV_LOW_BOUND, IV_HIGH_BOUND)


def black76_price(
    F: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    option_type: str,
) -> Union[float, np.ndarray]:
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
    # Handle scalar/array inputs uniformly
    F = np.asarray(F)
    K = np.asarray(K)
    T = np.asarray(T)
    sigma = np.asarray(sigma)

    # Mask for expired options
    expired = T <= 0
    # Calculate intrinsic value for expired
    if option_type == "C":
        intrinsic = np.maximum(0, F - K)
    else:
        intrinsic = np.maximum(0, K - F)

    # If all expired, return intrinsic
    if np.all(expired):
        return intrinsic

    # Setup arrays
    price = np.zeros_like(F)
    price[expired] = intrinsic[expired]

    # Calculate for non-expired
    mask = ~expired
    if np.any(mask):
        _F = F[mask]
        _K = K[mask]
        _T = T[mask]
        _sigma = sigma[mask]

        sqrt_T = np.sqrt(_T)
        d1 = (np.log(_F / _K) + (0.5 * _sigma**2) * _T) / (_sigma * sqrt_T)
        d2 = d1 - _sigma * sqrt_T

        discount_factor = np.exp(-r * _T) if np.isscalar(r) else np.exp(-r[mask] * _T)

        if option_type == "C":
            p = discount_factor * (_F * _norm_cdf(d1) - _K * _norm_cdf(d2))
        else:
            p = discount_factor * (_K * _norm_cdf(-d2) - _F * _norm_cdf(-d1))

        price[mask] = p

    return price if price.ndim > 0 else price.item()


def _calculate_iv_newton(
    price: np.ndarray,
    F: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    option_type: str,
) -> np.ndarray:
    """
    Vectorized Newton-Raphson solver for IV.
    """
    # Ensure arrays are at least 1D (handle scalars)
    was_scalar = np.isscalar(price)
    price = np.atleast_1d(price)
    F = np.atleast_1d(F)
    K = np.atleast_1d(K)
    T = np.atleast_1d(T)
    r = np.atleast_1d(r)

    # Initial guess
    sigma = _get_initial_guess(F, K, T, price, r, option_type)

    # Check intrinsic violation
    if option_type == "C":
        intrinsic = np.maximum(0, F - K)
    else:
        intrinsic = np.maximum(0, K - F)

    discount = np.exp(-r * T)
    intrinsic_pv = intrinsic * discount
    # Use a small epsilon for float comparison
    invalid = price < (intrinsic_pv - 1e-8)

    # We will iterate on everything, but mask out invalid/converged updates
    converged = np.zeros_like(sigma, dtype=bool)

    for _ in range(IV_MAX_ITERATIONS):
        # Only calculate for non-converged, valid, and non-expired
        mask = ~converged & ~invalid & (T > 0)
        if not np.any(mask):
            break

        # Gather values (using masking)
        _F = F[mask]
        _K = K[mask]
        _T = T[mask]
        # Handle r broadcasting if needed (though usually r is array or scalar)
        _r = r[mask] if r.ndim > 0 else r
        _sigma = sigma[mask]
        _price = price[mask]

        sqrt_T = np.sqrt(_T)
        d1 = (np.log(_F / _K) + (0.5 * _sigma**2) * _T) / (_sigma * sqrt_T)
        d2 = d1 - _sigma * sqrt_T

        _discount = np.exp(-_r * _T)
        pdf_d1 = _norm_pdf(d1)
        vega = _F * _discount * pdf_d1 * sqrt_T

        if option_type == "C":
            model_price = _discount * (_F * _norm_cdf(d1) - _K * _norm_cdf(d2))
        else:
            model_price = _discount * (_K * _norm_cdf(-d2) - _F * _norm_cdf(-d1))

        diff = model_price - _price

        # Check convergence for this subset
        subset_converged = np.abs(diff) < IV_TOLERANCE

        # Newton step: sigma_new = sigma - diff / vega
        vega_safe = np.where(vega < 1e-8, 1e-8, vega)
        step = diff / vega_safe

        # Limit step size to prevent oscillation and instability
        # Maximum step size is 50% of current sigma value to ensure stability
        max_step = np.abs(_sigma) * 0.5
        step = np.clip(step, -max_step, max_step)

        _sigma_new = _sigma - step
        _sigma_new = np.clip(_sigma_new, IV_LOW_BOUND, IV_HIGH_BOUND)

        # Update sigma
        sigma[mask] = _sigma_new

        # Update converged
        converged[mask] = subset_converged

    sigma[invalid] = np.nan
    
    # Return scalar if input was scalar
    if was_scalar:
        return sigma[0]
    return sigma


def calculate_implied_volatility(
    price: Union[float, np.ndarray],
    F: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    option_type: str,
) -> Union[float, np.ndarray]:
    """
    Calculate Implied Volatility using Vectorized Newton-Raphson Method.

    Supports both scalar and vectorized inputs.

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
    # Delegate to vectorized solver which handles scalars via np.asarray
    result = _calculate_iv_newton(price, F, K, T, r, option_type)

    # If input was scalar, return scalar
    if np.isscalar(price) and np.isscalar(F) and np.isscalar(K):
        return result.item()
    return result


def calculate_all_greeks(
    F: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    option_type: str,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate all Black-76 Greeks efficiently by reusing intermediate values.

    Args:
        F: Forward price of the underlying asset
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Volatility
        option_type: 'C' for Call, 'P' for Put

    Returns:
        Dictionary containing delta, gamma, vega, theta, and rho.
    """
    # Determine if we are in scalar or vector mode
    inputs = [F, K, T, r, sigma]
    is_scalar = not any(isinstance(x, (np.ndarray, list)) for x in inputs)

    # Convert to numpy arrays for uniform handling
    F_arr = np.atleast_1d(F)
    K_arr = np.atleast_1d(K)
    T_arr = np.atleast_1d(T)
    r_arr = np.atleast_1d(r)
    sigma_arr = np.atleast_1d(sigma)

    # Broadcast arrays to common shape to ensure alignment
    try:
        b_F, b_K, b_T, b_r, b_sigma = np.broadcast_arrays(
            F_arr, K_arr, T_arr, r_arr, sigma_arr
        )
    except ValueError:
        raise ValueError("Input arrays could not be broadcast together.")

    # Mask for valid inputs (handle NaNs and invalid ranges)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = (b_T > 0) & (b_sigma > 0) & (b_F > 0) & ~np.isnan(b_sigma)

    # Initialize result arrays with NaN
    shape = valid.shape
    delta = np.full(shape, np.nan)
    gamma = np.full(shape, np.nan)
    vega = np.full(shape, np.nan)
    theta = np.full(shape, np.nan)
    rho = np.full(shape, np.nan)

    # If no valid entries, return NaNs (handling scalar return if needed)
    if not np.any(valid):
        if is_scalar:
            return {
                "delta": np.nan,
                "gamma": np.nan,
                "vega": np.nan,
                "theta": np.nan,
                "rho": np.nan,
            }
        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho,
        }

    # Extract valid elements for calculation
    F_v = b_F[valid]
    K_v = b_K[valid]
    T_v = b_T[valid]
    r_v = b_r[valid]
    sigma_v = b_sigma[valid]

    sqrt_T = np.sqrt(T_v)
    d1 = (np.log(F_v / K_v) + (0.5 * sigma_v**2) * T_v) / (sigma_v * sqrt_T)
    d2 = d1 - sigma_v * sqrt_T

    # Pre-calculate common terms
    discount = np.exp(-r_v * T_v)
    pdf_d1 = _norm_pdf(d1)

    # Calculate Greeks
    gamma_v = (discount * pdf_d1) / (F_v * sigma_v * sqrt_T)
    vega_v = F_v * discount * pdf_d1 * sqrt_T

    if option_type == "C":
        cdf_d1 = _norm_cdf(d1)
        cdf_d2 = _norm_cdf(d2)

        delta_v = discount * cdf_d1
        theta_v = (
            -(F_v * discount * pdf_d1 * sigma_v) / (2 * sqrt_T)
            + r_v * K_v * discount * cdf_d2
            - r_v * F_v * discount * cdf_d1
        )
        price_v = discount * (F_v * cdf_d1 - K_v * cdf_d2)
    else:
        cdf_neg_d1 = _norm_cdf(-d1)
        cdf_neg_d2 = _norm_cdf(-d2)

        delta_v = -discount * cdf_neg_d1

        theta_v = (
            -(F_v * discount * pdf_d1 * sigma_v) / (2 * sqrt_T)
            + r_v * F_v * discount * cdf_neg_d1
            - r_v * K_v * discount * cdf_neg_d2
        )
        price_v = discount * (K_v * cdf_neg_d2 - F_v * cdf_neg_d1)

    rho_v = -T_v * price_v

    # Assign computed values back to result arrays
    delta[valid] = delta_v
    gamma[valid] = gamma_v
    vega[valid] = vega_v
    theta[valid] = theta_v
    rho[valid] = rho_v

    # Return scalars if input was scalar
    if is_scalar:
        return {
            "delta": float(delta[0]),
            "gamma": float(gamma[0]),
            "vega": float(vega[0]),
            "theta": float(theta[0]),
            "rho": float(rho[0]),
        }

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
