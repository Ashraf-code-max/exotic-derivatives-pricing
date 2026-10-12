"""European option pricing methods."""

import math
from statistics import NormalDist

import numpy as np


def price_european_call_mc(paths, K, r, T):
    """Estimate a European call price and its Monte Carlo standard error."""
    if K <= 0 or T <= 0:
        raise ValueError("K and T must be positive.")
    if paths.ndim != 2 or paths.shape[0] < 2 or paths.shape[1] < 2:
        raise ValueError("Provide at least two paths and one time step.")
    if np.any(paths <= 0):
        raise ValueError("All simulated prices must be positive.")

    discounted_payoffs = np.exp(-r * T) * np.maximum(paths[:, -1] - K, 0.0)
    price = float(np.mean(discounted_payoffs))
    standard_error = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(len(discounted_payoffs)))
    return price, standard_error


def black_scholes_call(S0, K, r, q, sigma, T):
    """Calculate a European call price using the Black-Scholes formula."""
    if S0 <= 0 or K <= 0 or T <= 0:
        raise ValueError("S0, K and T must be positive.")
    if sigma < 0:
        raise ValueError("sigma cannot be negative.")

    if sigma == 0:
        terminal_price = S0 * math.exp((r - q) * T)
        return math.exp(-r * T) * max(terminal_price - K, 0.0)

    normal_cdf = NormalDist().cdf
    vol_t = sigma * math.sqrt(T)
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / vol_t
    d2 = d1 - vol_t

    return (
        S0 * math.exp(-q * T) * normal_cdf(d1)
        - K * math.exp(-r * T) * normal_cdf(d2)
    )
