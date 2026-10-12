"""Pricing methods for continuously monitored up-and-out calls."""

import math
from statistics import NormalDist

import numpy as np


def _validate_barrier_paths(paths, K, H, T):
    if K <= 0 or H <= 0 or T <= 0:
        raise ValueError("K, H and T must be positive.")
    if paths.ndim != 2 or paths.shape[0] < 2 or paths.shape[1] < 2:
        raise ValueError("Provide at least two paths and one time step.")
    if np.any(paths <= 0):
        raise ValueError("All simulated prices must be positive.")


def price_up_and_out_call_discrete(paths, K, H, r, T):
    """Price an up-and-out call, checking the barrier only at observation dates."""
    _validate_barrier_paths(paths, K, H, T)

    terminal_prices = paths[:, -1]
    knocked_out = np.any(paths >= H, axis=1)
    payoffs = np.maximum(terminal_prices - K, 0.0)
    payoffs[knocked_out] = 0.0
    discounted_payoffs = math.exp(-r * T) * payoffs

    return (
        float(np.mean(discounted_payoffs)),
        float(np.std(discounted_payoffs, ddof=1) / np.sqrt(len(discounted_payoffs))),
    )


def price_up_and_out_call_bridge(paths, K, H, r, sigma, T):
    """Price a continuously monitored up-and-out call with bridge weighting.

    Conditional crossing probabilities are calculated in log-price space
    under constant-volatility GBM. This method requires sigma > 0.
    """
    _validate_barrier_paths(paths, K, H, T)
    if sigma <= 0:
        raise ValueError("sigma must be positive for bridge weighting.")

    dt = T / (paths.shape[1] - 1)
    start_prices = paths[:, :-1]
    end_prices = paths[:, 1:]
    below_barrier = (start_prices < H) & (end_prices < H)

    crossing_probability = np.zeros_like(start_prices, dtype=float)
    log_start_distance = np.zeros_like(start_prices, dtype=float)
    log_end_distance = np.zeros_like(end_prices, dtype=float)

    log_start_distance[below_barrier] = np.log(H / start_prices[below_barrier])
    log_end_distance[below_barrier] = np.log(H / end_prices[below_barrier])
    crossing_probability[below_barrier] = np.exp(
        -2.0
        * log_start_distance[below_barrier]
        * log_end_distance[below_barrier]
        / (sigma**2 * dt)
    )

    # The path survives only if it avoids crossings over every interval.
    survival_probability = np.prod(1.0 - crossing_probability, axis=1)
    observed_breach = np.any(paths >= H, axis=1)
    survival_probability[observed_breach] = 0.0

    terminal_payoffs = np.maximum(paths[:, -1] - K, 0.0)
    discounted_payoffs = math.exp(-r * T) * terminal_payoffs * survival_probability

    return (
        float(np.mean(discounted_payoffs)),
        float(np.std(discounted_payoffs, ddof=1) / np.sqrt(len(discounted_payoffs))),
    )


def price_up_and_out_call_exact(S0, K, H, r, q, sigma, T):
    """Analytical price for a continuously monitored up-and-out call.

    Supports S0 < H. If K >= H, the call payoff is always zero while the
    option is alive, so its value is zero. For positive volatility and K < H,
    this uses the standard Black-Scholes barrier formula.
    """
    if S0 <= 0 or K <= 0 or H <= 0 or T <= 0:
        raise ValueError("S0, K, H and T must be positive.")
    if sigma < 0:
        raise ValueError("sigma cannot be negative.")
    if S0 >= H:
        return 0.0  # Already knocked out at inception.
    if K >= H:
        return 0.0

    if sigma == 0:
        # With zero volatility, the path is deterministic and monotone.
        max_price = S0 * math.exp(max(r - q, 0.0) * T)
        if max_price >= H:
            return 0.0
        terminal_price = S0 * math.exp((r - q) * T)
        return math.exp(-r * T) * max(terminal_price - K, 0.0)

    normal_cdf = NormalDist().cdf
    vol_t = sigma * math.sqrt(T)
    mu = (r - q) / (sigma**2) - 0.5
    mu_sigma = (1.0 + mu) * vol_t
    barrier_ratio = H / S0

    # A and B are vanilla-like terms; C and D are reflected-path terms.
    x1 = math.log(S0 / K) / vol_t + mu_sigma
    x2 = math.log(S0 / H) / vol_t + mu_sigma
    y1 = math.log(H**2 / (S0 * K)) / vol_t + mu_sigma
    y2 = math.log(H / S0) / vol_t + mu_sigma

    discount_stock = S0 * math.exp(-q * T)
    discount_strike = K * math.exp(-r * T)
    power_2mu = barrier_ratio ** (2.0 * mu)
    power_2mu_plus_2 = barrier_ratio ** (2.0 * mu + 2.0)

    a = (
        discount_stock * normal_cdf(x1)
        - discount_strike * normal_cdf(x1 - vol_t)
    )
    b = (
        discount_stock * normal_cdf(x2)
        - discount_strike * normal_cdf(x2 - vol_t)
    )
    c = (
        discount_stock * power_2mu_plus_2 * normal_cdf(-y1)
        - discount_strike * power_2mu * normal_cdf(-y1 + vol_t)
    )
    d = (
        discount_stock * power_2mu_plus_2 * normal_cdf(-y2)
        - discount_strike * power_2mu * normal_cdf(-y2 + vol_t)
    )

    price = a - b + c - d
    # Check the basic financial bounds instead of silently hiding a bad result.
    tolerance = 1e-10 * max(1.0, abs(a))
    if price < -tolerance or price > a + tolerance:
        raise ArithmeticError("Barrier formula returned a price outside its bounds.")
    # Clamp only tiny rounding errors near zero or the vanilla-call price.
    return max(0.0, min(a, price))
