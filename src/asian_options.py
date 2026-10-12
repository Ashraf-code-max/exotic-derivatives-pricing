"""Monte Carlo pricing for geometric and arithmetic Asian calls."""

import math
from statistics import NormalDist

import numpy as np


def _validate_paths(paths):
    if paths.ndim != 2 or paths.shape[0] < 2 or paths.shape[1] < 2:
        raise ValueError("Provide at least two paths and one fixing date.")
    if np.any(paths <= 0):
        raise ValueError("All simulated prices must be positive.")


def _price_and_se(discounted_payoffs):
    price = float(np.mean(discounted_payoffs))
    standard_error = float(
        np.std(discounted_payoffs, ddof=1) / np.sqrt(len(discounted_payoffs))
    )
    return price, standard_error


def price_geometric_asian_call_mc(paths, K, r, T):
    """Estimate a discretely monitored geometric Asian call price."""
    if K <= 0 or T <= 0:
        raise ValueError("K and T must be positive.")
    _validate_paths(paths)

    fixing_prices = paths[:, 1:]
    geometric_average = np.exp(np.mean(np.log(fixing_prices), axis=1))
    payoffs = np.exp(-r * T) * np.maximum(geometric_average - K, 0.0)
    return _price_and_se(payoffs)


def geometric_asian_call_exact(S0, K, r, q, sigma, T, n_fixings):
    """Exact price for a geometric Asian call with equally spaced fixings.

    The fixing dates are T/n, 2T/n, ..., T. This is an analytical benchmark
    under risk-neutral geometric Brownian motion.
    """
    if S0 <= 0 or K <= 0 or T <= 0:
        raise ValueError("S0, K and T must be positive.")
    if sigma < 0:
        raise ValueError("sigma cannot be negative.")
    if not isinstance(n_fixings, (int, np.integer)) or n_fixings <= 0:
        raise ValueError("n_fixings must be a positive integer.")

    mean_fixing_time = T * (n_fixings + 1) / (2 * n_fixings)
    if sigma == 0:
        average_price = S0 * math.exp((r - q) * mean_fixing_time)
        return math.exp(-r * T) * max(average_price - K, 0.0)

    mean_log_average = (
        math.log(S0)
        + (r - q - 0.5 * sigma**2) * mean_fixing_time
    )
    variance_log_average = (
        sigma**2 * T * (n_fixings + 1) * (2 * n_fixings + 1)
        / (6 * n_fixings**2)
    )
    std_dev = math.sqrt(variance_log_average)
    d2 = (mean_log_average - math.log(K)) / std_dev
    d1 = d2 + std_dev
    normal_cdf = NormalDist().cdf

    return math.exp(-r * T) * (
        math.exp(mean_log_average + 0.5 * variance_log_average) * normal_cdf(d1)
        - K * normal_cdf(d2)
    )


def price_arithmetic_asian_call_mc(paths, K, r, T):
    """Estimate an arithmetic Asian call price and standard error."""
    if K <= 0 or T <= 0:
        raise ValueError("K and T must be positive.")
    _validate_paths(paths)

    arithmetic_average = np.mean(paths[:, 1:], axis=1)
    payoffs = np.exp(-r * T) * np.maximum(arithmetic_average - K, 0.0)
    return _price_and_se(payoffs)


def estimate_geometric_control_beta(pilot_paths, K, r, T):
    """Estimate the control-variate coefficient using independent pilot paths."""
    if K <= 0 or T <= 0:
        raise ValueError("K and T must be positive.")
    _validate_paths(pilot_paths)

    fixing_prices = pilot_paths[:, 1:]
    arithmetic_average = np.mean(fixing_prices, axis=1)
    geometric_average = np.exp(np.mean(np.log(fixing_prices), axis=1))
    discount = math.exp(-r * T)

    arithmetic_payoffs = discount * np.maximum(arithmetic_average - K, 0.0)
    geometric_payoffs = discount * np.maximum(geometric_average - K, 0.0)
    geometric_variance = np.var(geometric_payoffs, ddof=1)

    if geometric_variance <= 0:
        raise ValueError("The pilot control payoff has zero variance.")

    covariance = np.cov(arithmetic_payoffs, geometric_payoffs, ddof=1)[0, 1]
    return float(covariance / geometric_variance)


def price_arithmetic_asian_call_cv(paths, K, r, T, geometric_price, beta):
    """Estimate an arithmetic Asian call using a fixed control-variate beta.

    The beta should be estimated on a separate pilot sample. `geometric_price`
    must be the analytical price for the same fixing dates and contract terms.
    """
    if K <= 0 or T <= 0:
        raise ValueError("K and T must be positive.")
    if not np.isfinite(beta) or not np.isfinite(geometric_price):
        raise ValueError("beta and geometric_price must be finite.")
    _validate_paths(paths)

    fixing_prices = paths[:, 1:]
    arithmetic_average = np.mean(fixing_prices, axis=1)
    geometric_average = np.exp(np.mean(np.log(fixing_prices), axis=1))
    discount = math.exp(-r * T)

    arithmetic_payoffs = discount * np.maximum(arithmetic_average - K, 0.0)
    geometric_payoffs = discount * np.maximum(geometric_average - K, 0.0)
    adjusted_payoffs = arithmetic_payoffs - beta * (geometric_payoffs - geometric_price)
    return _price_and_se(adjusted_payoffs)
