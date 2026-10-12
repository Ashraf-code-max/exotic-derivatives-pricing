"""Finite-difference Greeks for arithmetic Asian calls."""

import math

import numpy as np

from src.asian_options import geometric_asian_call_exact
from src.models import simulate_gbm_paths_from_shocks


GREEKS = ("delta", "gamma", "vega")


def _payoff_components(S0, K, r, q, sigma, T, shocks):
    """Return discounted arithmetic and geometric payoffs for each path."""
    paths = simulate_gbm_paths_from_shocks(S0, r, q, sigma, T, shocks)
    fixing_prices = paths[:, 1:]
    discount = math.exp(-r * T)

    arithmetic_average = np.mean(fixing_prices, axis=1)
    geometric_average = np.exp(np.mean(np.log(fixing_prices), axis=1))

    arithmetic_payoffs = discount * np.maximum(arithmetic_average - K, 0.0)
    geometric_payoffs = discount * np.maximum(geometric_average - K, 0.0)
    geometric_price = geometric_asian_call_exact(
        S0, K, r, q, sigma, T, fixing_prices.shape[1]
    )

    return arithmetic_payoffs, geometric_payoffs, geometric_price


def _greek_payoff_components(
    S0, K, r, q, sigma, T, shocks, spot_bump, vol_bump
):
    """Build paired path-level finite-difference values for each Greek."""
    spot_up = _payoff_components(S0 + spot_bump, K, r, q, sigma, T, shocks)
    spot_mid = _payoff_components(S0, K, r, q, sigma, T, shocks)
    spot_down = _payoff_components(S0 - spot_bump, K, r, q, sigma, T, shocks)

    vol_up = _payoff_components(S0, K, r, q, sigma + vol_bump, T, shocks)
    vol_down = _payoff_components(S0, K, r, q, sigma - vol_bump, T, shocks)

    def difference(up, mid, down, bump, greek):
        if greek == "delta":
            return (up - down) / (2.0 * bump)
        if greek == "gamma":
            return (up - 2.0 * mid + down) / (bump**2)
        return (up - down) / (2.0 * bump)

    arithmetic_values = {
        "delta": difference(spot_up[0], None, spot_down[0], spot_bump, "delta"),
        "gamma": difference(spot_up[0], spot_mid[0], spot_down[0], spot_bump, "gamma"),
        "vega": difference(vol_up[0], None, vol_down[0], vol_bump, "vega"),
    }
    geometric_values = {
        "delta": difference(spot_up[1], None, spot_down[1], spot_bump, "delta"),
        "gamma": difference(spot_up[1], spot_mid[1], spot_down[1], spot_bump, "gamma"),
        "vega": difference(vol_up[1], None, vol_down[1], vol_bump, "vega"),
    }
    exact_geometric_values = {
        "delta": (spot_up[2] - spot_down[2]) / (2.0 * spot_bump),
        "gamma": (spot_up[2] - 2.0 * spot_mid[2] + spot_down[2]) / (spot_bump**2),
        "vega": (vol_up[2] - vol_down[2]) / (2.0 * vol_bump),
    }
    return arithmetic_values, geometric_values, exact_geometric_values


def estimate_geometric_control_betas_for_greeks(
    S0, K, r, q, sigma, T, pilot_shocks,
    spot_bump=0.5, vol_bump=0.01,
):
    """Estimate a separate control coefficient for Delta, Gamma and Vega.

    The pilot shocks must be independent of the shocks used for pricing.
    Each beta is estimated from the matching path-level finite-difference
    Greek, rather than reusing the coefficient for the option price.
    """
    if pilot_shocks.ndim != 2 or pilot_shocks.shape[0] < 2:
        raise ValueError("pilot_shocks must be a 2D array with at least two paths.")

    arithmetic_values, geometric_values, _ = _greek_payoff_components(
        S0, K, r, q, sigma, T, pilot_shocks, spot_bump, vol_bump
    )

    betas = {}
    for greek in GREEKS:
        control_variance = np.var(geometric_values[greek], ddof=1)
        if control_variance <= 0:
            raise ValueError(
                f"Pilot control values have zero variance for {greek}."
            )
        covariance = np.cov(
            arithmetic_values[greek], geometric_values[greek], ddof=1
        )[0, 1]
        betas[greek] = float(covariance / control_variance)

    return betas


def estimate_asian_call_greeks_mc(
    S0, K, r, q, sigma, T, shocks,
    spot_bump=0.5, vol_bump=0.01, betas=None,
):
    """Estimate Delta, Gamma and Vega with central finite differences.

    The same normal shocks are reused for each bumped scenario (common random
    numbers). Set betas=None for plain Monte Carlo. To use control variates,
    pass a dictionary of separate Greek coefficients estimated from an
    independent pilot sample using `estimate_geometric_control_betas_for_greeks`.

    Returns a dictionary in which each Greek maps to (estimate, standard error).
    Vega is per a 1.00 change in decimal volatility, not per volatility point.
    """
    if S0 <= 0 or K <= 0 or T <= 0:
        raise ValueError("S0, K and T must be positive.")
    if sigma <= 0 or vol_bump <= 0 or sigma <= vol_bump:
        raise ValueError("sigma must be greater than a positive vol_bump.")
    if spot_bump <= 0 or spot_bump >= S0:
        raise ValueError("spot_bump must be positive and smaller than S0.")
    if not isinstance(shocks, np.ndarray) or shocks.ndim != 2 or shocks.shape[0] < 2:
        raise ValueError("shocks must be a 2D array with at least two paths.")
    if betas is not None and any(
        greek not in betas or not np.isfinite(betas[greek]) for greek in GREEKS
    ):
        raise ValueError("betas must contain finite delta, gamma and vega values.")

    arithmetic_values, geometric_values, exact_geometric_values = (
        _greek_payoff_components(
            S0, K, r, q, sigma, T, shocks, spot_bump, vol_bump
        )
    )

    estimates = {}
    for greek in GREEKS:
        values = arithmetic_values[greek]
        if betas is not None:
            values = values - betas[greek] * (
                geometric_values[greek] - exact_geometric_values[greek]
            )
        estimates[greek] = (
            float(np.mean(values)),
            float(np.std(values, ddof=1) / np.sqrt(len(values))),
        )

    return estimates
