import numpy as np
import pytest

from src.asian_options import (
    estimate_geometric_control_beta,
    geometric_asian_call_exact,
    price_arithmetic_asian_call_cv,
    price_arithmetic_asian_call_mc,
    price_geometric_asian_call_mc,
)
from src.models import simulate_gbm_paths


def test_geometric_asian_monte_carlo_agrees_with_exact_price():
    paths = simulate_gbm_paths(
        100, 0.05, 0.02, 0.20, 1, 52, 25000,
        np.random.default_rng(43),
    )
    mc_price, standard_error = price_geometric_asian_call_mc(paths, 100, 0.05, 1)
    exact_price = geometric_asian_call_exact(100, 100, 0.05, 0.02, 0.20, 1, 52)
    assert abs(mc_price - exact_price) < 4 * standard_error


def test_geometric_asian_exact_price_with_zero_volatility():
    price = geometric_asian_call_exact(100, 100, 0.05, 0.02, 0, 1, 12)
    mean_fixing_time = (12 + 1) / (2 * 12)
    expected_average = 100 * np.exp(0.03 * mean_fixing_time)
    expected_price = np.exp(-0.05) * max(expected_average - 100, 0)
    assert price == pytest.approx(expected_price)


def test_control_variate_returns_finite_price_and_error():
    pilot = simulate_gbm_paths(
        100, 0.05, 0.02, 0.20, 1, 52, 3000,
        np.random.default_rng(44),
    )
    pricing = simulate_gbm_paths(
        100, 0.05, 0.02, 0.20, 1, 52, 5000,
        np.random.default_rng(45),
    )
    beta = estimate_geometric_control_beta(pilot, 100, 0.05, 1)
    geometric_price = geometric_asian_call_exact(100, 100, 0.05, 0.02, 0.20, 1, 52)
    price, standard_error = price_arithmetic_asian_call_cv(
        pricing, 100, 0.05, 1, geometric_price, beta
    )
    assert np.isfinite(price)
    assert np.isfinite(standard_error)
    assert standard_error > 0


def test_control_variate_reduces_error_for_fixed_seed_example():
    pilot = simulate_gbm_paths(
        100, 0.05, 0.02, 0.20, 1, 52, 5000,
        np.random.default_rng(46),
    )
    pricing = simulate_gbm_paths(
        100, 0.05, 0.02, 0.20, 1, 52, 8000,
        np.random.default_rng(47),
    )
    beta = estimate_geometric_control_beta(pilot, 100, 0.05, 1)
    geometric_price = geometric_asian_call_exact(100, 100, 0.05, 0.02, 0.20, 1, 52)
    _, plain_se = price_arithmetic_asian_call_mc(pricing, 100, 0.05, 1)
    _, cv_se = price_arithmetic_asian_call_cv(
        pricing, 100, 0.05, 1, geometric_price, beta
    )
    assert cv_se < plain_se
