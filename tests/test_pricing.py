import numpy as np
import pytest

from src.models import simulate_gbm_paths
from src.pricing import black_scholes_call, price_european_call_mc


def test_black_scholes_call_reference_value():
    price = black_scholes_call(100, 100, 0.05, 0.02, 0.20, 1)
    assert price == pytest.approx(9.2270055, abs=1e-6)


def test_black_scholes_zero_volatility():
    price = black_scholes_call(100, 100, 0.05, 0.02, 0.0, 1)
    expected = np.exp(-0.05) * (100 * np.exp(0.03) - 100)
    assert price == pytest.approx(expected)


def test_monte_carlo_call_agrees_with_black_scholes():
    paths = simulate_gbm_paths(
        100, 0.05, 0.02, 0.20, 1, 252, 25000,
        np.random.default_rng(42),
    )
    mc_price, standard_error = price_european_call_mc(paths, K=100, r=0.05, T=1)
    exact_price = black_scholes_call(100, 100, 0.05, 0.02, 0.20, 1)
    assert abs(mc_price - exact_price) < 4 * standard_error
