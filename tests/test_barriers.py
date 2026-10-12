import numpy as np
import pytest

from src.barriers import (
    price_up_and_out_call_bridge,
    price_up_and_out_call_discrete,
    price_up_and_out_call_exact,
)
from src.models import simulate_gbm_paths
from src.pricing import black_scholes_call


def test_analytical_barrier_price_matches_reference_value():
    price = price_up_and_out_call_exact(100, 100, 120, 0.05, 0.02, 0.20, 1)
    assert price == pytest.approx(1.132492141, abs=1e-8)


def test_barrier_price_is_between_zero_and_vanilla_price():
    barrier_price = price_up_and_out_call_exact(100, 100, 120, 0.05, 0.02, 0.20, 1)
    vanilla_price = black_scholes_call(100, 100, 0.05, 0.02, 0.20, 1)
    assert 0 <= barrier_price <= vanilla_price


def test_discrete_method_knocks_out_observed_breach():
    paths = np.array([[100.0, 121.0, 110.0], [100.0, 105.0, 110.0]])
    price, _ = price_up_and_out_call_discrete(paths, K=100, H=120, r=0.05, T=1)
    assert price == pytest.approx(np.exp(-0.05) * 10 / 2)


def test_bridge_price_is_zero_if_every_path_breaches():
    paths = np.array([[100.0, 121.0, 110.0], [100.0, 105.0, 121.0]])
    price, standard_error = price_up_and_out_call_bridge(
        paths, K=100, H=120, r=0.05, sigma=0.20, T=1
    )
    assert price == pytest.approx(0.0)
    assert standard_error == pytest.approx(0.0)


def test_bridge_weighting_does_not_exceed_discrete_price_on_same_paths():
    paths = simulate_gbm_paths(
        100, 0.05, 0.02, 0.20, 1, 52, 10000,
        np.random.default_rng(123),
    )
    discrete_price, _ = price_up_and_out_call_discrete(paths, 100, 120, 0.05, 1)
    bridge_price, _ = price_up_and_out_call_bridge(paths, 100, 120, 0.05, 0.20, 1)
    assert bridge_price <= discrete_price


def test_bridge_estimate_is_consistent_with_analytical_price():
    paths = simulate_gbm_paths(
        100, 0.05, 0.02, 0.20, 1, 52, 30000,
        np.random.default_rng(789),
    )
    bridge_price, standard_error = price_up_and_out_call_bridge(
        paths, 100, 120, 0.05, 0.20, 1
    )
    exact_price = price_up_and_out_call_exact(100, 100, 120, 0.05, 0.02, 0.20, 1)
    assert abs(bridge_price - exact_price) < 4 * standard_error


def test_option_is_zero_if_barrier_is_already_breached_at_inception():
    price = price_up_and_out_call_exact(100, 100, 90, 0.05, 0.02, 0.20, 1)
    assert price == pytest.approx(0.0)


def test_bridge_requires_positive_volatility():
    paths = np.array([[100.0, 101.0, 102.0], [100.0, 103.0, 104.0]])
    with pytest.raises(ValueError):
        price_up_and_out_call_bridge(paths, 100, 120, 0.05, 0.0, 1)
