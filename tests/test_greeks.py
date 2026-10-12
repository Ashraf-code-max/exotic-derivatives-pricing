import numpy as np

from src.greeks import (
    estimate_asian_call_greeks_mc,
    estimate_geometric_control_betas_for_greeks,
)


def test_greek_estimates_are_finite():
    shocks = np.random.default_rng(321).standard_normal((1200, 52))
    greeks = estimate_asian_call_greeks_mc(
        100, 100, 0.05, 0.02, 0.20, 1, shocks,
        spot_bump=0.5, vol_bump=0.01,
    )
    assert set(greeks) == {"delta", "gamma", "vega"}
    for estimate, standard_error in greeks.values():
        assert np.isfinite(estimate)
        assert np.isfinite(standard_error)
        assert standard_error >= 0


def test_separate_control_coefficients_are_available_for_each_greek():
    pilot_shocks = np.random.default_rng(322).standard_normal((1000, 52))
    pricing_shocks = np.random.default_rng(323).standard_normal((1200, 52))
    betas = estimate_geometric_control_betas_for_greeks(
        100, 100, 0.05, 0.02, 0.20, 1, pilot_shocks
    )
    assert set(betas) == {"delta", "gamma", "vega"}
    for value in betas.values():
        assert np.isfinite(value)

    greeks = estimate_asian_call_greeks_mc(
        100, 100, 0.05, 0.02, 0.20, 1, pricing_shocks, betas=betas
    )
    for estimate, standard_error in greeks.values():
        assert np.isfinite(estimate)
        assert np.isfinite(standard_error)
