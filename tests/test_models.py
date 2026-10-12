import numpy as np
import pytest

from src.models import simulate_gbm_paths, simulate_gbm_paths_from_shocks


def test_path_shape_and_initial_price():
    paths = simulate_gbm_paths(
        100, 0.05, 0.02, 0.20, 1, 252, 1000,
        np.random.default_rng(42),
    )
    assert paths.shape == (1000, 253)
    assert np.all(paths[:, 0] == 100)


def test_prices_are_positive():
    paths = simulate_gbm_paths(
        100, 0.05, 0.02, 0.20, 1, 252, 1000,
        np.random.default_rng(42),
    )
    assert np.all(paths > 0)


def test_zero_volatility_follows_deterministic_path():
    S0, r, q, T, n_steps = 100, 0.05, 0.02, 1, 252
    paths = simulate_gbm_paths(
        S0, r, q, 0.0, T, n_steps, 3, np.random.default_rng(42)
    )
    times = np.linspace(0, T, n_steps + 1)
    expected = S0 * np.exp((r - q) * times)
    assert np.allclose(paths, expected[None, :])


def test_same_seed_gives_same_paths():
    paths_1 = simulate_gbm_paths(
        100, 0.05, 0.02, 0.20, 1, 50, 100,
        np.random.default_rng(42),
    )
    paths_2 = simulate_gbm_paths(
        100, 0.05, 0.02, 0.20, 1, 50, 100,
        np.random.default_rng(42),
    )
    assert np.array_equal(paths_1, paths_2)


def test_simulation_from_supplied_shocks_is_reproducible():
    shocks = np.random.default_rng(12).standard_normal((25, 12))
    paths_1 = simulate_gbm_paths_from_shocks(100, 0.05, 0.02, 0.2, 1, shocks)
    paths_2 = simulate_gbm_paths_from_shocks(100, 0.05, 0.02, 0.2, 1, shocks)
    assert np.array_equal(paths_1, paths_2)


def test_invalid_simulation_inputs_raise_errors():
    with pytest.raises(ValueError):
        simulate_gbm_paths(0, 0.05, 0.02, 0.2, 1, 10, 10, np.random.default_rng(1))
    with pytest.raises(ValueError):
        simulate_gbm_paths(100, 0.05, 0.02, 0.2, 1, 0, 10, np.random.default_rng(1))
