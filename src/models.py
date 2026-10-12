"""Stock-price simulation models."""

import numpy as np


def simulate_gbm_paths_from_shocks(S0, r, q, sigma, T, shocks):
    """Build GBM paths from a supplied array of standard normal shocks.

    Supplying the same shocks for different parameters is useful when
    comparing estimates, for example when calculating Greeks by finite
    differences.
    """
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if sigma < 0:
        raise ValueError("sigma cannot be negative.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if not isinstance(shocks, np.ndarray) or shocks.ndim != 2:
        raise ValueError("shocks must be a two-dimensional NumPy array.")
    if shocks.shape[0] <= 0 or shocks.shape[1] <= 0:
        raise ValueError("shocks must contain paths and time steps.")
    if not np.all(np.isfinite(shocks)):
        raise ValueError("shocks must contain only finite values.")

    n_steps = shocks.shape[1]
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * shocks
    log_returns = drift + diffusion

    paths = np.empty((shocks.shape[0], n_steps + 1), dtype=float)
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(np.cumsum(log_returns, axis=1))
    return paths


def simulate_gbm_paths(S0, r, q, sigma, T, n_steps, n_paths, rng):
    """Simulate risk-neutral geometric Brownian motion paths.

    Returns an array with shape (n_paths, n_steps + 1). The first column
    contains the initial stock price, and the last column contains terminal
    prices at maturity.
    """
    if not isinstance(n_steps, (int, np.integer)) or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer.")
    if not isinstance(n_paths, (int, np.integer)) or n_paths <= 0:
        raise ValueError("n_paths must be a positive integer.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")

    shocks = rng.standard_normal((n_paths, n_steps))
    return simulate_gbm_paths_from_shocks(S0, r, q, sigma, T, shocks)
