"""Compare discrete barrier monitoring with Brownian-bridge weighting."""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.barriers import (  # noqa: E402
    price_up_and_out_call_bridge,
    price_up_and_out_call_discrete,
    price_up_and_out_call_exact,
)
from src.models import simulate_gbm_paths  # noqa: E402


S0, K, H = 100.0, 100.0, 120.0
R, Q, SIGMA, T = 0.05, 0.02, 0.20, 1.0
N_PATHS = 15_000
TIME_STEPS = [12, 52, 252]


def run_experiment():
    exact_price = price_up_and_out_call_exact(S0, K, H, R, Q, SIGMA, T)
    output_path = ROOT / "outputs" / "barrier_convergence.csv"
    output_path.parent.mkdir(exist_ok=True)
    rows = []

    for n_steps in TIME_STEPS:
        paths = simulate_gbm_paths(
            S0, R, Q, SIGMA, T, n_steps, N_PATHS,
            np.random.default_rng(20_000 + n_steps),
        )
        discrete_price, discrete_se = price_up_and_out_call_discrete(
            paths, K, H, R, T
        )
        bridge_price, bridge_se = price_up_and_out_call_bridge(
            paths, K, H, R, SIGMA, T
        )
        rows.append({
            "time_steps": n_steps,
            "paths": N_PATHS,
            "analytical_price": exact_price,
            "discrete_price": discrete_price,
            "discrete_standard_error": discrete_se,
            "discrete_absolute_error": abs(discrete_price - exact_price),
            "bridge_price": bridge_price,
            "bridge_standard_error": bridge_se,
            "bridge_absolute_error": abs(bridge_price - exact_price),
        })

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Analytical continuous-barrier price: {exact_price:.6f}")
    print(f"Saved results to {output_path}")
    print("Steps  Discrete price (SE)  Bridge price (SE)  Exact price")
    for row in rows:
        print(
            f"{row['time_steps']:5d}  "
            f"{row['discrete_price']:.6f} ({row['discrete_standard_error']:.5f})  "
            f"{row['bridge_price']:.6f} ({row['bridge_standard_error']:.5f})  "
            f"{row['analytical_price']:.6f}"
        )


if __name__ == "__main__":
    run_experiment()
