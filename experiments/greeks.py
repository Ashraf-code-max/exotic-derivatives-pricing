"""Study arithmetic Asian-call Greeks with and without a control variate."""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.greeks import (  # noqa: E402
    estimate_asian_call_greeks_mc,
    estimate_geometric_control_betas_for_greeks,
)


S0, R, Q, T = 100.0, 0.05, 0.02, 1.0
N_STEPS = 252
PILOT_SIZE = 500
PRICING_SIZE = 1200
N_REPEATS = 8
STRIKES = [90.0, 100.0, 110.0]
VOLATILITIES = [0.15, 0.20, 0.30]


def run_experiment():
    output_path = ROOT / "outputs" / "asian_greeks.csv"
    output_path.parent.mkdir(exist_ok=True)
    rows = []

    for sigma in VOLATILITIES:
        for strike in STRIKES:
            estimates = {
                "plain": {g: [] for g in ("delta", "gamma", "vega")},
                "cv": {g: [] for g in ("delta", "gamma", "vega")},
            }
            standard_errors = {
                "plain": {g: [] for g in ("delta", "gamma", "vega")},
                "cv": {g: [] for g in ("delta", "gamma", "vega")},
            }

            for repeat in range(N_REPEATS):
                seed = 700_000 + repeat * 10_000 + int(strike * 10) + int(sigma * 1000)
                pilot_rng = np.random.default_rng(seed)
                pricing_rng = np.random.default_rng(seed + 1)
                pilot_shocks = pilot_rng.standard_normal((PILOT_SIZE, N_STEPS))
                pricing_shocks = pricing_rng.standard_normal((PRICING_SIZE, N_STEPS))

                greek_betas = estimate_geometric_control_betas_for_greeks(
                    S0, strike, R, Q, sigma, T, pilot_shocks,
                    spot_bump=0.5, vol_bump=0.01,
                )

                plain_results = estimate_asian_call_greeks_mc(
                    S0, strike, R, Q, sigma, T, pricing_shocks,
                    spot_bump=0.5, vol_bump=0.01, betas=None,
                )
                cv_results = estimate_asian_call_greeks_mc(
                    S0, strike, R, Q, sigma, T, pricing_shocks,
                    spot_bump=0.5, vol_bump=0.01, betas=greek_betas,
                )

                for method, result in (("plain", plain_results), ("cv", cv_results)):
                    for greek, (estimate, se) in result.items():
                        estimates[method][greek].append(estimate)
                        standard_errors[method][greek].append(se)

            for greek in ("delta", "gamma", "vega"):
                plain_sd = float(np.std(estimates["plain"][greek], ddof=1))
                cv_sd = float(np.std(estimates["cv"][greek], ddof=1))
                rows.append({
                    "strike": strike,
                    "volatility": sigma,
                    "greek": greek,
                    "repetitions": N_REPEATS,
                    "pilot_paths": PILOT_SIZE,
                    "pricing_paths": PRICING_SIZE,
                    "plain_mean_estimate": float(np.mean(estimates["plain"][greek])),
                    "cv_mean_estimate": float(np.mean(estimates["cv"][greek])),
                    "plain_sd_across_runs": plain_sd,
                    "cv_sd_across_runs": cv_sd,
                    "sd_improvement": plain_sd / cv_sd if cv_sd > 0 else np.nan,
                    "mean_plain_within_run_se": float(np.mean(standard_errors["plain"][greek])),
                    "mean_cv_within_run_se": float(np.mean(standard_errors["cv"][greek])),
                })

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} Greek results to {output_path}")
    print("Strike Volatility Greek Plain mean CV mean Plain SD CV SD SD improvement")
    for row in rows:
        print(
            f"{row['strike']:6.1f} {row['volatility']:10.2f} {row['greek']:>5s} "
            f"{row['plain_mean_estimate']:10.5f} {row['cv_mean_estimate']:9.5f} "
            f"{row['plain_sd_across_runs']:8.5f} {row['cv_sd_across_runs']:7.5f} "
            f"{row['sd_improvement']:15.2f}x"
        )


if __name__ == "__main__":
    run_experiment()
