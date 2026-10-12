"""Compare plain Monte Carlo with an Asian-option control variate.

The comparison uses the same total number of simulated paths for both methods.
The plain estimator uses both the pilot and pricing paths; the control-variate
estimator uses the pilot paths to estimate beta and the independent pricing
paths to estimate the option price.
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asian_options import (  # noqa: E402
    estimate_geometric_control_beta,
    geometric_asian_call_exact,
    price_arithmetic_asian_call_cv,
    price_arithmetic_asian_call_mc,
)
from src.models import simulate_gbm_paths  # noqa: E402


S0, R, Q, T = 100.0, 0.05, 0.02, 1.0
N_STEPS = 252
PILOT_SIZE = 500
PRICING_SIZE = 1500
N_REPEATS = 8
STRIKES = [80.0, 90.0, 100.0, 110.0, 120.0]
VOLATILITIES = [0.15, 0.20, 0.30, 0.40]


def run_experiment():
    output_path = ROOT / "outputs" / "variance_reduction.csv"
    output_path.parent.mkdir(exist_ok=True)
    rows = []

    for sigma in VOLATILITIES:
        for strike in STRIKES:
            geometric_price = geometric_asian_call_exact(
                S0, strike, R, Q, sigma, T, N_STEPS
            )
            plain_estimates = []
            cv_estimates = []
            beta_values = []
            plain_ses = []
            cv_ses = []

            for repeat in range(N_REPEATS):
                base_seed = 100_000 + repeat * 10_000 + int(strike * 10) + int(sigma * 1000)
                pilot_paths = simulate_gbm_paths(
                    S0, R, Q, sigma, T, N_STEPS, PILOT_SIZE,
                    np.random.default_rng(base_seed),
                )
                pricing_paths = simulate_gbm_paths(
                    S0, R, Q, sigma, T, N_STEPS, PRICING_SIZE,
                    np.random.default_rng(base_seed + 1),
                )

                beta = estimate_geometric_control_beta(
                    pilot_paths, strike, R, T
                )
                # Same total path budget for both estimators.
                all_plain_paths = np.vstack((pilot_paths, pricing_paths))
                plain_price, plain_se = price_arithmetic_asian_call_mc(
                    all_plain_paths, strike, R, T
                )
                cv_price, cv_se = price_arithmetic_asian_call_cv(
                    pricing_paths, strike, R, T, geometric_price, beta
                )

                plain_estimates.append(plain_price)
                cv_estimates.append(cv_price)
                beta_values.append(beta)
                plain_ses.append(plain_se)
                cv_ses.append(cv_se)

            plain_sd = float(np.std(plain_estimates, ddof=1))
            cv_sd = float(np.std(cv_estimates, ddof=1))
            rows.append({
                "strike": strike,
                "volatility": sigma,
                "repetitions": N_REPEATS,
                "pilot_paths": PILOT_SIZE,
                "pricing_paths": PRICING_SIZE,
                "total_path_budget_each_method": PILOT_SIZE + PRICING_SIZE,
                "plain_mean_price": float(np.mean(plain_estimates)),
                "control_variate_mean_price": float(np.mean(cv_estimates)),
                "plain_sd_across_runs": plain_sd,
                "control_variate_sd_across_runs": cv_sd,
                "standard_deviation_improvement": plain_sd / cv_sd if cv_sd > 0 else np.nan,
                "variance_reduction_percent": 100.0 * (1.0 - (cv_sd / plain_sd) ** 2) if plain_sd > 0 else np.nan,
                "mean_plain_within_run_se": float(np.mean(plain_ses)),
                "mean_cv_within_run_se": float(np.mean(cv_ses)),
                "mean_pilot_beta": float(np.mean(beta_values)),
            })

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} scenarios to {output_path}")
    print("Strike  Volatility  Plain SD  CV SD  SD improvement  Variance reduction")
    for row in rows:
        print(
            f"{row['strike']:6.1f}  {row['volatility']:10.2f}  "
            f"{row['plain_sd_across_runs']:8.5f}  "
            f"{row['control_variate_sd_across_runs']:6.5f}  "
            f"{row['standard_deviation_improvement']:14.2f}x  "
            f"{row['variance_reduction_percent']:17.2f}%"
        )


if __name__ == "__main__":
    run_experiment()
