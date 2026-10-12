"""Create simple plots from the CSV files produced by the experiments."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def read_csv(filename):
    with (OUTPUTS / filename).open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def plot_variance_reduction():
    rows = read_csv("variance_reduction.csv")
    by_volatility = defaultdict(list)
    for row in rows:
        by_volatility[float(row["volatility"])].append(row)

    fig, ax = plt.subplots(figsize=(8, 5))
    for volatility, group in sorted(by_volatility.items()):
        group.sort(key=lambda item: float(item["strike"]))
        ax.plot(
            [float(row["strike"]) for row in group],
            [float(row["variance_reduction_percent"]) for row in group],
            marker="o",
            label=f"Volatility {volatility:.0%}",
        )

    ax.set_xlabel("Strike")
    ax.set_ylabel("Observed variance reduction (%)")
    ax.set_title("Geometric control variate across strikes and volatility")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS / "variance_reduction.png", dpi=160)
    plt.close(fig)


def plot_barrier_convergence():
    rows = read_csv("barrier_convergence.csv")
    steps = [int(row["time_steps"]) for row in rows]
    discrete = [float(row["discrete_price"]) for row in rows]
    bridge = [float(row["bridge_price"]) for row in rows]
    exact = float(rows[0]["analytical_price"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, discrete, marker="o", label="Discrete monitoring")
    ax.plot(steps, bridge, marker="o", label="Brownian-bridge weighting")
    ax.axhline(exact, linestyle="--", label=f"Analytical price ({exact:.4f})")
    ax.set_xlabel("Number of time steps")
    ax.set_ylabel("Option price")
    ax.set_title("Up-and-out call: effect of monitoring resolution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS / "barrier_convergence.png", dpi=160)
    plt.close(fig)


def plot_greek_variability():
    rows = read_csv("asian_greeks.csv")
    rows = [row for row in rows if float(row["strike"]) == 100.0]
    greeks = ["delta", "gamma", "vega"]
    volatilities = sorted({float(row["volatility"]) for row in rows})

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, greek in zip(axes, greeks):
        group = [row for row in rows if row["greek"] == greek]
        group.sort(key=lambda item: float(item["volatility"]))
        ax.plot(
            [float(row["volatility"]) for row in group],
            [float(row["sd_improvement"]) for row in group],
            marker="o",
        )
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_title(greek.capitalize())
        ax.set_xlabel("Volatility")
        ax.set_ylabel("Plain SD / control-variate SD")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Control-variate effect on Asian-call Greeks (K = 100)")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "greek_variability.png", dpi=160)
    plt.close(fig)


def main():
    needed = [
        OUTPUTS / "variance_reduction.csv",
        OUTPUTS / "barrier_convergence.csv",
        OUTPUTS / "asian_greeks.csv",
    ]
    missing = [path.name for path in needed if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run the experiment scripts first; missing: " + ", ".join(missing)
        )

    plot_variance_reduction()
    plot_barrier_convergence()
    plot_greek_variability()
    print("Saved variance_reduction.png, barrier_convergence.png, and greek_variability.png")


if __name__ == "__main__":
    main()
