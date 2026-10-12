# Exotic Derivatives Pricing

This project is about using Monte Carlo simulation to price options whose payoff depends on the stock price at one or more dates. I use it to compare a few numerical methods, check their accuracy, and see how much the results change when the simulation settings change.

The code is written in Python with NumPy. The examples use a simple risk-neutral model, so the focus is on the pricing methods rather than on building a market-calibrated model.

## What is included

- Simulating stock-price paths under geometric Brownian motion (GBM).
- Pricing European calls with Monte Carlo and comparing the result with the Black–Scholes formula.
- Pricing arithmetic-average and geometric-average Asian calls.
- Using the geometric Asian option, whose price is known under this model, as a control variate for the arithmetic Asian option.
- Estimating Delta, Gamma and Vega for an arithmetic Asian call using finite differences.
- Comparing Greek estimates with and without control variates, while reusing the same random shocks for bumped scenarios.
- Pricing a continuously monitored up-and-out call in three ways: checking the barrier only at simulated dates, using Brownian-bridge weighting between dates, and using an analytical formula as a benchmark.
- Running tests for the main pricing and simulation functions.

## How the experiments work

### 1. Control variates for an Asian call

An arithmetic Asian call pays according to the arithmetic average of the observed stock prices. Its price does not have a simple closed-form expression in this model, so it can be estimated with Monte Carlo simulation.

A geometric Asian call uses the geometric average instead. Under GBM and equally spaced fixing dates, its price can be calculated analytically. The two payoffs are related, so the geometric option can help reduce the noise in the arithmetic-option estimate. The control-variate coefficient is estimated using a separate pilot sample, then applied to independent pricing paths.

The experiment repeats this comparison across several strikes and volatility levels. Both estimators use the same total number of simulated paths, including the pilot sample used to estimate the coefficient. The CSV file records average prices, standard errors and how much the estimates vary across repeated runs. The amount of variance reduction can differ by strike and volatility; it is not guaranteed to be the same in every case.

### 2. Barrier monitoring and Brownian bridge

An up-and-out call becomes worthless if the stock reaches an upper barrier. If the simulation checks the barrier only at a small number of dates, it can miss a crossing that happens between two dates. This can make the estimated price too high.

The Brownian-bridge method uses the two simulated prices at the ends of each time interval to estimate the probability that the stock crossed the barrier between them. The method uses these probabilities to weight each path's payoff. The experiment compares this estimate with ordinary discrete monitoring and with the analytical price for a continuously monitored barrier option, using different numbers of time steps.

### 3. Greeks for an Asian call

Delta measures how the option price changes with the initial stock price, Gamma measures how Delta changes, and Vega measures sensitivity to volatility. Here they are estimated with central finite differences rather than with closed-form formulas.

The same random shocks are used for each bumped scenario. This is called using common random numbers and helps make the differences between estimates less noisy. The experiment also compares plain Monte Carlo with a geometric-Asian control variate. Separate control-variate coefficients are estimated for Delta, Gamma and Vega using pilot shocks that are independent of the pricing shocks.

The results show how the variability of each Greek estimate changes across strikes and volatility levels. Gamma can be especially sensitive to the bump size and simulation noise, so its estimates need to be interpreted with care.

## Project structure

```text
exotic-derivatives-pricing/
├── src/
│   ├── models.py          # GBM path simulation
│   ├── pricing.py         # European call pricing and Black–Scholes benchmark
│   ├── asian_options.py   # Asian option pricing and control variate
│   ├── barriers.py        # Up-and-out barrier pricing methods
│   └── greeks.py          # Finite-difference Greek estimates
├── experiments/
│   ├── variance_reduction.py
│   ├── greeks.py
│   ├── barrier_convergence.py
│   └── make_plots.py
├── tests/                 # Tests for the main functions
├── outputs/               # CSV results and generated plots
├── requirements.txt
└── README.md
```

## Setup

You need Python 3.10 or newer. Open a terminal in the project folder and create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS or Linux**

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Run the tests

```bash
python -m pytest -q
```

The tests check the main simulation and pricing functions, including comparisons with analytical benchmarks and basic behaviour in edge cases.

## Run the experiments

Run these commands from the project folder:

```bash
python experiments/variance_reduction.py
python experiments/greeks.py
python experiments/barrier_convergence.py
python experiments/make_plots.py
```

The first three scripts write CSV results to `outputs/`. The plotting script creates the figures used below. The experiments use fixed random seeds so the same setup can be run again and compared. Small differences may still appear if the code or simulation settings are changed.

## Results and plots

### Variance reduction for the arithmetic Asian call

This plot compares the variation in price estimates from plain Monte Carlo and the control-variate method across different strikes and volatility levels. It is useful for seeing where the control variate helps most, and where the improvement is smaller.

![Variance reduction across strikes and volatility](outputs/variance_reduction.png)

### Barrier option pricing

This plot compares discrete barrier monitoring and Brownian-bridge weighting with the analytical price for a continuously monitored up-and-out call. It shows how the monitoring method and the number of time steps affect the estimate.

![Barrier pricing comparison](outputs/barrier_convergence.png)

### Variability of Asian option Greeks

This plot compares the variability of the Delta, Gamma and Vega estimates with and without the control variate. The effect is different for each Greek, and Gamma can be more sensitive to the numerical settings.

![Greeks comparison](outputs/greek_variability.png)

The CSV files contain the underlying numerical results, so the comparisons do not depend on the plots alone.

## Model assumptions and limitations

- The stock follows risk-neutral geometric Brownian motion with constant volatility, interest rate and dividend yield.
- Asian options use equally spaced fixing dates and exclude the initial stock price from the average.
- The barrier example is an up-and-out call with continuous monitoring and no rebate.
- The Greek estimates use finite-difference bump sizes. Different bumps can change the balance between discretisation error and Monte Carlo noise.
- The experiments cover a limited set of strikes, volatilities, path counts and time steps. They are useful comparisons, not proof that one method is always better.

This is a learning and numerical-testing project, not a production pricing library. The methods used here are established methods. The aim is to implement them carefully, compare their behaviour and document the results, rather than to claim a new pricing algorithm.
