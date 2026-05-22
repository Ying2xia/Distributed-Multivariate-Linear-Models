# DMLM-S Paper Code

This directory contains a runnable Python implementation for the paper
`Communication-Efficient Distributed Inference and Diagnostics for Multivariate Linear Models`.

The core implementation has only two required dependencies: `numpy` and `pandas`.
Plotting is optional and uses `matplotlib` when it is installed.

## What Is Implemented

- Full-data MLM estimator (`GMLM`)
- One-shot local averaging estimator (`AMLM`)
- Pilot-only estimator (`MLM-S`)
- First-worker-only estimator (`MLM-F`)
- Proposed pilot-subsampling distributed surrogate estimator (`DMLM-S`)
- Random, covariate-sorted, and heterogeneous-worker storage strategies
- Simulation experiments 1-6 from the paper
- Observation-level and worker-level local influence diagnostics
- Real-data entry points for Beijing air quality and SARCOS-style data

## Quick Smoke Run

```bash
python3 scripts/run_simulations.py --experiment all --quick --replications 3
```

Outputs are written to `results/`.

## Paper-Scale Runs

The default settings now match the paper-scale configuration and can be expensive
because they use large `N` and many Monte Carlo replications. For example:

```bash
python3 scripts/run_simulations.py --experiment 1 --replications 200
python3 scripts/run_simulations.py --experiment 2 --replications 200
python3 scripts/run_simulations.py --experiment 6 --replications 200
```

The default simulation setting is the challenging non-random storage setting used
to emphasize the advantage of DMLM-S:

- `N = 50000`
- `K = 200`
- `p = 50`
- `m = 10`
- `rho_X = 0.7`
- `rho_Y = 0.6`
- pilot proportion `rho_r = 0.02`
- DMLM-S refinement rounds `T = 3`
- Strategy C sorts observations by the latent multivariate linear signal
  `(X_i B_0) 1_m / sqrt(m)` before assigning consecutive blocks to workers.
- Strategy H uses stronger worker heterogeneity:
  `mu_k = 3 * (k - (K - 1)/2)/(K - 1) * 1_p` and
  `rho_{X,k}` ranges linearly from `0.05` to `0.95`.

## Real Data

If both datasets are already available under the default local paths, run all
Chapter 5 real-data analyses with:

```bash
python3 scripts/run_real_data.py
```

Beijing data can be supplied as a single CSV file or a directory of station CSV files:

```bash
python3 scripts/run_real_data.py beijing --path data/beijing
```

Or download it automatically from the UCI Machine Learning Repository:

```bash
python3 scripts/run_real_data.py beijing --download
```

By default this runs `rho_r in {0.002, 0.005, 0.01, 0.02}` with three DMLM-S
refinement rounds, and writes:

- `beijing_performance_grid.csv`
- `beijing_main_results.csv`
- `beijing_pilot_sensitivity.csv`
- `beijing_observation_influence.csv`
- `beijing_worker_influence.csv`

SARCOS data can be supplied as train/test CSV or whitespace-delimited files with
21 input columns followed by 7 output columns:

```bash
python3 scripts/run_real_data.py sarcos --train data/sarcos_inv.csv --test data/sarcos_inv_test.csv
```

Or download the official GPML MATLAB files and convert them to CSV:

```bash
python3 scripts/run_real_data.py sarcos --download
```

The SARCOS auto-conversion requires `scipy`:

```bash
python3 -m pip install scipy
```

By default this runs storage schemes `random,motion`, `K in {20, 50}`,
`rho_r in {0.005, 0.01, 0.02}`, and three DMLM-S refinement rounds. It writes:

- `sarcos_performance_grid.csv`
- `sarcos_main_results.csv`
- `sarcos_pilot_sensitivity.csv`
- `sarcos_motion_worker_influence.csv`

## Optional Figures

After generating CSV results:

```bash
python3 scripts/make_plots.py --results-dir results --figures-dir /Users/chenxiaoxia/Desktop/DMLM论文/figures/ch4
python3 scripts/make_ch5_plots.py --results-dir results --figures-dir /Users/chenxiaoxia/Desktop/DMLM论文/figures/ch5
```

If `matplotlib` is not installed, the script will print a clear message and exit.

The reported `Time` column is an estimated parallel wall-clock time for distributed
estimators. Local worker computations are counted by the slowest worker on each
parallel step, while centralized pilot and coordinator computations are kept in
full. The serial single-process implementation time is stored in each estimator's
`extra` dictionary for diagnostics.
