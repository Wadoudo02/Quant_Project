# Equity Factor Alpha Engine

This repository implements a **morally compliant**
equity‐factor research pipeline inspired by my
`Crypto Price Anomaly` project.  The goal is to provide a
clean, extensible framework for exploring fundamental and
technical factors on equities and training an
eXtreme Gradient Boosting (XGBoost) model to forecast
future returns.  A lightweight back‑tester converts
factor forecasts into a simple long‑only strategy and reports
key performance statistics.

## Motivation

After building an end‑to‑end crypto anomaly detector I
wanted to apply a similar structured approach to
equity markets.  Factor investing is conceptually close to the
dimensional analysis I used in my Physics days: define
inputs (size, value, momentum, volatility), derive
transformations (rolling means, standard deviations, z‑scores)
and observe how the system evolves.  An XGBoost regressor
acts as the “alpha engine” by finding patterns in
multi‑dimensional data without sacrificing interpretability.

This repository is configured via a single `params.yaml`
file, uses local caching to avoid unnecessary network
requests, and exposes every step (loading, feature
engineering, modelling, backtesting) as a Python function.
Comments throughout the code explain *why* certain
decisions were made, both from a software engineering
perspective and through the lens of my own quantitative
experience (bridging academic physics, applied data science
and finance).  Wherever possible I keep the code
morally compliant – long‑only and avoiding leverage –
in line with ethical investing principles.

## Quick‑start

```bash
# Create a virtual environment and install dependencies (no obscure packages)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the end‑to‑end driver on the default tickers
python run_efa.py --show-plots --save-plots

# Inspect the generated equity curve
open outputs/equity_combined.png
```

## Project structure

```
Equity_Factor_Alpha/
├─ README.md               # this file
├─ params.yaml             # configuration for assets, features and model
├─ requirements.txt        # minimal dependencies
├─ run_efa.py              # one‑click driver
├─ src/
│  ├─ __init__.py
│  ├─ config.py            # config loader (mirrors my crypto project)
│  ├─ data_loader.py       # download/cache equity price data
│  ├─ features.py          # factor engineering helpers
│  ├─ model.py             # XGBoost training and inference
│  ├─ backtest.py          # simple long‑only back‑tester
│  ├─ metrics.py           # Sharpe, drawdown, etc. (borrowed and reused)
│  └─ plotting.py          # equity curve plotting
├─ data_raw/               # local cache of downloaded price data
├─ data_proc/              # intermediate processed data (optional)
├─ outputs/
│  └─ figures/             # saved charts
└─ logs/                   # log files for debugging
```

## Morally compliant investing

The back‑tester only takes long positions and sizes trades
equally.  There is no leverage or shorting, and fees are
modeled as proportional basis points per trade.  This
structure mirrors the simple, physically inspired
experiments I ran during my particle physics research: apply
an impulse, observe the response, and always stay within
ethical boundaries.

## Future extensions

- Integrate fundamental data (e.g. price‑to‑earnings, book‑to‑market) once a
  suitable API is available.
- Implement cross‑sectional ranking (top‑decile vs bottom‑decile) and
  multi‑period holding strategies.
- Expose a Jupyter notebook for interactive experimentation and
  hyper‑parameter tuning.
