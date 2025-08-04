# Crypto Price Anomaly Engine

![CI](https://github.com/Wadoudo02/Quant_Project/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)

A research pipeline that tests whether sudden positive return shocks in **BTC and ETH** keep running for a few more bars, then converts those shocks into a simple, long-only strategy.
Everything is driven by a single `params.yaml` file—change the assets, bar interval, thresholds *or even the quote currency* in one place and the rest of the code adapts automatically.

---

## ✨ Key Features
| Feature | Why it matters |
|---------|----------------|
| **Config-first design** | `params.yaml` controls assets, dates, Z-score thresholds, holding period and fees so experiments stay reproducible. |
| **Disk-cached ccxt data** | The first run downloads Binance OHLCV bars; subsequent runs read local parquet files for instant replays. |
| **Z-score anomaly engine** | Log-returns are standardised into Z-scores so only statistically significant shocks trigger trades. |
| **Lightweight long-only back-tester** | Vectorised PnL, Sharpe and drawdown metrics with a flat notional per trade and transparent cost model. |
| **Machine-learning option** | Plug-in XGBoost classifier filters trades and keeps the framework model-agnostic. |

---

## 🔗 Links to Past Projects

### My Master’s Project — Effective Field Theory in Higgs Boson Measurements
This crypto-anomaly engine borrows heavily from the **data-pipeline discipline** I refined during my Imperial master’s dissertation.
Back then we *initially* planned to deploy **XGBoost** for Higgs-event classification but pivoted to a fully-connected **Parametric neural network** once richer non-linear interactions proved essential.
That journey—from feature engineering to model selection and interpretability—directly informs how features are curated and explained here.

### My KAUST Project — Speed Enhancements of the ARFF Algorithm
During the KAUST VSRP I re-implemented the Adaptive Random Fourier Features (ARFF) algorithm in **JAX/CUDA**, achieving a 115 × training speed-up.
The same focus on *efficient, hardware-aware numerics* appears in this repository through vectorised Pandas, parquet caching and a lean back-test that completes in seconds.

### Other Projects (quick-fire)
- **Neutrino Oscillation Data Minimiser** — Negative-log-likelihood optimisation for T2K simulations; mirrors the statistical fitting used in the Sharpe and drawdown estimators.
- **Binary Star Systems Stability (UROP)** — REBOUND N-Body simulations of chaotic stellar orbits; the discrete-time integrator mindset translates to the event-driven back-tester used here.
- **Medical Image Registration @ The Christie Hospital** — High-dimensional optimisation and image preprocessing, experiences that carry over to feature scaling and dimensionality-reduction in financial datasets.

---

## 🗂️ Repository Structure

```
Crypto_Price_Anomaly_Live/
├─ README.md            # <— you are here
├─ params.yaml          # central configuration (assets, Z-score, costs)
├─ run_mvp.py           # one-click driver / CLI entry-point
├─ src/
│  ├─ __init__.py
│  ├─ config.py         # load + validate params.yaml
│  ├─ data_loader.py    # download & cache OHLCV via ccxt
│  ├─ features.py       # log-return and rolling μ/σ
│  ├─ signal_logic.py   # Z-score threshold → boolean signals
│  ├─ backtest.py       # long-only forward-hold simulator
│  ├─ metrics.py        # Sharpe & max drawdown
│  ├─ plotting.py       # consistent matplotlib plots
│  ├─ model.py          # machine-learning wrapper (XGBoost etc.)
│  └─ resample.py       # optional OHLCV resampling helper
├─ data_raw/            # raw exchange pulls
├─ outputs/
│  └─ Figures/          # auto-saved charts
├─ ML_comparison.py     # compare Sharpe & equity curves (ML vs baseline)
└─ notebooks/           # exploratory analysis
```

---

## 🚀 Quick-Start

```bash
# 1) Set up environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Edit params.yaml — at minimum pick:
#    • assets (e.g. BTC/USDT, ETH/USDT)
#    • start_date
#    • zscore.threshold

# 3) Run the pipeline and view the equity curve
python run_mvp.py --show-plots
open outputs/Figures/combined_equity.png
```

Optional flags
```
--assets BTC/USDT            # run a single symbol
--save-plots                 # write all figures to outputs/Figures
--show-plots                 # display plots using matplotlib only (without saving)
--use-ML False               # disable the XGBoost classifier
```

Compare machine-learning vs. baseline performance:

```bash
python ML_comparison.py --show-plot --save-plot
# or use --show-plot / --save-plot individually
```

---

## 🔄 How the Pieces Fit Together

1. **Load config** — `config.py` reads `params.yaml` and exposes settings via `cfg()`.
2. **Download data** — `data_loader.py` pulls 30‑minute OHLCV bars from Binance and caches them on disk.
3. **Engineer features** — `features.py` adds log‑returns, rolling mean/std and Z‑scores.
4. **Generate signals** — `signal_logic.py` flags bars whose Z-score crosses the threshold.
5. **Back‑test** — `backtest.py` enters next-bar open, holds for `hold_bars` and subtracts fees.
5. **Evaluate** — `metrics.py` reports Sharpe, hit‑rate and drawdown.
6. **Report** — `plotting.py` saves publication-quality PNGs of individual and combined equity curves.

---

## Why Machine Learning?

The original anomaly engine relied solely on Z‑scores to flag outsized return
shocks.  While effective, Z‑scores capture only linear deviations from the
recent mean.  The optional XGBoost classifier augments this by learning
non‑linear interactions between engineered features.  In practice this means
potentially filtering out false positives and adapting to subtle regime shifts
without rewriting the strategy logic.  The model lives in ``src/model.py`` and
is configured entirely via ``params.yaml`` so future architectures can slot in
with minimal code changes.

---

## 🛠 Module Reference

| File | Purpose |
|------|---------|
| **`run_mvp.py`** | CLI glue script—parses flags, loads config, calls each stage in turn. |
| **`config.py`** | Loads `params.yaml`, performs path resolution, exposes settings via `cfg()` or `cfg_ns()`. |
| **`data_loader.py`** | Wraps **ccxt** Binance feeds with a local parquet cache and paginated fetch loop. |
| **`features.py`** | Adds log returns, rolling statistics and Z‑score columns. |
| **`signal_logic.py`** | Builds entry signals by comparing Z‑scores against a configurable threshold. |
| **`backtest.py`** | Converts signals into positions, applies proportional trading costs and returns trade logs & equity. |
| **`metrics.py`** | Computes Sharpe ratio and maximum drawdown. |
| **`plotting.py`** | Centralised matplotlib style; function for equity curve charts. |
| **`model.py`** | Wraps XGBoost (and future models) for signal classification. |
| **`resample.py`** | Optional helper to aggregate OHLCV to coarser bar intervals. |

---

## ☝️ Long-Only, No Investment Advice
The default strategy is long-only with flat sizing and no leverage. This repository is for research and education—**not** investment advice.

---

## 📊 Performance Metrics
After each run, the console prints something like:

```
BTC/USDT metrics: {'total_net_pnl': 123.4, 'num_trades': 30, 'hit_rate': 0.60, 'avg_trade_ret': 0.004, 'max_drawdown_abs': 20.1, 'max_drawdown_pct': 0.02, 'sharpe': 4.1}
```

This summarises cumulative PnL, trade statistics and risk-adjusted return (Sharpe).

---

## 📊 Results

### Performance snapshot – 1 Jun 2025 → 22 Jul 2025

| Metric | Value |
|--------|-------|
| **Total net PnL** | **$ 297.26** |
| **Annualised Sharpe** | **5.68** |
| **Max drawdown** | **$ 41.00 (198 %)** |
| **Trades executed** | 56 |
| **Hit‑rate** | 64.3 % |
| **Average trade return** | 0.53 % |

<small>All figures are quoted in the native quote currency (USDT by default).</small>

### Visuals

<p align="center">
  <img src="outputs/Figures/btc_equity.png" alt="BTC Equity Curve" width="60%">
</p>

<p align="center">
  <img src="outputs/Figures/eth_equity.png" alt="ETH Equity Curve" width="60%">
</p>

<p align="center">
  <img src="outputs/Figures/combined_equity.png" alt="Combined Equity Curve" width="75%">
</p>

### Re‑producing these numbers

```bash
python run_mvp.py --save-plots --assets BTC/USDT,ETH/USDT
```

<details>
<summary>Example console output</summary>

```text
2025-07-22 14:31:00 | INFO | [binance] loading BTC/USDT with interval=30m (cache: data_raw/BTCUSDT_30m.parquet)
...
BTC/USDT metrics: {'total_net_pnl': 180.9, 'num_trades': 28, 'hit_rate': 0.61, 'avg_trade_ret': 0.0048, 'max_drawdown_abs': 25.0, 'max_drawdown_pct': 0.15, 'sharpe': 4.7}
ETH/USDT metrics: {'total_net_pnl': 116.3, 'num_trades': 28, 'hit_rate': 0.68, 'avg_trade_ret': 0.0058, 'max_drawdown_abs': 16.0, 'max_drawdown_pct': 0.05, 'sharpe': 5.1}
Finished – results saved to outputs/
```
</details>

### Interpreting the numbers

* **Sharpe above 5** suggests the anomaly is persistent over this window, albeit on a small sample size.
* **Drawdowns remain shallow** thanks to the flat position sizing and six-hour holding period.
* **Hit-rate > 60 %** indicates the edge comes from both directionality and sizing, not just bet allocation.

### Real‑world analogy
Think of each Z‑score spike as a particle collision: we observe an energetic impulse and track how the system relaxes over the next few hours. Just as in accelerator physics, clean data acquisition and precise error bars matter more than fancy models.

---

## comparison between Ml and non-ML

*To be added.*

---

## 🛤 Future Roadmap
- Live data feed and order execution prototype
- Web-based dashboard for monitoring open positions
- Parameter sweep over Z-score thresholds and holding periods
- Better risk controls (volatility scaling, concurrent positions)

