#!/usr/bin/env python
"""
run_efa.py – One‑click driver for the Equity Factor Alpha Engine
---------------------------------------------------------------

This script orchestrates the full pipeline from data acquisition
through feature engineering, model training, prediction and
back‑testing.  It parallels my crypto anomaly runner but targets
equities and leverages XGBoost to learn factor relationships.

Workflow
~~~~~~~~
1. **Config** – reads ``params.yaml`` via :func:`src.config.cfg`.
2. **Data acquisition** – :func:`src.data_loader.get_all_equities` pulls
   OHLCV data from Stooq, caching results on disk.
3. **Feature engineering** – :func:`src.features.add_returns` and
   :func:`src.features.add_rolling_features` compute returns and
   factor exposures.
4. **Prepare matrix** – :func:`src.features.prepare_feature_matrix`
   combines per‑ticker DataFrames into a single feature matrix X and
   target vector y.
5. **Model training** – :func:`src.model.train_model` fits an
   XGBoost regressor.
6. **Prediction** – :func:`src.model.predict` generates out‑of‑sample
   forecasts on the same feature matrix (for demonstration).
7. **Back‑test** – :func:`src.backtest.run_backtest` converts
   predictions into a simple long‑only strategy.
8. **Visualisation** – :func:`src.plotting.plot_equity` plots the
   cumulative PnL.
9. **Persistence** – dumps ``trades.csv`` and ``equity.csv`` to
   ``outputs/``.

Comments throughout the code reference design decisions and tie them
back to my previous work.  Explanations are written in a neutral
technical tone to assist other researchers who might pick up this
repository.  When a comment includes a first‑person example it draws
from my own experiences captured in my CV and professional posts.  For
instance, having built a high‑frequency trading system early in my
career, I learned the importance of clearly separating data loading,
feature engineering, modelling and back‑testing.  That perspective
informs the modular structure you see here.  The project also
emphasises morally compliant practices; we avoid short selling and
opaque metrics, and we source data from reputable providers.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import cfg
from src.data_loader import get_all_equities
from src.features import add_returns, add_rolling_features, prepare_feature_matrix
from src.model import train_model, predict
from src.backtest import run_backtest
from src.plotting import plot_equity, plot_returns_vs_benchmark

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    # Load configuration
    p = cfg()
    # Map ISO currency code to common symbol for plotting
    _CURR_MAP = {"GBP": "£", "USD": "$", "EUR": "€", "JPY": "¥"}
    base_currency = p.get("base_currency", "GBP").upper()
    currency_symbol = _CURR_MAP.get(base_currency, base_currency)
    selected_assets = [a["ticker"] for a in p["assets"]]
    print(selected_assets)
    if args.assets.lower() != "all":
        wanted = {s.strip().upper() for s in args.assets.split(",")}
        selected_assets = [t for t in selected_assets if t.upper() in wanted]
        if not selected_assets:
            raise ValueError(f"No matching tickers found for: {wanted}")

    start = p.get("start_date")
    end = p.get("end_date")

    # Fetch data for each ticker
    logger.info("Loading price data for: %s", ", ".join(selected_assets))
    data = get_all_equities(force_refresh=args.refresh, start_date=start, end_date=end)
    data = {t: df for t, df in data.items() if t in selected_assets}
    if not data:
        raise RuntimeError(
            "No data available for the selected tickers.  Check params.yaml or network connectivity."
        )

    """
    # Simulated fundamentals for screening (replace with real loader later)
    fundamentals = pd.DataFrame(
        {
            "debt_to_equity": [0.1, 0.5, 0.2, 0.4, 0.25],
            "cash_to_equity": [0.2, 0.6, 0.3, 0.1, 0.1],
            "non_compliant_income_pct": [0.01, 0.1, 0.04, 0.03, 0.02],
        },
        index=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    )

    
    # Apply moral screen
    screened = moral_screen(fundamentals)
    screened_tickers = set(screened.index)
    logger.info("Tickers after moral screen: %s", sorted(screened_tickers))

    # Filter price data to only those passing the screen
    data = {t: df for t, df in data.items() if t in screened_tickers}
    if not data:
        raise RuntimeError(
            "No tickers passed the moral screen. Check your ratios or data."
        )
    """
    # Feature engineering
    logger.info("Computing returns and rolling features…")
    processed: dict[str, pd.DataFrame] = {}
    for ticker, df in data.items():
        # Compute returns
        df = add_returns(df, price_col="close", out_col=p["features"]["return_col"])
        # Drop the first row because return is NaN
        df = df.dropna(subset=[p["features"]["return_col"]])
        # Add rolling features
        df = add_rolling_features(
            df, col=p["features"]["return_col"], windows=p["features"]["windows"]
        )
        processed[ticker] = df

    # Prepare feature matrix and target
    logger.info("Preparing feature matrix…")
    X, y = prepare_feature_matrix(
        processed,
        windows=p["features"]["windows"],
        lookahead=1,
        return_col=p["features"]["return_col"],
    )

    # Equal‑weight average return across all tickers serves as a benchmark
    benchmark = y.groupby(level=0).mean()

    # Train model
    logger.info("Training XGBoost model…")
    model, rmse = train_model(X, y, params=p["model"], test_size=0.2)
    logger.info("Validation RMSE: %.6f", rmse)

    # Predict on entire data set (for demonstration).  In a real
    # experiment you'd generate predictions on an out‑of‑sample period.
    logger.info("Generating predictions…")
    preds = predict(model, X)

    # Align returns for back‑test: we need the realised returns over the
    # lookahead period.  y already contains next‑day returns aligned to X.
    logger.info("Running back‑test…")
    results = run_backtest(
        predictions=preds,
        returns=y,
        benchmark_returns=benchmark,
        risk_free_rate=p["backtest"].get("risk_free_rate", 0.0),
        hold_days=p["backtest"]["hold_days"],
        top_n=p["backtest"]["top_n"],
        notional=p["backtest"]["notional"],
        fee_bps=p["backtest"]["fee_bps"],
    )

    equity = results["equity"]
    trades = results["trades"]
    metrics = results["metrics"]
    initial_capital = results.get("initial_capital")
    logger.info("Back‑test metrics: %s", metrics)
    logger.info(
        "Alpha %.4f (t=%.2f) | Beta %.2f | Sharpe %.2f",
        metrics["alpha"],
        metrics["alpha_tstat"],
        metrics["beta"],
        metrics["sharpe"],
    )

    # Save results
    out_dir = Path(p["paths"]["outputs"])
    out_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_dir / "trades.csv", index=False)
    equity.to_csv(out_dir / "equity.csv")

    # Plot equity curve
    if args.show_plots or args.save_plots:
        save_path = (out_dir / "equity_combined.png") if args.save_plots else None
        plot_equity(
            equity,
            title="Equity Curve – Combined Portfolio",
            show=args.show_plots,
            save_path=save_path,
            metrics=metrics,
            initial_capital=initial_capital,
            currency_symbol=currency_symbol,
            scale=args.plot_scale,
        )
    if args.show_plots or args.save_plots:
        scatter_path = (out_dir / "alpha_beta_scatter.png") if args.save_plots else None
        plot_returns_vs_benchmark(
            equity.diff().dropna(),
            benchmark.reindex(equity.index).fillna(0.0),
            alpha=metrics["alpha"],
            beta=metrics["beta"],
            show=args.show_plots,
            save_path=scatter_path,
        )
    logger.info("Finished – results saved to %s", out_dir.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Equity Factor Alpha Engine end‑to‑end."
    )
    parser.add_argument(
        "--assets",
        default="all",
        help="Comma‑separated list of tickers (e.g. 'AAPL,MSFT') or 'all' (default).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re‑download of price data instead of using cache.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display equity curve plot using Matplotlib.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save equity curve plot to the outputs directory (PNG).",
    )
    parser.add_argument(
        "--plot-scale",
        choices=["currency", "pct"],
        default="pct",
        help="Choose 'currency' to plot portfolio value in £ starting at the initial capital, or 'pct' for percentage return (default).",
    )
    main(parser.parse_args())
