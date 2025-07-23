#!/usr/bin/env python
"""
run_mvp.py – One‑click driver for the Crypto Price Anomaly MVP
-------------------------------------------------------------

This script stitches together every layer of the repo and hands off all graphing
responsibilities to **src.plotting.plot_equity** so plotting style remains
consistent across notebooks, reports, and the CLI.

Pipeline
~~~~~~~~
1. Config – reads params.yaml via src.config.Config (single source of truth)
2. Data acquisition – src.data_loader.get_asset_df
3. Feature engineering – log‑returns, rolling μ/σ, Z‑scores
4. Signal generation – src.signal_logic.signal_from_zscore
5. Back‑test – src.backtest.run_backtest
6. Visual diagnostics – delegates to src.plotting.plot_equity
7. Persistence – dumps trades.csv and equity_curve.csv to outputs/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# --- Project modules -------------------------------------------------------
from src.config import cfg as load_cfg  # cfg() returns the params.yaml dict
from src.data_loader import get_asset_df
from src.features import add_log_return, add_rolling_stats, add_zscore
from src.signal_logic import signal_from_zscore
from src.backtest import run_backtest
from src.plotting import plot_equity

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------


def process_asset(asset_cfg: dict, cfg: dict) -> Tuple[pd.DataFrame, dict]:
    """Run the full pipeline for a single asset symbol."""
    sym: str = asset_cfg["symbol"]
    exc: str = asset_cfg["exchange"]
    interval: str = cfg["bar_interval"]

    lookback: int = cfg["zscore"]["lookback_bars"]
    thresh: float = cfg["zscore"]["threshold"]
    mode: str = cfg["mode"]
    hold_bars: int = cfg["hold_bars"]
    fee_bps: float = cfg["fees"]["taker_bps"]

    # ---------------------------------------------------------------
    # Pull global start/end dates from params.yaml (if provided)
    start_date_raw = cfg.get("start_date")
    end_date_raw = cfg.get("end_date")

    start_dt = pd.to_datetime(start_date_raw, utc=True) if start_date_raw else None
    end_dt = pd.to_datetime(end_date_raw, utc=True) if end_date_raw else None
    # ---------------------------------------------------------------

    paths_cfg = cfg.get("paths", {})
    cache_dir = Path(paths_cfg.get("data_raw", "data_raw"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{sym.replace('/', '')}_{interval}.parquet"

    logger.info(
        "[%-4s] loading %s with interval=%s (cache: %s)", exc, sym, interval, cache_path
    )
    df: pd.DataFrame = get_asset_df(
        sym,
        exchange_name=exc,
        timeframe=interval,
        start_dt=start_dt,
        end_dt=end_dt,
    )

    # Trim any out‑of‑window rows the loader might have returned
    if start_dt is not None:
        df = df[df.index >= start_dt]
    if end_dt is not None:
        df = df[df.index <= end_dt]

    # If trimming wiped everything, try a cache refresh once
    if df.empty:
        logger.warning(
            "Cached data for %s does not cover %s → %s. Refreshing cache and retrying…",
            sym,
            start_dt.date() if start_dt else "beginning",
            end_dt.date() if end_dt else "today",
        )
        if cache_path.exists():
            cache_path.unlink()  # drop stale cache
        df = get_asset_df(
            sym,
            exchange_name=exc,
            timeframe=interval,
            start_dt=start_dt,
            end_dt=end_dt,
        )
        if start_dt is not None:
            df = df[df.index >= start_dt]
        if end_dt is not None:
            df = df[df.index <= end_dt]

    if df.empty:
        raise ValueError(
            f"No data available for {sym} between "
            f"{start_dt.date() if start_dt is not None else 'beginning'} and "
            f"{end_dt.date() if end_dt is not None else 'today'}"
        )

    # Feature engineering ---------------------------------------------------
    df = (
        df.pipe(add_log_return)
        .pipe(add_rolling_stats, lookback=lookback)
        .pipe(add_zscore, lookback=lookback)
    )

    # Signal + back‑test ----------------------------------------------------
    signal = signal_from_zscore(df, threshold=thresh, mode=mode)
    results = run_backtest(df, signal, hold_bars=hold_bars, fee_bps=fee_bps)

    logger.info("%s metrics: %s", sym, results["metrics"])
    return df, results


def main(args: argparse.Namespace) -> None:
    cfg = load_cfg()  # now a plain dict, not a class instance

    # Choose which assets to run -------------------------------------------
    if args.assets.lower() == "all":
        asset_cfgs: List[dict] = cfg["assets"]
    else:
        wanted = {s.strip() for s in args.assets.split(",")}
        asset_cfgs = [a for a in cfg["assets"] if a["symbol"] in wanted]
        if not asset_cfgs:
            raise ValueError(f"No matching assets found for: {wanted}")

    # Save results ----------------------------------------------------------
    paths_cfg = cfg.get("paths", {})
    out_dir = Path(paths_cfg.get("outputs", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_equity: pd.Series | None = None
    all_trades: list[pd.DataFrame] = []

    for asset_cfg in asset_cfgs:
        _, res = process_asset(asset_cfg, cfg)
        trades = res["trades"].assign(symbol=asset_cfg["symbol"])
        all_trades.append(trades)

        if combined_equity is None:
            combined_equity = res["equity"].copy()
        else:
            combined_equity = combined_equity.add(res["equity"], fill_value=0.0)

        # Plot/save equity curve for this asset if requested
        if args.show_plots or args.save_plots:
            save_path = (
                out_dir / f"equity_{asset_cfg['symbol'].replace('/', '')}.pdf"
                if args.save_plots
                else None
            )
            plot_equity(
                res["equity"],
                title=f"Equity Curve – {asset_cfg['symbol']}",
                show=args.show_plots,
                save_path=save_path,
            )

    # Portfolio‑level plot --------------------------------------------------
    if (
        (args.show_plots or args.save_plots)
        and combined_equity is not None
        and len(asset_cfgs) > 1
    ):
        save_path = (out_dir / "equity_combined.pdf") if args.save_plots else None
        plot_equity(
            combined_equity,
            title="Combined Equity Curve (simple sum)",
            show=args.show_plots,
            save_path=save_path,
        )

    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(
            out_dir / "trades.csv", index=False
        )
    if combined_equity is not None:
        combined_equity.to_csv(out_dir / "equity_curve.csv")

    logger.info("Finished – results saved to %s", out_dir.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Crypto Price Anomaly MVP end‑to‑end."
    )
    parser.add_argument(
        "--assets",
        default="all",
        help="Comma‑separated list of symbols (e.g. 'BTC/USDT,ETH/USDT') or 'all' (default).",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display equity curve plots using Matplotlib.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save equity curve plots to the outputs directory.",
    )

    main(parser.parse_args())
