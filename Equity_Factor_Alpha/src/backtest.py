"""
backtest.py
-----------

Simple long‑only back‑tester for the *Equity Factor Alpha* project.

This back‑tester takes a panel of model predictions and real
subsequent returns and simulates a naive strategy: each day we
select the ``top_n`` stocks ranked by predicted return, open
positions with equal weight, hold them for ``hold_days`` trading
days and then close them.  Positions are allowed to overlap, so the
portfolio can hold more than ``top_n`` positions at once.  Fees are
applied on both entry and exit as a fixed basis‑points charge.

The output mirrors that of my crypto back‑tester: a trades ledger,
equity curve aligned to the price timeline and a metrics summary.
Avoiding short selling ensures the strategy remains morally
compliant.

When I first back‑tested trading strategies as a graduate student I
was tempted to over‑fit and to chase leverage.  Over the years I’ve
come to appreciate the virtues of restraint.  The long‑only design
here reflects that evolution: by limiting downside exposure we keep
risk transparent and avoid the psychological pressure of large
drawdowns.  It also simplifies the implementation, which helps
other researchers extend or audit the code.  In several of my
professional posts I’ve advocated for such transparent workflows –
this module puts those principles into practice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import numpy as np

from .config import cfg
from .metrics import sharpe_ratio, max_drawdown, compute_alpha

__all__ = ["run_backtest"]


@dataclass
class Position:
    ticker: str
    days_left: int
    entry_date: pd.Timestamp
    entry_price: float


def run_backtest(
    predictions: pd.Series,
    returns: pd.Series,
    *,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float | None = None,
    hold_days: int | None = None,
    top_n: int | None = None,
    fee_bps: float | None = None,
    notional: float = 1_000.0,
) -> Dict[str, object]:
    """
    Execute a daily long‑only back‑test based on model predictions.

    Parameters
    ----------
    predictions : pd.Series
        Predicted returns indexed by MultiIndex (date, ticker).  Higher
        values imply a stronger bullish signal.  Only the relative order
        matters for ranking.
    returns : pd.Series
        Actual realised returns over the lookahead period, aligned to
        ``predictions``.  Should share the same MultiIndex.
    hold_days : int, optional
        Number of days to hold each position.  Defaults to the value
        specified in ``params.yaml`` under ``backtest.hold_days``.
    top_n : int, optional
        Number of stocks to select each day.  Defaults to the value
        specified in ``params.yaml`` under ``backtest.top_n``.
    fee_bps : float, optional
        Transaction cost in basis points per side.  Defaults to
        ``backtest.fee_bps`` from config.  Set to zero for frictionless
        trading.
    notional : float, default 1_000
        Cash value allocated to *each* new position.  You can think
        of this as the unit trade size; if ``top_n=2`` you deploy
        ``2×notional`` on day one, then more as positions overlap.

    benchmark_returns : pd.Series, optional
        Benchmark returns aligned to ``returns``.  If provided, the
        function computes Jensen’s α, β and the α t‑statistic.

    risk_free_rate : float, optional
        Per‑period (e.g. daily) risk‑free rate used when computing the
        Sharpe ratio.  If omitted, defaults to ``backtest.risk_free_rate``
        from ``params.yaml`` (falls back to 0.0).

    Returns
    -------
    dict with keys
        'trades' : pd.DataFrame
            Entry/exit dates, tickers, returns and PnL for each trade.
        'equity' : pd.Series
            Cumulative net PnL indexed by date.
        'metrics' : dict
            Aggregate performance statistics: total PnL, number of trades,
            hit rate, average trade return, max drawdown and Sharpe ratio.
    """
    # Pull defaults from configuration
    p = cfg()
    bt_cfg = p.get("backtest", {})
    hold_days = bt_cfg.get("hold_days") if hold_days is None else hold_days
    top_n = bt_cfg.get("top_n") if top_n is None else top_n
    fee_bps = bt_cfg.get("fee_bps") if fee_bps is None else fee_bps
    fee_rate = (fee_bps or 0) / 10_000
    risk_free_rate = (
        bt_cfg.get("risk_free_rate", 0.0) if risk_free_rate is None else risk_free_rate
    )

    # Ensure predictions and returns align
    if not predictions.index.equals(returns.index):
        raise ValueError("predictions and returns must have identical index")

    dates = sorted({idx[0] for idx in predictions.index})
    # Pre‑allocate PnL time series
    daily_pnl: Dict[pd.Timestamp, float] = {d: 0.0 for d in dates}
    trades: List[Dict[str, object]] = []
    open_positions: List[Position] = []

    # For each trading day
    for date in dates:
        # 1. Update existing positions
        new_open_positions: List[Position] = []
        for pos in open_positions:
            # Realised return on this day
            ret = returns.loc[(date, pos.ticker)]
            gross_pnl = notional * ret
            # Only book PnL if we are still in the position
            daily_pnl[date] += gross_pnl
            pos.days_left -= 1
            if pos.days_left <= 0:
                # Close the trade: apply fees on exit only (entry fees were charged at entry)
                cost = notional * fee_rate
                net_pnl = gross_pnl - cost
                trades.append(
                    {
                        "entry_date": pos.entry_date,
                        "exit_date": date,
                        "ticker": pos.ticker,
                        "gross_ret": ret,
                        "gross_pnl": gross_pnl,
                        "fees": cost,
                        "net_pnl": net_pnl,
                    }
                )
            else:
                new_open_positions.append(pos)
        open_positions = new_open_positions

        # 2. Select new positions based on predictions for this date
        day_preds = predictions.loc[date]
        # Sort descending and take the top N tickers
        ranked = day_preds.sort_values(ascending=False)
        selected = ranked.iloc[:top_n].index.get_level_values(0)  # just tickers
        # Open new positions
        for ticker in selected:
            # Entry fee on open
            cost = notional * fee_rate
            open_positions.append(
                Position(
                    ticker=ticker,
                    days_left=hold_days,
                    entry_date=date,
                    entry_price=np.nan,  # price not tracked explicitly
                )
            )
            daily_pnl[date] -= cost  # pay fee up front

    # Build equity curve: cumulative sum of daily PnL
    equity_series = pd.Series(daily_pnl).cumsum().sort_index().rename("equity")

    # Compile trade ledger into DataFrame
    trades_df = pd.DataFrame(trades)
    alpha = beta = alpha_t = np.nan
    if not trades_df.empty:
        dd_abs, dd_pct = max_drawdown(equity_series)
        equity_diff = equity_series.diff().fillna(0.0)
        if benchmark_returns is not None:
            # Align benchmark with portfolio return series
            bench = benchmark_returns.reindex(equity_diff.index).fillna(0.0)
            alpha, beta, alpha_t = compute_alpha(equity_diff, bench)
        metrics = {
            "total_net_pnl": trades_df["net_pnl"].sum(),
            "num_trades": len(trades_df),
            "hit_rate": (trades_df["gross_ret"] > 0).mean(),
            "avg_trade_ret": trades_df["gross_ret"].mean(),
            "max_drawdown_abs": float(dd_abs),
            "max_drawdown_pct": float(dd_pct),
            "sharpe": sharpe_ratio(equity_diff, risk_free_rate=risk_free_rate),
            "alpha": alpha,
            "beta": beta,
            "alpha_tstat": alpha_t,
        }
    else:
        metrics = {
            "total_net_pnl": 0.0,
            "num_trades": 0,
            "hit_rate": np.nan,
            "avg_trade_ret": np.nan,
            "max_drawdown_abs": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe": sharpe_ratio(
                pd.Series(dtype=float), risk_free_rate=risk_free_rate
            ),
            "alpha": alpha,
            "beta": beta,
            "alpha_tstat": alpha_t,
        }

    return {"trades": trades_df, "equity": equity_series, "metrics": metrics}
