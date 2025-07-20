# src/backtest.py
"""
Bar-based long-only back-tester for the Crypto-Price-Anomaly project.

Key assumptions
---------------
* **Long-only, one position at a time** – the MVP stays halal by avoiding
  margin/shorting.  We therefore track a single open position flag.
* **Enter on the *next* bar open** after a signal fires, mirroring realistic
  execution (you only know today's close once the bar ends).
* **Hold a fixed number of bars** (`hold_bars`).  This is the simplest
  “physics-style” experiment: apply an impulse (the entry) and let the system
  evolve for a set time before measuring the outcome.
* **Flat notional sizing** – every trade commits the same cash amount
  (`notional`, e.g. \$1 000).  Position PnL is therefore directly interpretable
  in dollars.
* **Constant proportional costs** – fees/slippage are charged symmetrically on
  both entry and exit (`fee_bps` basis points per side).

The function returns:
    1. A **trades DataFrame** – granular record of every entry/exit.
    2. An **equity Series** – cumulative net PnL indexed to the original
       price-bar timeline so it can be plotted alongside price.
    3. A **metrics dict** – headline performance figures.

This keeps analytics decoupled from plotting or I/O, respecting single-responsibility.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    *,
    hold_bars: int = 12,
    fee_bps: float = 5.0,
    notional: float = 1_000.0,
) -> Dict[str, Any]:
    """
    Execute a vectorised forward-holding back-test.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain **'open'** prices and share the same DatetimeIndex as
        `signal`. (We never look at 'high'/'low'; simplify first experiment.)
    signal : pd.Series[bool]
        Boolean flag marking when a trade *should* be opened.
    hold_bars : int, default 12
        Number of bars to remain in the position before exit.
    fee_bps : float, default 5
        Exchange taker fee expressed in basis-points *per side*.
        Total round-trip cost = 2 × fee_bps.
    notional : float, default 1_000
        Cash size of each trade (e.g. \$1 000 ≈ one “unit mass” in a physics
        analogy).

    Returns
    -------
    dict with keys
        'trades' : pd.DataFrame
            Entry/exit timestamps, prices, returns, fees, net PnL.
        'equity' : pd.Series
            Cumulative net PnL aligned to original df.index.
        'metrics' : dict
            Aggregate performance statistics.
    """
    # --- 0. Sanity checks ----------------------------------------------------
    if not df.index.equals(signal.index):
        raise ValueError("`df` and `signal` must share the same index.")
    if 'open' not in df.columns:
        raise KeyError("Input DataFrame requires an 'open' column.")

    fee_rate = fee_bps / 10_000  # bps → proportion
    o = df['open'].to_numpy()    # ndarray for speed
    sig = signal.to_numpy(bool)

    # Pre-allocate lists (faster than repeatedly growing DataFrame)
    entries, exits, e_price, x_price = [], [], [], []
    gross_ret, gross_pnl, fees, net_pnl = [], [], [], []

    in_pos = False
    entry_bar = -1  # index of bar we entered on (for hold tracking)

    # --- 1. Iterate through bars --------------------------------------------
    # We stop at len(df) - 1 because we need *next* bar's open for entry.
    for i in range(len(df) - 1):
        # 1A. Entry condition – only if flat and current bar flagged True
        if (not in_pos) and sig[i]:
            entry_bar = i + 1                 # trade triggers *next* bar
            entry_price = o[entry_bar]
            in_pos = True
            continue  # move to next iteration so we don't accidentally exit on entry bar

        # 1B. Exit condition – when position has lived `hold_bars`
        if in_pos:
            hold_elapsed = i - entry_bar
            if hold_elapsed >= hold_bars:
                exit_bar = i          # exit at *current* bar open
                exit_price = o[exit_bar]

                # --- 2. Book-keeping ---------------------------------------
                r = (exit_price - entry_price) / entry_price
                g_pnl = notional * r
                cost = notional * fee_rate * 2      # entry + exit
                n_pnl = g_pnl - cost

                # Append to running log
                entries.append(df.index[entry_bar])
                exits.append(df.index[exit_bar])
                e_price.append(entry_price)
                x_price.append(exit_price)
                gross_ret.append(r)
                gross_pnl.append(g_pnl)
                fees.append(cost)
                net_pnl.append(n_pnl)

                # Reset
                in_pos = False
                entry_bar = -1

    # --- 3. Assemble trade ledger -------------------------------------------
    trades = pd.DataFrame(
        {
            'entry_time': entries,
            'exit_time': exits,
            'entry_price': e_price,
            'exit_price': x_price,
            'gross_ret': gross_ret,
            'gross_pnl': gross_pnl,
            'fees': fees,
            'net_pnl': net_pnl,
        }
    )

    # --- 4. Equity curve -----------------------------------------------------
    if trades.empty:
        # No trades: flat equity series of zeros for compatibility
        equity = pd.Series(0.0, index=df.index, name='equity')
    else:
        # Timestamp equity at trade *exit* so PnL is not clairvoyant
        equity = (
            trades.set_index('exit_time')['net_pnl']
            .cumsum()
            .reindex(df.index, method='ffill')
            .fillna(0.0)
            .rename('equity')
        )

    # --- 5. Headline metrics -------------------------------------------------
    if trades.empty:
        metrics = {
            'total_net_pnl': 0.0,
            'num_trades': 0,
            'hit_rate': np.nan,
            'avg_trade_ret': np.nan,
            'max_drawdown_abs': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe': np.nan,
        }
    else:
        from .metrics import sharpe_ratio, max_drawdown  # local import avoids circularity

        # Instantaneous equity returns for Sharpe (diff of cumulative net pnl)
        equity_ret = equity.diff().fillna(0.0)

        dd_abs, dd_pct = max_drawdown(equity)

        metrics = {
            'total_net_pnl': trades['net_pnl'].sum(),
            'num_trades': len(trades),
            'hit_rate': (trades['gross_ret'] > 0).mean(),
            'avg_trade_ret': trades['gross_ret'].mean(),
            'max_drawdown_abs': float(dd_abs),
            'max_drawdown_pct': float(dd_pct),
            'sharpe': sharpe_ratio(equity_ret),
        }

    return {'trades': trades, 'equity': equity, 'metrics': metrics}