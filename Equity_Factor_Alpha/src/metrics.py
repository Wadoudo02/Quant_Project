"""
metrics.py
-----------

Light‑weight performance metric utilities reused from my Crypto Price
Anomaly project.  Two helpers are exported for now:

    • :func:`sharpe_ratio` → float
    • :func:`max_drawdown` → (abs_drawdown, pct_drawdown)

Both functions operate on ``pandas`` objects but avoid any
project‑specific dependencies so they can be reused in notebooks
or future live‑trading code without modification.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm

__all__ = ["sharpe_ratio", "max_drawdown", "compute_alpha"]


def _infer_periods_per_year(idx: pd.Index) -> float:
    """Best‑effort inference of the annualisation factor from a DatetimeIndex."""
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 3:
        return 252.0  # sensible default for daily data
    median_step = idx.to_series().diff().median()
    if pd.isna(median_step) or median_step == pd.Timedelta(0):
        return 252.0
    return pd.Timedelta(days=365).total_seconds() / median_step.total_seconds()


def sharpe_ratio(
    returns: Union[pd.Series, np.ndarray],
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: float | None = None,
    ddof: int = 0,
) -> float:
    """
    Compute the annualised Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series or 1‑D np.ndarray
        Period‑by‑period **arithmetic** returns (can be $PnL or % – the
        Sharpe is scale‑invariant).  A Pandas Series with a DateTimeIndex
        lets the function auto‑infer `periods_per_year`.
    risk_free_rate : float, default 0
        Annual risk‑free rate expressed in the *same unit* as ``returns``
        (e.g. if returns are in %, RF should also be in %).
    periods_per_year : float, optional
        Override the annualisation factor.  If None, the function
        auto‑infers it from the index (falls back to 252).
    ddof : int, default 0
        Delta‑degrees‑of‑freedom used in the standard‑deviation estimator.

    Returns
    -------
    float
        Annualised Sharpe ratio; ``np.nan`` if undefined (e.g. zero variance).
    """
    r = pd.Series(returns).dropna()
    if r.empty:
        return np.nan
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(r.index)
    rf_per_period = risk_free_rate / periods_per_year
    excess = r - rf_per_period
    std = excess.std(ddof=ddof)
    if std == 0:
        return np.nan
    return (excess.mean() / std) * np.sqrt(periods_per_year)


def max_drawdown(equity: Union[pd.Series, np.ndarray]) -> Tuple[float, float]:
    """
    Calculate absolute and percentage max drawdown of an equity curve.

    Parameters
    ----------
    equity : pd.Series or 1‑D np.ndarray
        Cumulative net‑PnL (or account equity).  Must be on a non‑decreasing
        time axis but need not start at zero.

    Returns
    -------
    (abs_dd, pct_dd) : tuple[float, float]
        ``abs_dd`` – largest *absolute* drop from a peak (same units as equity).
        ``pct_dd`` – that drop as a proportion of the peak (in [0, 1]).
    """
    eq = pd.Series(equity).ffill()
    if eq.empty:
        return 0.0, 0.0
    running_max = eq.cummax()
    underwater = running_max - eq
    pct_under = underwater / running_max.replace(0, np.nan)
    abs_dd = underwater.max()
    pct_dd = pct_under.max()
    return float(abs_dd), float(0.0 if pd.isna(pct_dd) else pct_dd)


def compute_alpha(
    portfolio_returns: Union[pd.Series, np.ndarray],
    benchmark_returns: Union[pd.Series, np.ndarray],
) -> Tuple[float, float, float]:
    """
    Estimate Jensen’s α and β by regressing portfolio returns on a benchmark.

    Parameters
    ----------
    portfolio_returns : pd.Series or 1‑D np.ndarray
        Periodic arithmetic returns of the strategy.
    benchmark_returns : pd.Series or 1‑D np.ndarray
        Synchronous benchmark returns (e.g. a morally‑compliant equity index).

    Returns
    -------
    (alpha, beta, alpha_t) : tuple[float, float, float]
        alpha      – intercept of the regression (Jensen’s α) per period.
        beta       – slope versus the benchmark.
        alpha_t    – t‑statistic of the α coefficient.
    """
    # Ensure aligned pandas Series
    p = pd.Series(portfolio_returns).dropna()
    b = pd.Series(benchmark_returns).dropna()
    p, b = p.align(b, join="inner")
    if p.empty:
        return np.nan, np.nan, np.nan

    X = sm.add_constant(b)  # adds intercept
    model = sm.OLS(p, X).fit()

    alpha = float(model.params["const"])
    beta = float(model.params.iloc[1])
    alpha_t = float(model.tvalues["const"])
    return alpha, beta, alpha_t
