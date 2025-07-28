"""
features.py
-----------

Lightweight feature engineering utilities for the *Equity Factor Alpha*
project.  The helpers in this module transform raw price series into
inputs suitable for machine learning models.  Inspired by my crypto
project, each function returns a new DataFrame rather than modifying
the input in place; this makes chaining with the ``pipe`` method
trivial and keeps side‑effects contained.

The emphasis here is on simplicity and interpretability.  Basic
statistics such as returns, rolling means and standard deviations
capture momentum and volatility – the bread and butter of factor
investing.  Additional features can easily be added (e.g. RSI,
MACD) by following the same pattern.  All operations are vectorised
using NumPy/Pandas for efficiency.

As someone who bridged the worlds of physics and finance early on,
I’ve seen how over‑engineering features can lead to fragile models.  In
my previous work on the Crypto Price Anomaly project I deliberately
kept the feature set concise; the same philosophy applies here.  For
example, choosing lookback windows of 5, 20 and 60 days roughly
corresponds to weekly, monthly and quarterly horizons – a nod to the
classic Fama–French factor framework.  Such choices are motivated by
both academic research and practical experience building trading
signals.  Keeping the design transparent also reinforces moral
compliance: we avoid hidden state or leverage that might mislead
users of our models.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .config import cfg

__all__ = [
    "add_returns",
    "add_rolling_features",
    "prepare_feature_matrix",
]


def add_returns(
    df: pd.DataFrame,
    price_col: str = "close",
    out_col: str = "returns",
) -> pd.DataFrame:
    """
    Compute simple arithmetic returns and append them as a new column.

    ``returns[t] = (price[t] / price[t-1]) - 1``

    Log returns could be used instead but for equity data daily
    arithmetic returns are appropriate and intuitive.  The first
    element is NaN because there is no prior price.
    """
    df = df.copy()
    prev = df[price_col].shift(1)
    df[out_col] = (df[price_col] / prev) - 1
    return df


def add_rolling_features(
    df: pd.DataFrame,
    col: str = "returns",
    windows: Iterable[int] | None = None,
) -> pd.DataFrame:
    """
    Add rolling mean, standard deviation and momentum features.

    For each window ``w`` in ``windows``, the following columns are added:

    - ``{col}_mean_{w}`` – the average return over the last ``w`` bars.
    - ``{col}_std_{w}``  – the population standard deviation (ddof=0).
    - ``{col}_mom_{w}``  – the cumulative return (sum of returns) over ``w`` bars.

    These features correspond loosely to value (mean), risk
    (volatility) and momentum factors and are widely used in asset
    pricing models.  The windows default to those specified in
    ``params.yaml`` under the ``features.windows`` key.
    """
    df = df.copy()
    if windows is None:
        windows = cfg()["features"].get("windows", [5, 20, 60])
    for w in windows:
        mean_col = f"{col}_mean_{w}"
        std_col = f"{col}_std_{w}"
        mom_col = f"{col}_mom_{w}"
        df[mean_col] = df[col].rolling(window=w).mean()
        df[std_col] = df[col].rolling(window=w).std(ddof=0)
        # Momentum as cumulative return (sum) over the window
        df[mom_col] = df[col].rolling(window=w).sum()
    return df


def prepare_feature_matrix(
    dfs: Dict[str, pd.DataFrame],
    *,
    windows: Iterable[int] | None = None,
    lookahead: int = 1,
    return_col: str | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Construct a multi‑asset feature matrix and target vector.

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        Mapping from ticker to its DataFrame of prices and engineered
        features.  Each DataFrame must already contain the return column
        specified by ``return_col``.
    windows : iterable[int], optional
        Rolling windows for additional features; defaults to the
        ``features.windows`` list from ``params.yaml``.  This is used
        when features have not yet been added via :func:`add_rolling_features`.
    lookahead : int, default 1
        Number of periods ahead to compute the target return.  ``1``
        corresponds to next‑day returns.  A value of ``5`` would train the
        model to predict five days ahead.
    return_col : str, optional
        Name of the return column.  If ``None`` the default from
        ``params.yaml`` (``features.return_col``) is used.

    Returns
    -------
    (X, y) : (pd.DataFrame, pd.Series)
        ``X`` is a 2‑D DataFrame with MultiIndex (date, ticker) and
        feature columns; ``y`` is a Series aligned to ``X`` containing
        the future returns over ``lookahead`` periods.  Rows with any
        NaN values are dropped to avoid contaminating the model.
    """
    if return_col is None:
        return_col = cfg()["features"].get("return_col", "returns")
    if windows is None:
        windows = cfg()["features"].get("windows", [5, 20, 60])

    frames: List[pd.DataFrame] = []
    targets: List[pd.Series] = []

    for ticker, df in dfs.items():
        # Ensure the return column exists
        if return_col not in df.columns:
            raise KeyError(
                f"DataFrame for {ticker} is missing return column '{return_col}'"
            )

        # Add rolling features if not already present
        required_cols = [f"{return_col}_mean_{w}" for w in windows]
        if not all(c in df.columns for c in required_cols):
            df = add_rolling_features(df, col=return_col, windows=windows)

        # Compute the target: future return over lookahead periods
        tgt = df[return_col].shift(-lookahead)

        # Select feature columns (exclude the original return column)
        feature_cols = [c for c in df.columns if c != return_col]
        X = df[feature_cols].copy()
        # Assign a MultiIndex: (date, ticker)
        X.index = pd.MultiIndex.from_arrays(
            [df.index, [ticker] * len(df)], names=["date", "ticker"]
        )
        tgt.index = X.index
        frames.append(X)
        targets.append(tgt)

    # Concatenate along rows
    X_all = pd.concat(frames)
    y_all = pd.concat(targets)

    # Drop rows with NaNs in any feature or target
    mask = ~(X_all.isna().any(axis=1) | y_all.isna())
    X_all = X_all[mask]
    y_all = y_all[mask]

    return X_all, y_all
