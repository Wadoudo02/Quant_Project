# src/features.py

import numpy as np
import pandas as pd

def add_log_return(
    df: pd.DataFrame,
    price_col: str = "close",
    out_col: str = "log_return",
) -> pd.DataFrame:
    """
    Add a column of log returns to df.

    log_return[t] = log(price[t] / price[t-1])

    Converting to log returns makes percentage changes additive,
    smoothing multiplicative noise—similar to taking logs in
    exponential decay experiments.
    """
    # Shift by one bar to get previous price
    prev = df[price_col].shift(1)
    df[out_col] = np.log(df[price_col] / prev)
    return df


def add_rolling_stats(
    df: pd.DataFrame,
    col: str = "log_return",
    lookback: int = 96,
) -> pd.DataFrame:
    """
    Compute rolling mean and rolling std of `col` over `lookback` bars.

    - Mean:    (analogous to a moving average filter)
    - Std dev: (analogous to noise level / volatility estimate)
    """
    mean_col = f"{col}_mean_{lookback}"
    std_col  = f"{col}_std_{lookback}"

    # .rolling().mean() and .std(ddof=0) mirror population statistics
    df[mean_col] = df[col].rolling(window=lookback).mean()
    df[std_col]  = df[col].rolling(window=lookback).std(ddof=0)
    return df


def add_zscore(
    df: pd.DataFrame,
    col: str = "log_return",
    lookback: int = 96,
    out_col: str = "zscore",
) -> pd.DataFrame:
    """
    Standardise each return by its recent rolling mean & volatility:

    zscore[t] = (col[t] - rolling_mean[t]) / rolling_std[t]

    A Z-score ≥ 2.5 means the return was 2.5σ above its 2-day average,
    marking a statistically significant “shock.”
    """
    mean_col = f"{col}_mean_{lookback}"
    std_col  = f"{col}_std_{lookback}"
    df[out_col] = (df[col] - df[mean_col]) / df[std_col]
    return df