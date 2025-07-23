# src/resample.py

import pandas as pd


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample an OHLCV DataFrame to a new time interval.

    Parameters
    ----------
    df : pd.DataFrame
        Original data indexed by a timezone-aware UTC DatetimeIndex,
        with columns ['open', 'high', 'low', 'close', 'volume'].
    rule : str
        A pandas offset alias string (e.g. '5T' for 5-minute,
        '30T' for 30-minute, '1H' for 1-hour). See
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

    Returns
    -------
    pd.DataFrame
        A new DataFrame at the requested frequency, with:
          - open  = first non-null open in each period
          - high  = maximum high in each period
          - low   = minimum low in each period
          - close = last non-null close in each period
          - volume= sum of volumes in each period
        Any partially filled period (i.e. containing NaNs in any OHLCV column)
        is dropped to ensure you only get “complete” bars.
    """
    # 1. For each new bucket of length `rule`, take the first 'open'
    o = df["open"].resample(rule).first()
    # 2. The highest 'high' over that bucket
    h = df["high"].resample(rule).max()
    # 3. The lowest 'low' over that bucket
    low = df["low"].resample(rule).min()
    # 4. The last 'close' in the bucket
    c = df["close"].resample(rule).last()
    # 5. Sum up all the 'volume' trades in the bucket
    v = df["volume"].resample(rule).sum()

    # 6. Concatenate into one DataFrame
    out = pd.concat([o, h, low, c, v], axis=1)
    out.columns = ["open", "high", "low", "close", "volume"]

    # 7. Drop any row where at least one of OHLCV is NaN
    #    (e.g. incomplete first/last period)
    out = out.dropna(how="any")
    return out
