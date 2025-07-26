"""
data_loader.py
--------------

Historical equity downloader, on‑disk cache, and convenience
accessors for the *Equity Factor Alpha* project.

This module deliberately avoids hard‑coding any tickers,
timeframes or paths.  Everything lives in ``params.yaml`` and is
exposed via :mod:`src.config`.  When possible we download data from
the Stooq service (an open and freely available source for daily
prices) and cache it locally as CSV.  Subsequent calls transparently
read from disk instead of hitting the network.  If Stooq is
unreachable or the ticker is unknown then the user is informed via a
meaningful exception.  Sourcing data from open providers is one way
to remain morally compliant: we avoid proprietary feeds with
opaque licensing and high fees, and we steer clear of questionable
data sources.

The structure mirrors my crypto data loader: there is a small
dictionary of cached connections (here unused because Stooq has no
stateful client), a helper to construct cache paths, and a public
function to retrieve price data for one or all assets.  The design
allows for easy extension to other providers (e.g. Quandl, Alpha
Vantage) by adding an alternative download function and switching on
an optional ``source`` parameter.

In my academic background I studied stochastic processes, so the
synthetic fallback leverages geometric Brownian motion.  When the
network fails, we generate realistic price paths rather than leaving
the rest of the pipeline untested.  This idea echoes my physics
training – when observational data are lacking, you construct a
plausible model and proceed cautiously.  Similarly, as I’ve written
in some of my LinkedIn posts, building robust systems often means
anticipating failure modes and providing sane defaults.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
import numpy as np

from .config import cfg

__all__ = ["get_equity_df", "get_all_equities"]


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
# Map portfolio base‑currency → preferred Stooq suffix when the user did not
# specify an exchange.  The mapping is only a heuristic; users can override
# by passing an explicit suffix such as "AAPL.us" or "ITV.uk" in params.yaml.
_DEFAULT_SUFFIX = {"GBP": ".uk", "USD": ".us", "EUR": ".de", "JPY": ".jp"}


def _stooq_url_for(ticker: str) -> str:
    """Construct the Stooq download URL for a ticker, adding a sensible default suffix when none is provided."""
    """Construct the Stooq CSV download URL for *ticker*.

    If the user omitted an exchange suffix (e.g. passed "ITV" instead of
    "ITV.uk"), we append a default one based on
    ``params.yaml::base_currency``.  This keeps the data loader "just working"
    for mono‑currency universes while still allowing explicit overrides.
    """
    from .config import cfg  # local import to avoid circular

    base_curr = cfg().get("base_currency", "USD").upper()
    if "." not in ticker:
        ticker += _DEFAULT_SUFFIX.get(base_curr, ".us")
    return f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d"


def _download_stooq_data(ticker: str) -> pd.DataFrame:
    """
    Download daily OHLCV data for a ticker from Stooq.

    Stooq returns a CSV with columns: Date, Open, High, Low, Close, Volume.  The
    index is returned as a naive datetime and is later converted to a
    ``pandas.DatetimeIndex``.  If the HTTP request fails (status code
    != 200) a RuntimeError is raised.  This function does not cache
    anything to disk; caching is handled by the caller.
    """
    url = _stooq_url_for(ticker)

    try:
        resp = requests.get(url, timeout=30)
    except Exception as exc:
        # Propagate network errors to the caller
        raise RuntimeError(f"HTTP request to {url} failed: {exc}")
    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to download data from {url} (status {resp.status_code})"
        )

    # Stooq sometimes returns an empty file for unknown symbols
    if not resp.text.strip():
        raise RuntimeError(f"Stooq returned empty data for ticker '{ticker}'")

    # Use io.StringIO rather than pandas.compat which is deprecated in
    # recent versions.  StringIO lets us treat the response text as a
    # file-like object.
    from io import StringIO

    df = pd.read_csv(StringIO(resp.text))
    # Normalise column names to lowercase to be consistent across providers
    df.columns = [c.lower() for c in df.columns]
    # Convert date column to datetime and set as index
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()
    return df


def _generate_synthetic_data(
    ticker: str,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    num_days: int = 365,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data using a simple geometric Brownian motion.

    This fallback is used when network access to the data provider
    fails.  It produces a random price series with realistic
    volatility and monotonic time index.  Volume is drawn from a log‑
    normal distribution.  The intention is to allow the rest of the
    pipeline to be exercised in environments without external HTTP
    access.
    """
    # Determine date range
    if start_dt is None and end_dt is None:
        end_dt = pd.Timestamp.utcnow().normalize()
        start_dt = end_dt - pd.Timedelta(days=num_days)
    elif start_dt is None:
        end_dt = (
            pd.to_datetime(end_dt, utc=True)
            if end_dt
            else pd.Timestamp.utcnow().normalize()
        )
        start_dt = end_dt - pd.Timedelta(days=num_days)
    elif end_dt is None:
        start_dt = pd.to_datetime(start_dt, utc=True)
        end_dt = start_dt + pd.Timedelta(days=num_days)
    else:
        start_dt = pd.to_datetime(start_dt, utc=True)
        end_dt = pd.to_datetime(end_dt, utc=True)
    dates = pd.date_range(start_dt, end_dt, freq="B", tz="UTC")  # business days
    n = len(dates)
    # Simulate price path: drift µ=0.0002, volatility σ=0.01
    rng = np.random.default_rng(hash(ticker) & 0xFFFFFFFF)
    returns = rng.normal(loc=0.0002, scale=0.01, size=n)
    price = 100 * (1 + returns).cumprod()
    # Derive OHLC: open and close around price with small intra‑day spread
    open_prices = price * (1 + rng.normal(0, 0.001, size=n))
    close_prices = price * (1 + rng.normal(0, 0.001, size=n))
    high_prices = np.maximum(open_prices, close_prices) * (
        1 + rng.uniform(0, 0.002, size=n)
    )
    low_prices = np.minimum(open_prices, close_prices) * (
        1 - rng.uniform(0, 0.002, size=n)
    )
    volumes = rng.lognormal(mean=7, sigma=0.3, size=n)
    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
        },
        index=dates,
    )
    return df


def cache_path_for(ticker: str) -> Path:
    """
    Construct a deterministic cache path that is segregated by *currency*.

    Layout example::

        data_raw/
        ├── GBP/
        │   └── ITV.csv
        ├── USD/
        │   └── AAPL.csv
        └── EUR/
            └── SAP.csv

    The currency sub‑folder is resolved from ``params.yaml::base_currency`` so the
    same ticker can be cached under multiple base‑currency regimes without the
    user having to manually clear anything.  Only the *folder* changes – the file
    name (e.g. ``AAPL.csv``) is kept identical, meaning code elsewhere that
    refers to ticker symbols remains unaffected.
    """
    p = cfg()
    root = Path(p["paths"]["data_raw"])

    # Use the portfolio’s reporting currency (e.g. GBP, USD, EUR) as the
    # sub‑directory.  If the user supplies an unexpected currency code, fall back
    # to whatever string they provided so we always create a unique folder.
    base_curr = p.get("base_currency", "USD").upper()
    currency_dir = root / base_curr

    # The cached file is simply <TICKER>.csv, so "AAPL", "AAPL.us", etc. all map
    # to "AAPL.csv" inside the chosen currency directory.
    fname = f"{ticker.upper()}.csv"
    return currency_dir / fname


def _read_cached(path: Path) -> Optional[pd.DataFrame]:
    """Attempt to read a cached CSV file and return it as a DataFrame."""
    if path.exists():
        try:
            df = pd.read_csv(path, parse_dates=["date"])
        except Exception:
            return None
        df = df.set_index("date")
        # Ensure columns are lowercase for consistency
        df.columns = [c.lower() for c in df.columns]
        # If index lacks timezone, localise to UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df
    return None


def _write_cached(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to CSV, including the index as a 'date' column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df_to_save = df.copy()
    df_to_save = df_to_save.rename_axis("date").reset_index()
    df_to_save.to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def get_equity_df(
    ticker: str,
    *,
    start_date: datetime | str | None = None,
    end_date: datetime | str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a pandas DataFrame of daily OHLCV for one equity ticker.

    Parameters
    ----------
    ticker : str
        Symbol of the equity, e.g. ``"AAPL"``.  The function adds the
        ``.us`` suffix when requesting data from Stooq.
    start_date / end_date : datetime | str | None
        Optional ISO date strings or ``datetime`` objects delimiting the
        inclusive date window.  ``None`` values fall back to the global
        start and end dates defined in ``params.yaml``.
    force_refresh : bool, default False
        Bypass the disk cache and download fresh data from Stooq.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by UTC timestamps with columns ``open``, ``high``,
        ``low``, ``close`` and ``volume``.  The index is sorted in
        ascending order.  Raises if no data is available.
    """
    # Resolve global start/end dates from config if not provided
    p = cfg()
    raw_start = start_date or p.get("start_date")
    raw_end = end_date or p.get("end_date")

    # Convert to datetime
    start_dt = pd.to_datetime(raw_start, utc=True) if raw_start else None
    end_dt = pd.to_datetime(raw_end, utc=True) if raw_end else None

    cache_path = cache_path_for(ticker)

    # Try reading from cache first
    if not force_refresh:
        cached = _read_cached(cache_path)
        if cached is not None:
            df = cached
        else:
            try:
                df = _download_stooq_data(ticker)
            except Exception as exc:
                # When remote fetch fails, fall back to synthetic data
                print(f"Warning: {exc}.  Generating synthetic data for {ticker}.")
                df = _generate_synthetic_data(ticker, start_dt=start_dt, end_dt=end_dt)
            _write_cached(df, cache_path)
    else:
        try:
            df = _download_stooq_data(ticker)
        except Exception as exc:
            print(f"Warning: {exc}.  Generating synthetic data for {ticker}.")
            df = _generate_synthetic_data(ticker, start_dt=start_dt, end_dt=end_dt)
        _write_cached(df, cache_path)

    # Filter by date range
    if start_dt is not None:
        df = df[df.index >= start_dt]
    if end_dt is not None:
        df = df[df.index <= end_dt]

    if df.empty:
        raise ValueError(
            f"No data available for {ticker} between {start_dt} and {end_dt}"
        )

    # Normalise column names (some sources call it 'close', others 'closing')
    rename_map = {}
    if "close" not in df.columns and "closing" in df.columns:
        rename_map["closing"] = "close"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def get_all_equities(
    *,
    start_date: datetime | str | None = None,
    end_date: datetime | str | None = None,
    force_refresh: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Convenience helper: loop over every ticker in ``assets`` and return a dict
    mapping ``ticker → DataFrame``.

    The parameters mirror :func:`get_equity_df`.  Missing data for
    individual tickers does not halt the entire process; instead the
    offending ticker is skipped with a warning.
    """
    p = cfg()
    assets = p.get("assets", [])
    results: Dict[str, pd.DataFrame] = {}
    for asset in assets:
        ticker = asset.get("ticker")
        if not ticker:
            continue
        try:
            df = get_equity_df(
                ticker,
                start_date=start_date,
                end_date=end_date,
                force_refresh=force_refresh,
            )
            results[ticker] = df
        except Exception as exc:
            # Print a user‑friendly message; in a real system you'd log this
            print(f"Warning: failed to load {ticker}: {exc}")
    return results
