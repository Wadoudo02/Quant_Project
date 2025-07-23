"""
data_loader.py
--------------

Historical OHLCV downloader, on-disk cache, and convenience accessors
for the *Crypto Price Anomaly* project.

Key design points
=================
1. **Zero hard-coding** – everything lives in ``params.yaml`` and is
   surfaced by ``src.config.cfg()``.
2. **Disk-level cache** – first call hits the exchange; subsequent calls
   transparently `read_parquet` / `read_csv` instead.
3. **Paginated fetch loop** – Binance returns max 1000 bars; we iterate
   until we cover the full date range requested.
4. **Stateless outside** → **stateful inside** – exchange objects are
   memoised so we respect CCXT’s rate-limit guidance with minimal fuss.

Example
-------
>>> from src.data_loader import get_asset_df
>>> df_btc = get_asset_df("BTC/USDT")        # returns 30-minute bars
>>> print(df_btc.head())
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import ccxt
import pandas as pd
from .config import cfg, cfg_ns

# --------------------------------------------------------------------------- #
# Timeframe alias mapping (user‑friendly → CCXT native)
# --------------------------------------------------------------------------- #
_TIMEFRAME_ALIASES: Dict[str, str] = {
    "1min": "1m",
    "3min": "3m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "60min": "1h",
    "90min": "90m",
}


__all__ = [
    "get_exchange",
    "fetch_ohlcv_chunk",
    "fetch_full_history",
    "cache_path_for",
    "get_asset_df",
    "get_all_assets",
]

# --------------------------------------------------------------------------- #
# Internal singletons
# --------------------------------------------------------------------------- #
_EXCH_CACHE: Dict[str, ccxt.Exchange] = {}


def get_exchange(name: str) -> ccxt.Exchange:
    """
    Return a memoised ccxt exchange instance.

    *Why?*
    Each ccxt object maintains its own rate limiter. Re-using one object per
    process avoids blowing through the exchange’s request quota – analogous
    to re-using a single HTTP `Session` instead of opening new sockets for
    every GET in web-scraping.
    """
    if name not in _EXCH_CACHE:
        klass = getattr(ccxt, name)
        _EXCH_CACHE[name] = klass({"enableRateLimit": True})
    return _EXCH_CACHE[name]


# --------------------------------------------------------------------------- #
# Raw fetch helpers
# --------------------------------------------------------------------------- #
def _to_ms(dt: datetime) -> int:
    """Utility: convert aware `datetime` → milliseconds since epoch."""
    return int(dt.timestamp() * 1_000)


def fetch_ohlcv_chunk(
    symbol: str,
    exchange_name: str,
    timeframe: str,
    *,
    since_ms: int | None = None,
    limit: int = 1_000,
) -> List[list]:
    """
    Fetch *one* chunk (≤ ``limit`` rows) of candles.

    Returns the raw list of lists as given by ccxt:
    ``[timestamp, open, high, low, close, volume]``.
    """
    ex = get_exchange(exchange_name)
    # Normalise timeframe to CCXT native string
    timeframe = _TIMEFRAME_ALIASES.get(timeframe, timeframe)
    return ex.fetch_ohlcv(
        symbol=symbol, timeframe=timeframe, since=since_ms, limit=limit
    )


def fetch_full_history(
    symbol: str,
    exchange_name: str,
    timeframe: str,
    *,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
) -> pd.DataFrame:
    """
    Loop through successive 1000-bar windows until we cover the
    ``start_dt → end_dt`` range requested.

    Parameters
    ----------
    start_dt : first candle **included** (UTC, aware).
               ``None`` → let Binance give us the earliest it can.
    end_dt   : last candle **included** (UTC, aware).
               ``None`` → up to *now*.

    Notes
    -----
    - CCXT uses inclusive *since* but exclusive *until* semantics.
    - Binance gives **closing** timestamps → we advance ``since`` by +1 ms
      to avoid duplicate rows on the next loop.
    """
    if start_dt is not None and start_dt.tzinfo is None:
        raise ValueError("start_dt must be timezone-aware (UTC) or None.")
    if end_dt is not None and end_dt.tzinfo is None:
        raise ValueError("end_dt must be timezone-aware (UTC) or None.")

    since_ms = _to_ms(start_dt) if start_dt else None
    until_ms = _to_ms(end_dt) if end_dt else None

    all_rows: List[list] = []
    while True:
        chunk = fetch_ohlcv_chunk(symbol, exchange_name, timeframe, since_ms=since_ms)
        if not chunk:
            break

        # If we reached beyond end_dt stop early
        if until_ms and chunk[0][0] > until_ms:
            break

        all_rows.extend(chunk)

        # Advance since to last ts + 1 ms (close time given by Binance)
        since_ms = chunk[-1][0] + 1

        # Binance returns <1000 rows when it hits the most recent candle
        if len(chunk) < 1_000:
            break

    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(all_rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)

    # Trim rows beyond end_dt (if supplied)
    if end_dt is not None:
        df = df.loc[:end_dt]

    return df


# --------------------------------------------------------------------------- #
# Cache helpers
# --------------------------------------------------------------------------- #
def cache_path_for(symbol: str, timeframe: str) -> Path:
    """
    Construct a deterministic *relative* cache path, e.g.

    ``data_raw/BTCUSDT_30m.parquet``

    The root folder comes from ``paths.data_raw`` in params.yaml.
    """
    p = cfg()
    root = Path(p["paths"]["data_raw"])
    fname = f"{symbol.replace('/', '')}_{timeframe}.parquet"
    return root / fname


def _read_cached(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    return None


def _write_cached(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)


# --------------------------------------------------------------------------- #
# Public user-facing API
# --------------------------------------------------------------------------- #
def get_asset_df(
    symbol: str,
    *,
    exchange_name: (
        str | None
    ) = None,  # means "I expect a string here, but if you don’t give me anything I’ll use None.""
    timeframe: (
        str | None
    ) = None,  # Without | None, a type‐checker would complain “Hey, you defaulted to None but your annotation never said it could be None!”
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a **pandas** DataFrame of OHLCV for one asset.

    The signature mirrors CCXT but defaults are injected from ``params.yaml``
    so you seldom need to specify anything except the symbol.

    Parameters
    ----------
    symbol        : e.g. ``"BTC/USDT"`` (must match CCXT format)
    exchange_name : falls back to the ``exchange`` field for *that* symbol
                    in ``assets`` list; else ``"binance"``.
    timeframe     : defaults to global ``bar_interval`` param.
    start_dt / end_dt : optional UTC aware datetimes.
    force_refresh : bypass disk cache and hit the exchange anyway.
    """
    # Resolve default exchange for the symbol --------------------------------
    if exchange_name is None:
        matches = [a for a in cfg()["assets"] if a["symbol"] == symbol]
        if not matches:
            raise ValueError(
                f"Symbol {symbol!r} not found in params.yaml 'assets' list."
            )
        exchange_name = matches[0]["exchange"]

    # Resolve timeframe ------------------------------------------------------
    timeframe = timeframe or cfg()["bar_interval"]
    # Map common aliases like "1min" → "1m" before you hand off to CCXT
    timeframe = _TIMEFRAME_ALIASES.get(timeframe, timeframe)

    # -------------------------------------------------------------------
    # If the caller did not specify an explicit date window, fall back to
    # the global defaults from params.yaml so every script can control the
    # history length purely via configuration.
    if start_dt is None:
        raw_start = cfg().get("start_date")
        if raw_start:
            start_dt = pd.to_datetime(raw_start, utc=True)

    if end_dt is None:
        raw_end = cfg().get("end_date")
        if raw_end:
            end_dt = pd.to_datetime(raw_end, utc=True)
    # -------------------------------------------------------------------

    # Check disk cache -------------------------------------------------------
    cache_path = cache_path_for(symbol, timeframe)
    if not force_refresh:
        cached = _read_cached(cache_path)
        if cached is not None:
            return cached

    # Else download afresh ---------------------------------------------------
    df = fetch_full_history(
        symbol,
        exchange_name,
        timeframe,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    _write_cached(df, cache_path)
    return df


def get_all_assets(force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Convenience helper: loop over every entry in ``assets`` and return a dict
    ``{symbol: DataFrame}``.
    """
    bar_interval = cfg_ns().bar_interval
    out: Dict[str, pd.DataFrame] = {}
    for asset in cfg_ns().assets:
        df = get_asset_df(
            asset["symbol"],
            exchange_name=asset["exchange"],
            timeframe=bar_interval,
            force_refresh=force_refresh,
        )
        out[asset["symbol"]] = df
    return out
