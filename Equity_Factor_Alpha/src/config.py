"""
config.py
---------

Central configuration loader for the *Equity Factor Alpha* project.

This module lazily parses the YAML configuration file exactly once
and caches the resulting dictionary for subsequent calls.  The
approach mirrors the pattern used in my Crypto Price Anomaly project
because it avoids repeated disk I/O and makes the configuration
accessible from any module.  Use :func:`cfg` to obtain a plain
`dict` and :func:`cfg_ns` for attribute access convenience.

Path resolution order
---------------------
1. Environment variable ``EFA_PARAMS_PATH`` (absolute or relative path)
2. Default ``params.yaml`` next to your repository root

In several roles I’ve needed to manage configuration across
development, test and production environments.  By honouring an
environment variable we make it easy to swap parameter files without
touching the code.  This pattern has saved me from accidentally
overwriting production data in the past.  Documenting it here helps
other users adopt the same safe practice.  Keeping configuration
externalised also supports moral compliance: one can review and
audit the YAML file independently of the code.

Examples
--------
>>> from src.config import cfg, cfg_ns
>>> tickers = [a["ticker"] for a in cfg()["assets"]]
>>> hold_days = cfg_ns().backtest["hold_days"]
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import yaml

__all__ = ["cfg", "cfg_ns", "to_dict", "reload"]

# --------------------------------------------------------------------------- #
# Internal constants
# --------------------------------------------------------------------------- #
_ENV_VAR: str = "EFA_PARAMS_PATH"
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
_DEFAULT_PATH: Path = _PROJECT_ROOT / "params.yaml"

# --------------------------------------------------------------------------- #
# Internal caches (populated lazily)
# --------------------------------------------------------------------------- #
_cfg_cache: Dict[str, Any] | None = None
_cfg_ns_cache: SimpleNamespace | None = None


def _resolve_yaml_path() -> Path:
    """Determine which YAML file to load based on environment or default."""
    override = os.getenv(_ENV_VAR)
    return Path(override).expanduser().resolve() if override else _DEFAULT_PATH


def _load_yaml() -> Dict[str, Any]:
    """Read ``params.yaml`` from disk and return it as a dict."""
    path = _resolve_yaml_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. Set {_ENV_VAR} or ensure {path.name} exists."
        )
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def cfg() -> Dict[str, Any]:
    """
    Return the cached configuration mapping.

    The first call triggers a disk read; subsequent calls reuse the
    cached object.  This means any modifications to the YAML file
    require calling :func:`reload` to see the changes.
    """
    global _cfg_cache
    if _cfg_cache is None:
        _cfg_cache = _load_yaml()
    return _cfg_cache


def cfg_ns() -> SimpleNamespace:
    """
    Return the configuration as a namespace for dot‑access convenience.

    You can still mutate nested dictionaries inside this namespace but
    remember that updates to the YAML file will not be visible until
    :func:`reload` is called.
    """
    global _cfg_ns_cache
    if _cfg_ns_cache is None:
        _cfg_ns_cache = SimpleNamespace(**cfg())
    return _cfg_ns_cache


def to_dict() -> Dict[str, Any]:
    """Alias for backwards compatibility – returns the config dict."""
    return cfg()


def reload() -> Dict[str, Any]:
    """
    Force a reload of the configuration from disk and return the fresh dict.

    This is handy in interactive sessions or tests where you tweak
    ``params.yaml`` and want the latest values without restarting
    Python.
    """
    global _cfg_cache, _cfg_ns_cache
    _cfg_cache = None
    _cfg_ns_cache = None
    return cfg()


def __getattr__(name: str):  # noqa: D401
    """
    Fallback attribute access to the underlying mapping.

    This allows you to import the config module directly and access
    keys as attributes:

        >>> import src.config as config
        >>> print(config.assets)
    """
    data = cfg()
    try:
        return data[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
