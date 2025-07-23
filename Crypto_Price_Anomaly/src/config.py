"""
config.py
---------

Central configuration loader for the *Crypto Price Anomaly* project.

The module lazily parses ``params.yaml`` **once** and caches the resulting
dictionary for all subsequent calls.

Path resolution order
---------------------
1. Environment variable ``CPA_PARAMS_PATH`` (absolute or relative path).
2. Default ``<project‑root>/params.yaml`` next to your Git repository root.

Public API
~~~~~~~~~~
- ``cfg()``        → returns the cached dict
- ``cfg_ns()``     → returns a :class:`types.SimpleNamespace`
- ``to_dict()``    → alias for :pyfunc:`cfg`
- ``reload()``     → bust the cache and re‑read the file

Examples
--------
>>> from src.config import cfg, cfg_ns
>>> symbols = [a["symbol"] for a in cfg()["assets"]]
>>> interval = cfg_ns().bar_interval
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
_ENV_VAR: str = "CPA_PARAMS_PATH"
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
_DEFAULT_PATH: Path = _PROJECT_ROOT / "params.yaml"

# --------------------------------------------------------------------------- #
# Internal caches (populated lazily)
# --------------------------------------------------------------------------- #
_cfg_cache: Dict[str, Any] | None = None
_cfg_ns_cache: SimpleNamespace | None = None


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def _resolve_yaml_path() -> Path:
    """
    Determine which YAML file to load.

    Environment variable wins; otherwise fall back to the default path.
    """
    override = os.getenv(_ENV_VAR)
    return Path(override).expanduser().resolve() if override else _DEFAULT_PATH


def _load_yaml() -> Dict[str, Any]:
    """Read ``params.yaml`` from disk and return it as a dict."""
    path = _resolve_yaml_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. "
            f"Set {_ENV_VAR} or ensure {_DEFAULT_PATH.name} exists."
        )
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def cfg() -> Dict[str, Any]:
    """
    Return the cached configuration mapping.

    The first call triggers a disk read; subsequent calls are cheap.
    """
    global _cfg_cache
    if _cfg_cache is None:
        _cfg_cache = _load_yaml()
    return _cfg_cache


def cfg_ns() -> SimpleNamespace:
    """
    Return the configuration as a namespace for dot‑access convenience.

    Example
    -------
    >>> from src.config import cfg_ns
    >>> print(cfg_ns().bar_interval)
    '30m'
    """
    global _cfg_ns_cache
    if _cfg_ns_cache is None:
        _cfg_ns_cache = SimpleNamespace(**cfg())
    return _cfg_ns_cache


def to_dict() -> Dict[str, Any]:
    """Alias kept for backwards compatibility with earlier drafts."""
    return cfg()


def reload() -> Dict[str, Any]:
    """
    Force a reload of the configuration from disk and return the fresh dict.

    Useful inside tests or notebooks where you tweak ``params.yaml`` on the fly.
    """
    global _cfg_cache, _cfg_ns_cache
    _cfg_cache = None
    _cfg_ns_cache = None
    return cfg()


# --------------------------------------------------------------------------- #
# Module‑level getattr magic
# --------------------------------------------------------------------------- #
def __getattr__(name: str):  # noqa: D401, PLW012
    """
    Fallback attribute access to the underlying mapping.

    This lets you do ``import src.config as config`` and access keys directly:

    >>> import src.config as config
    >>> print(config.bar_interval)
    """
    data = cfg()
    try:
        return data[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
