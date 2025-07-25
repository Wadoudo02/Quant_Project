"""
Equity Factor Alpha package.

This package contains modules for configuration, data loading,
feature engineering, model training, backtesting and plotting.
Use `from .config import cfg` to access the YAML configuration.
"""

from .config import cfg, cfg_ns, reload

__all__ = ["cfg", "cfg_ns", "reload"]
