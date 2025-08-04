"""
model.py
--------

Lightweight wrapper around ML classifiers used for signal generation.

The design intentionally mirrors the rest of the project: configuration is
supplied via ``params.yaml`` and passed in as a simple ``dict``.  The current
implementation supports an ``xgboost.XGBClassifier`` but the class structure
is kept general so that a different model (neural network, random forest …)
can be dropped in later without touching the calling code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

try:  # Lazy import so the project works without ML extras
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "xgboost is required for ML mode; install via `pip install xgboost`"
    ) from exc


@dataclass
class MLModel:
    """Simple wrapper that trains a classifier and returns trade signals.

    Parameters
    ----------
    cfg : dict
        Sub-tree from ``params.yaml`` under the ``model`` key.  Expected keys:

        ``type`` : str, default ``'xgboost'``
            Identifier of the underlying model.  Only ``xgboost`` is currently
            implemented but the attribute allows future extension.

        ``features`` : list[str]
            Column names to feed into the classifier.  If omitted, the model
            will default to a minimal feature set based on the engineered
            columns present in the DataFrame.

        ``decision_threshold`` : float, default ``0.5``
            Probability threshold above which we emit a long signal.

        ``params`` : dict
            Keyword arguments passed to the underlying estimator.
    """

    cfg: dict

    def __post_init__(self) -> None:
        model_type = self.cfg.get("type", "xgboost").lower()
        params = self.cfg.get("params", {})
        if model_type == "xgboost":
            self.model = XGBClassifier(**params)
        else:  # pragma: no cover - defensive for future models
            raise ValueError(f"Unsupported model type: {model_type!r}")

        self.features: Sequence[str] = self.cfg.get("features", [])
        self.threshold: float = float(self.cfg.get("decision_threshold", 0.5))

    # ------------------------------------------------------------------
    def _default_features(self, df: pd.DataFrame) -> list[str]:
        lookback = self.cfg.get("lookback_bars", 96)
        candidates = [
            "zscore",
            "log_return",
            f"log_return_mean_{lookback}",
            f"log_return_std_{lookback}",
        ]
        return [c for c in candidates if c in df.columns]

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """Train the classifier on historical data and return boolean signals.

        The current policy is deliberately simple: we treat ``log_return`` one
        bar ahead as the binary target (positive → 1, otherwise 0) and fit the
        classifier once on the whole dataset.  Predictions are probabilities of
        the positive class; those exceeding ``decision_threshold`` yield a long
        signal.  This mirrors the long-only, flat-sizing philosophy elsewhere in
        the project.
        """

        feature_cols = list(self.features) or self._default_features(df)
        if not feature_cols:
            raise ValueError("No feature columns available for ML model")

        # Build training frame: drop rows with missing features or target
        train_df = df.copy()
        train_df["_target"] = (train_df["log_return"].shift(-1) > 0).astype(int)
        train_df = train_df.dropna(subset=feature_cols + ["_target"])

        X = train_df[feature_cols]
        y = train_df["_target"]
        self.model.fit(X, y)

        # Predict on the full dataset; fill missing feature rows with zeros
        probs = self.model.predict_proba(df[feature_cols].fillna(0.0))[:, 1]
        signal = probs >= self.threshold
        return pd.Series(signal, index=df.index)
