"""
model.py
--------

XGBoost training and inference routines for the *Equity Factor Alpha*
project.  This module wraps the `xgboost.XGBRegressor` class to
provide a clean interface for fitting models on the feature matrix
produced by :func:`src.features.prepare_feature_matrix` and for
generating predictions.  Keeping model code separate from data and
features improves testability and makes it easy to swap in another
estimator (e.g. RandomForestRegressor) without touching the rest of
the pipeline.

The training function expects a `pandas.DataFrame` with a MultiIndex
(date, ticker) and numeric feature columns.  Index levels are dropped
before fitting because XGBoost does not consume them directly.  You
can supply hyperparameters via the `params` argument or rely on
defaults defined in `params.yaml`.  After training the returned
estimator can be used to predict on unseen data.

I chose XGBoost here because it strikes a balance between predictive
power and interpretability.  In my own work I’ve favoured models I can
explain to colleagues and management – a lesson I picked up during a
stint developing factor models for a commodity fund.  Although
neural networks might squeeze out a few extra basis points, they make
it harder to audit feature importance or reason about why a trade was
placed.  By contrast, tree‑based models naturally expose feature
gain, which aligns with my preference for morally compliant and
transparent decision making.  That said, the model itself does not
guarantee morally compliant outcomes – that responsibility lies with
the back‑tester.  However, using a transparent and widely adopted
algorithm like XGBoost helps ensure that predictions are based on
observable characteristics rather than opaque black‑box heuristics.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from math import sqrt

from .config import cfg

__all__ = ["train_model", "predict"]


def _prepare_xy(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert feature matrix and target series into numpy arrays.

    Drops any non‑numeric columns; index levels are discarded.  If the
    DataFrame contains object or category dtypes they are cast to
    floating point where possible.  Raises if any column cannot be
    coerced to numeric.
    """
    # Drop ticker/date index and ensure a flat table
    X_flat = X.copy().reset_index(drop=True)
    # Attempt to convert all columns to float
    X_numeric = X_flat.apply(pd.to_numeric, errors="raise")
    return X_numeric.to_numpy(), y.to_numpy()


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    params: Optional[Dict[str, float]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[XGBRegressor, float]:
    """
    Fit an XGBoost regressor on the provided features and targets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with MultiIndex (date, ticker) and numeric columns.
    y : pd.Series
        Target vector aligned to X.
    params : dict, optional
        Hyperparameters for XGBRegressor.  If None, defaults from
        ``params.yaml`` under the ``model`` key are used.
    test_size : float, default 0.2
        Fraction of the data to allocate to the validation split.  The
        random split ensures the model generalises and guards against
        overfitting.  Setting this to 0 disables splitting.
    random_state : int
        Seed for reproducible train/test splits.

    Returns
    -------
    (model, rmse) : (XGBRegressor, float)
        The fitted regressor and the root mean squared error on the
        validation set.  When ``test_size=0`` the RMSE will be `np.nan`.
    """
    if params is None:
        params = cfg().get("model", {})
    X_arr, y_arr = _prepare_xy(X, y)

    if test_size > 0.0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_arr, y_arr, test_size=test_size, random_state=random_state
        )
    else:
        X_train, y_train = X_arr, y_arr
        X_val, y_val = None, None

    model = XGBRegressor(
        **params,
    )

    model.fit(X_train, y_train)

    if X_val is not None:
        y_pred = model.predict(X_val)
        rmse = float(sqrt(mean_squared_error(y_val, y_pred)))
    else:
        rmse = float("nan")

    return model, rmse


def predict(model: XGBRegressor, X: pd.DataFrame) -> pd.Series:
    """
    Generate predictions for a trained model on the given features.

    The index of ``X`` is preserved on the returned Series so that
    predictions can be easily aligned with the original observations.
    """
    X_arr, _ = _prepare_xy(X, y=pd.Series([0] * len(X)))  # dummy y to satisfy signature
    preds = model.predict(X_arr)
    return pd.Series(preds, index=X.index, name="prediction")
