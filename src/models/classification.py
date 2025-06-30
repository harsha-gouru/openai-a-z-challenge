#!/usr/bin/env python
"""
Amazon Deep Insights – Classification helpers
============================================

Utility functions for **training, evaluating and persisting** machine-learning
classifiers that label LiDAR-derived features.  Typical use-cases:

* Classifying candidate *archaeological* objects detected in a Digital Elevation
  Model (mound / not-mound, geoglyph shape, etc.)
* Differentiating *vegetation* strata (low/medium/high canopy) from engineered
  features using CHM metrics
* Distinguishing *terrain types* (floodplain, terrace, plateau) based on slope /
  curvature descriptors

The API purposefully stays **simple** so it can be called from notebooks or
batch pipelines without deep ML boiler-plate.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Literal, Any

import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import ClassifierMixin

try:  # xgboost is optional – fall back gracefully
    from xgboost import XGBClassifier  # type: ignore

    _HAS_XGB = True
except ImportError:  # pragma: no cover
    _HAS_XGB = False

# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #


def prepare_data(
    df: pd.DataFrame,
    label_col: str,
    *,
    test_size: float = 0.2,
    random_state: int | None = 42,
    drop_na: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a *feature* dataframe into **train/test** matrices.

    Parameters
    ----------
    df : DataFrame
        Source dataframe containing predictors **and** the label column.
    label_col : str
        Name of the target column.
    test_size : float, default 0.2
        Fraction of samples reserved for testing.
    random_state : int | None, default 42
        Reproducibility seed.
    drop_na : bool, default True
        Drop rows with missing values before split.
    """

    if drop_na:
        df = df.dropna(subset=[label_col])

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe")

    X = df.drop(columns=[label_col]).values
    y = df[label_col].values

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# --------------------------------------------------------------------------- #
# Model training
# --------------------------------------------------------------------------- #

ModelType = Literal["random_forest", "gbt", "xgboost"]


def build_classifier(
    model_type: ModelType = "random_forest",
    **kwargs: Any,
) -> ClassifierMixin:
    """
    Factory for a scikit-learn compatible classifier.

    Supported `model_type`s
    ----------------------
    random_forest : sklearn.ensemble.RandomForestClassifier  (default)
    gbt           : sklearn.ensemble.GradientBoostingClassifier
    xgboost       : xgboost.XGBClassifier  (if xgboost is installed)
    """

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            class_weight="balanced",
            **kwargs,
        )

    if model_type == "gbt":
        return GradientBoostingClassifier(**kwargs)

    if model_type == "xgboost":
        if not _HAS_XGB:
            raise ImportError("xgboost is not installed – install `xgboost` or use another model_type")
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            objective="binary:logistic",
            n_jobs=-1,
            **kwargs,
        )

    raise ValueError(f"Unsupported model_type '{model_type}'")


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    model_type: ModelType = "random_forest",
    **kwargs: Any,
) -> ClassifierMixin:
    """
    Fit a classifier on training data and return the trained model.
    """

    model = build_classifier(model_type, **kwargs)
    model.fit(X_train, y_train)
    return model


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def evaluate_classifier(
    model: ClassifierMixin,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    average: str = "weighted",
) -> Dict[str, float]:
    """
    Compute common classification metrics on a hold-out set.
    """

    preds = model.predict(X_test)

    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average=average, zero_division=0),
        "recall": recall_score(y_test, preds, average=average, zero_division=0),
        "f1": f1_score(y_test, preds, average=average, zero_division=0),
    }
    return metrics


def cross_validate_classifier(
    model: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv: int = 5,
    scoring: str | List[str] = "f1_weighted",
    n_jobs: int = -1,
) -> Dict[str, float]:
    """
    Run k-fold cross-validation and return average scores.
    """

    if isinstance(scoring, str):
        scoring = [scoring]

    cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, return_train_score=False)
    return {score: float(np.mean(cv_res[f"test_{score}"])) for score in scoring}


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #


try:
    import joblib

    def save_model(model: ClassifierMixin, path: str) -> None:
        """Persist a trained model to disk."""

        joblib.dump(model, path)

    def load_model(path: str) -> ClassifierMixin:
        """Load a previously saved model."""

        return joblib.load(path)

except ImportError:  # pragma: no cover

    def save_model(model: ClassifierMixin, path: str) -> None:  # type: ignore[override]
        warnings.warn("joblib not installed – model will not be saved")

    def load_model(path: str) -> ClassifierMixin:  # type: ignore[override]
        raise ImportError("joblib not installed – cannot load model")


# --------------------------------------------------------------------------- #
# Public exports
# --------------------------------------------------------------------------- #

__all__ = [
    "prepare_data",
    "build_classifier",
    "train_classifier",
    "evaluate_classifier",
    "cross_validate_classifier",
    "save_model",
    "load_model",
]

