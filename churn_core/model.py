"""

Loads the saved models ONCE and exposes probabilities. HEAVY (scikit-learn, LightGBM): API only.

  calibrated model  = the one whose probabilities are real-world probabilities (used for decisions)
  raw model         = the uncalibrated LightGBM (used only to explain a prediction, see explain.py)
"""
import warnings
from functools import cache
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd

from churn_core import config

PICKLE_SENSITIVE = ("scikit-learn", "lightgbm")     # versions that must match what trained the models


def version_mismatches() -> dict:
    """{library: (trained with, installed)} for libraries whose version differs from training."""
    trained = config.model_config().get("library_versions", {})
    out = {}
    for lib in PICKLE_SENSITIVE:
        if lib in trained and trained[lib] != version(lib):
            out[lib] = (trained[lib], version(lib))
    return out


@cache
def load_models() -> dict:
    for lib, (trained, installed) in version_mismatches().items():
        warnings.warn(f"models were trained with {lib} {trained}, installed is {installed}", stacklevel=2)
    d = config.models_dir()
    return {"calibrated": joblib.load(d / "churn_model_calibrated.joblib"),
            "raw": joblib.load(d / "lgbm_raw.joblib")}


def predict_proba_calibrated(features: pd.DataFrame) -> np.ndarray:
    """Calibrated probability of churn, one per row of the feature matrix."""
    return load_models()["calibrated"].predict_proba(features)[:, 1]


def predict_proba_raw(features: pd.DataFrame) -> np.ndarray:
    """Uncalibrated probability (for explanations only, never for money decisions)."""
    return load_models()["raw"].predict_proba(features)[:, 1]