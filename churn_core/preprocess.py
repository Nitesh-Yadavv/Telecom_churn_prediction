"""

Turns CLEAN-format customers (see schema.py) into the exact feature matrix the model was trained
on. This is a direct copy of build_features() from notebook 03, plus guards. HEAVY: it loads the
saved scikit-learn encoder, so only the API imports it (never the dashboard).

Train/serve skew lives here: if the app encodes a customer even slightly differently from
notebook 03, predictions go wrong silently. tests/test_preprocess.py checks the output against the
saved training and test matrices.
"""
import warnings
from functools import cache

import joblib
import pandas as pd
import sklearn

from churn_core import config
from churn_core.schema import CLEAN_FEATURE_COLUMNS, Customer, customer_frame


@cache
def load_bundle() -> dict:
    """The encoder bundle saved by notebook 03 (encoder, column groups, feature order)."""
    bundle = joblib.load(config.models_dir() / "encoder.joblib")
    saved = bundle.get("sklearn_version")
    if saved and saved != sklearn.__version__:
        warnings.warn(f"encoder.joblib was saved with scikit-learn {saved}, "
                      f"this environment has {sklearn.__version__}", stacklevel=2)
    if list(bundle["feature_order"]) != config.feature_order():
        raise RuntimeError("encoder.joblib and model_config.json disagree on the feature order")
    return bundle


def _require_known_categories(raw: pd.DataFrame, bundle: dict) -> None:
    """The encoder would silently turn an unseen category into all zeros. Refuse instead."""
    for col, known in zip(bundle["multi_cat_cols"], bundle["encoder"].categories_, strict=True):
        unseen = sorted(set(raw[col].unique()) - set(known))
        if unseen:
            raise ValueError(f"{col}: unseen value(s) {unseen}; allowed {sorted(known)}")


def build_features(df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """CLEAN-format customers -> model feature matrix, columns in the trained order."""
    b = bundle or load_bundle()
    missing = [c for c in CLEAN_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing column(s): {missing}")
    raw = df[CLEAN_FEATURE_COLUMNS].reset_index(drop=True)
    _require_known_categories(raw, b)

    enc = b["encoder"]
    binary = raw[b["binary_cols"]].astype(int)
    numeric = raw[b["numeric_cols"]].astype(float)
    cats = pd.DataFrame(enc.transform(raw[b["multi_cat_cols"]]),
                        columns=enc.get_feature_names_out(b["multi_cat_cols"]))
    out = pd.concat([binary, numeric, cats], axis=1)[b["feature_order"]]
    if out.isna().any().any():
        raise ValueError("features contain missing values")
    return out


def encode_customers(customers: list[Customer]) -> pd.DataFrame:
    """Validated Customer objects -> model feature matrix."""
    return build_features(customer_frame(customers))