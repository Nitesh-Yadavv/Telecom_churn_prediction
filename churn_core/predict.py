"""
The one function the API calls: CLEAN-format customers in, a decision table out.
HEAVY (loads the model); the dashboard never imports this.
"""
import pandas as pd

from churn_core import bands, config, economics, model
from churn_core.preprocess import build_features
from churn_core.schema import Customer, customer_frame


def predict(df: pd.DataFrame, r: float | None = None, d: float | None = None,
            months: float | None = None) -> pd.DataFrame:
    """Score CLEAN-format customers.

    Returns one row per input row (same index) with:
      probability     calibrated chance of churning
      band            Low / Medium / High / Very high
      flag            True if the campaign would contact them (probability >= cutoff for r and d)
      expected_gain   expected net value of contacting them over `months`, in the bill's currency
      model_version

    r, d, months default to the saved assumptions (0.5, 0.2, 12). They are assumptions, not facts."""
    a = config.default_assumptions()
    r = a.r if r is None else r
    d = a.d if d is None else d
    months = a.months if months is None else months

    p = model.predict_proba_calibrated(build_features(df))
    bill = df["monthly_charges"].to_numpy(dtype=float)
    return pd.DataFrame({
        "probability": p,
        "band": bands.assign_band(p),
        "flag": p >= economics.cutoff_from(r, d),
        "expected_gain": economics.ev_per_customer(p, bill, r, d, months),
        "model_version": config.MODEL_VERSION,
    }, index=df.index)


def predict_customers(customers: list[Customer], **assumptions) -> pd.DataFrame:
    """Validated Customer objects -> decision table."""
    return predict(customer_frame(customers), **assumptions)