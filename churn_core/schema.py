"""
What a customer looks like, in two formats, and how to convert between them.

  RAW format   = the Kaggle Telco CSV (CamelCase names, Yes/No text, TotalCharges blank for new
                 customers). This is the upload-template format.
  CLEAN format = what notebook 03 and the database use (snake_case, booleans, TotalCharges 0).
                 The encoder and the model expect this format.

clean_raw() is the ONE cleaning function. Phase 5 makes db/load_data.py call it too, so the
database and the app can never clean the data differently.

Light on purpose: pandas and pydantic only (no model libraries).
"""
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_GENDER = "Female"   # the encoder's reference category; gender has zero effect on predictions

RAW_TO_CLEAN = {
    "customerID": "customer_id", "gender": "gender", "SeniorCitizen": "senior_citizen",
    "Partner": "partner", "Dependents": "dependents", "tenure": "tenure",
    "PhoneService": "phone_service", "MultipleLines": "multiple_lines",
    "InternetService": "internet_service", "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup", "DeviceProtection": "device_protection",
    "TechSupport": "tech_support", "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies", "Contract": "contract",
    "PaperlessBilling": "paperless_billing", "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges", "TotalCharges": "total_charges", "Churn": "churn",
}
CLEAN_TO_RAW = {v: k for k, v in RAW_TO_CLEAN.items()}

# The 19 input columns of the model, in the order the database and notebook 03 use.
CLEAN_FEATURE_COLUMNS = [
    "gender", "senior_citizen", "partner", "dependents", "tenure", "phone_service",
    "multiple_lines", "internet_service", "online_security", "online_backup",
    "device_protection", "tech_support", "streaming_tv", "streaming_movies", "contract",
    "paperless_billing", "payment_method", "monthly_charges", "total_charges",
]
YES_NO_COLUMNS = ["partner", "dependents", "phone_service", "paperless_billing"]   # raw Yes/No -> bool
NUMERIC_COLUMNS = ["tenure", "monthly_charges", "total_charges"]
OPTIONAL_RAW_COLUMNS = {"gender"}     # the model ignores gender; a default is filled in if absent
INTERNET_SERVICE_COLUMNS = ["online_security", "online_backup", "device_protection",
                            "tech_support", "streaming_tv", "streaming_movies"]

_ServiceChoice = Literal["No", "No internet service", "Yes"]


class Customer(BaseModel):
    """One customer in CLEAN format, as the API receives it. Unknown fields are rejected."""
    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"] = DEFAULT_GENDER
    senior_citizen: bool
    partner: bool
    dependents: bool
    tenure: int = Field(ge=0)
    phone_service: bool
    multiple_lines: Literal["No", "No phone service", "Yes"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: _ServiceChoice
    online_backup: _ServiceChoice
    device_protection: _ServiceChoice
    tech_support: _ServiceChoice
    streaming_tv: _ServiceChoice
    streaming_movies: _ServiceChoice
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: bool
    payment_method: Literal["Bank transfer (automatic)", "Credit card (automatic)",
                            "Electronic check", "Mailed check"]
    monthly_charges: float = Field(ge=0)
    total_charges: float = Field(ge=0)


def customer_frame(customers: list[Customer]) -> pd.DataFrame:
    """Customers as a CLEAN-format DataFrame with the columns in the canonical order."""
    return pd.DataFrame([c.model_dump() for c in customers])[CLEAN_FEATURE_COLUMNS]


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Kaggle-format frame -> CLEAN format. Same rules as the original database loader:
    blank TotalCharges -> 0, Yes/No -> bool, SeniorCitizen 0/1 -> bool.
    Run checks.check_upload() first: values that are not Yes/No raise an error here."""
    out = df.rename(columns=RAW_TO_CLEAN).copy()
    if "gender" not in out.columns:
        out["gender"] = DEFAULT_GENDER

    out["total_charges"] = pd.to_numeric(out["total_charges"], errors="coerce").fillna(0)
    out["senior_citizen"] = pd.to_numeric(out["senior_citizen"]).astype(int).astype(bool)

    for col in YES_NO_COLUMNS + (["churn"] if "churn" in out.columns else []):
        mapped = out[col].map({"Yes": True, "No": False})
        if mapped.isna().any():
            raise ValueError(f"Column {col} must contain only Yes/No; run check_upload() first")
        out[col] = mapped

    keep = [c for c in ["customer_id"] if c in out.columns] + CLEAN_FEATURE_COLUMNS
    keep += [c for c in ["churn"] if c in out.columns]
    return out[keep]