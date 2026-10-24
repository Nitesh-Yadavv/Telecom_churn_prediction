from typing import get_args

import pandas as pd
import pytest
from pydantic import ValidationError

from churn_core import config
from churn_core.schema import (
    CLEAN_FEATURE_COLUMNS,
    DEFAULT_GENDER,
    RAW_TO_CLEAN,
    Customer,
    clean_raw,
    customer_frame,
)

GOOD = {
    "senior_citizen": False, "partner": True, "dependents": False, "tenure": 12,
    "phone_service": True, "multiple_lines": "No", "internet_service": "DSL",
    "online_security": "Yes", "online_backup": "No", "device_protection": "No",
    "tech_support": "No", "streaming_tv": "No", "streaming_movies": "No",
    "contract": "Month-to-month", "paperless_billing": True,
    "payment_method": "Electronic check", "monthly_charges": 55.0, "total_charges": 660.0,
}

RAW_ROW = {
    "customerID": "T-1", "gender": "Male", "SeniorCitizen": 1, "Partner": "Yes", "Dependents": "No",
    "tenure": 0, "PhoneService": "No", "MultipleLines": "No phone service", "InternetService": "DSL",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Mailed check", "MonthlyCharges": 29.85,
    "TotalCharges": " ", "Churn": "Yes",
}


def test_allowed_values_match_the_saved_training_categories():
    """The hard-coded choices in Customer must equal what the model was trained on."""
    saved = config.reference_stats()["data_health"]["categorical"]
    for raw_col, clean_col in RAW_TO_CLEAN.items():
        field = Customer.model_fields.get(clean_col)
        if field is None or raw_col not in saved:
            continue
        choices = get_args(field.annotation)
        if choices:                                   # the text-valued columns
            assert set(choices) == set(saved[raw_col]), raw_col


def test_valid_customer_and_default_gender():
    c = Customer(**GOOD)
    assert c.gender == DEFAULT_GENDER
    frame = customer_frame([c])
    assert list(frame.columns) == CLEAN_FEATURE_COLUMNS and len(frame) == 1


@pytest.mark.parametrize("change", [
    {"contract": "Weekly"}, {"tenure": -1}, {"monthly_charges": -5.0},
    {"internet_service": "Cable"}, {"unknown_field": 1},
])
def test_invalid_customers_are_rejected(change):
    with pytest.raises(ValidationError):
        Customer(**{**GOOD, **change})


def test_clean_raw_applies_the_loader_rules():
    out = clean_raw(pd.DataFrame([RAW_ROW]))
    row = out.iloc[0]
    assert list(out.columns) == ["customer_id"] + CLEAN_FEATURE_COLUMNS + ["churn"]
    assert row.total_charges == 0                       # blank -> 0
    assert bool(row.senior_citizen) is True
    assert bool(row.partner) is True and bool(row.dependents) is False and bool(row.churn) is True
    assert row.contract == "Month-to-month"


def test_clean_raw_fills_gender_when_the_column_is_absent():
    raw = {k: v for k, v in RAW_ROW.items() if k not in ("gender", "Churn", "customerID")}
    out = clean_raw(pd.DataFrame([raw]))
    assert out.gender.iloc[0] == DEFAULT_GENDER and "churn" not in out.columns


def test_clean_raw_refuses_non_yes_no_values():
    with pytest.raises(ValueError):
        clean_raw(pd.DataFrame([{**RAW_ROW, "Partner": "Maybe"}]))