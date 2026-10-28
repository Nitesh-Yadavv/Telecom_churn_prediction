import pandas as pd
import pytest

from churn_core import config

BASE = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 12,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL", "OnlineSecurity": "Yes",
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "TotalCharges": "660.0", "Churn": "No",
}


@pytest.fixture()
def loader_parts():
    """(raw, split, scores) for 10 real customers (5 training, 5 test): raw rows are built to agree
    with the committed scores.csv and split_assignment.csv, so the loader's cross-checks can pass."""
    scores = pd.read_csv(config.reports_dir() / "scores.csv")
    scores = pd.concat([scores[scores.score_kind == "oof"].head(5),
                        scores[scores.score_kind == "test"].head(5)], ignore_index=True)
    split = pd.read_csv(config.data_dir() / "split_assignment.csv")
    rows = []
    for i, r in scores.iterrows():
        row = {**BASE, "customerID": r.customer_id, "MonthlyCharges": r.monthly_charges}
        if i == 0:
            row.update(tenure=0, TotalCharges=" ")            # a brand-new customer: blank TotalCharges
        if i == 1:
            row.update(SeniorCitizen=1, Churn="Yes")
        rows.append(row)
    return pd.DataFrame(rows), split, scores