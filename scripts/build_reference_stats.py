"""
scripts/build_reference_stats.py

Adds a "data_health" section to models/reference_stats.json: the allowed values for every
categorical column and the observed ranges for every numeric column, learned from the TRAINING
rows only. Existing keys in the file are kept untouched.

Used later by churn_core/checks.py to flag problems in uploaded files (unseen category,
out-of-range value, missing column) before they reach the model.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF_PATH = ROOT / "models" / "reference_stats.json"

raw = pd.read_csv(ROOT / "data" / "Telco-Customer-Churn.csv")
split = pd.read_csv(ROOT / "data" / "split_assignment.csv")
train_ids = set(split.loc[split.split == "train", "customer_id"])

raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce")   # blanks -> NaN
train = raw[raw.customerID.isin(train_ids)]
assert len(train) == 5634, len(train)

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
               "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
               "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]

health = {
    "format": "Kaggle Telco column names (the upload template format)",
    "learned_from": "training rows only (5,634)",
    "required_columns": CATEGORICAL + NUMERIC,
    "categorical": {c: sorted(train[c].dropna().astype(str).unique().tolist()) for c in CATEGORICAL},
    "numeric": {
        c: {
            "min": float(train[c].min()),
            "max": float(train[c].max()),
            "p01": float(train[c].quantile(0.01)),
            "p99": float(train[c].quantile(0.99)),
            "mean": float(train[c].mean()),
            "missing_share": float(train[c].isna().mean()),
        }
        for c in NUMERIC
    },
    "notes": {
        "TotalCharges": "Blank for brand-new customers (tenure 0). The cleaning step fills these with 0, "
                        "so a blank is acceptable only when tenure is 0.",
    },
}

ref = json.loads(REF_PATH.read_text())
ref["data_health"] = health
REF_PATH.write_text(json.dumps(ref, indent=2))

# ---- Report: how do the HELD-OUT rows look against limits learned from training? ----
test = raw[~raw.customerID.isin(train_ids)]
unseen = {c: sorted(set(test[c].dropna().astype(str)) - set(health["categorical"][c])) for c in CATEGORICAL}
unseen = {c: v for c, v in unseen.items() if v}
out_of_range = {c: int(((test[c] < health["numeric"][c]["min"]) | (test[c] > health["numeric"][c]["max"])).sum())
                for c in NUMERIC}
print("Saved data_health to", REF_PATH)
print("Categorical columns:", len(CATEGORICAL), "| numeric columns:", len(NUMERIC))
print("Test rows with unseen categories:", unseen or "none")
print("Test rows outside the training min-max:", out_of_range)
for c in CATEGORICAL:
    print(f"  {c:18s}", health["categorical"][c])