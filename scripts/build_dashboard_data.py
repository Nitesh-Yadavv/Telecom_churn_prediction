"""
scripts/build_dashboard_data.py

Writes reports/dashboard_data.csv: one row per customer with everything the dashboard pages need, so
they read one small file (about 1 MB) instead of the database or the raw CSV:
the score and band, the real outcome, the bill, and the segment columns (contract, tenure band,
internet service, payment method, gender, senior citizen).

Run from the repo root:  python scripts/build_dashboard_data.py
"""
import numpy as np
import pandas as pd

from churn_core import bands, config
from churn_core.schema import clean_raw

raw = clean_raw(pd.read_csv(config.data_dir() / "Telco-Customer-Churn.csv"))
scores = pd.read_csv(config.reports_dir() / "scores.csv")

df = scores.merge(raw, on="customer_id", how="inner", suffixes=("", "_raw"), validate="one_to_one")
assert len(df) == len(scores) == len(raw), "customers in scores.csv and the raw CSV do not match"
assert (df["actual_churn"] == df["churn"].astype(int)).all(), "outcomes in scores.csv differ from the raw CSV"
assert np.allclose(df["monthly_charges"], df["monthly_charges_raw"], atol=0.005), "monthly charges differ"

out = pd.DataFrame({
    "customer_id": df["customer_id"],
    "split": df["split"],
    "score_kind": df["score_kind"],
    "probability": df["p_cal"],
    "band": bands.assign_band(df["p_cal"].to_numpy()),
    "flag": (df["p_cal"] >= config.default_cutoff()).astype(int),
    "actual_churn": df["actual_churn"],
    "monthly_charges": df["monthly_charges"],
    "tenure": df["tenure"],
    "tenure_band": pd.cut(df["tenure"], [-1, 11, 35, 72], labels=["0-11", "12-35", "36-72"]).astype(str),
    "contract": df["contract"],
    "internet_service": df["internet_service"],
    "payment_method": df["payment_method"],
    "gender": df["gender"],
    "senior_citizen": df["senior_citizen"].astype(int),
})
out.to_csv(config.reports_dir() / "dashboard_data.csv", index=False)
test = out[out["split"] == "test"]
print(f"Saved {len(out)} rows. Test customers: {len(test)}, flagged: {int(test['flag'].sum())}, "
      f"churn rate {test['actual_churn'].mean():.3f}")