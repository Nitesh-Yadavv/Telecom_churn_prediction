"""
scripts/make_golden_predictions.py

Picks 8 test customers spread from the lowest to the highest score and saves their details with the
probability notebook 04 produced (reports/test_predictions.csv). tests/test_predict.py then checks
that churn_core reproduces those probabilities. The file is committed, so the check also runs in CI,
which has no access to the raw CSV.

Run once from the repo root:  python scripts/make_golden_predictions.py
"""
import json
from pathlib import Path

import pandas as pd

from churn_core import config
from churn_core.schema import clean_raw

QUANTILES = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1]

tp = pd.read_csv(config.reports_dir() / "test_predictions.csv").sort_values("p_cal").reset_index(drop=True)
picks = sorted({round(q * (len(tp) - 1)) for q in QUANTILES})
chosen = tp.loc[picks, ["customer_id", "p_cal"]]

raw = pd.read_csv(config.data_dir() / "Telco-Customer-Churn.csv").set_index("customerID")
clean = clean_raw(raw.loc[chosen.customer_id].reset_index())

rows = []
for (_, c), (_, r) in zip(chosen.iterrows(), clean.iterrows(), strict=True):
    record = {k: (v.item() if hasattr(v, "item") else v) for k, v in r.drop(["customer_id", "churn"], errors="ignore").items()}
    rows.append({"customer_id": c.customer_id, "expected_probability": float(c.p_cal), "customer": record})

out = Path(__file__).resolve().parents[1] / "tests" / "golden_predictions.json"
out.write_text(json.dumps({"source": "reports/test_predictions.csv (notebook 04)", "rows": rows}, indent=2))
print(f"Saved {len(rows)} golden customers to {out}")
for row in rows:
    print(f"  {row['customer_id']}  p = {row['expected_probability']:.4f}")