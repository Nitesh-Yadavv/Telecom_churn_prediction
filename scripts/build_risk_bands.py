"""

Writes models/risk_bands.json and reports/band_table.csv.

  * cutoff edge     = the cost-derived break-even probability (not tuned)
  * other two edges = quantiles of the TEST-set scores (scores only, no outcomes used), rounded
  * the table       = observed churn rate per band, on the test set (headline) and on the
                      out-of-fold scores of the training customers (a second, independent check)

Run from the repo root (with `pip install -e .` done):  python scripts/build_risk_bands.py
"""
import json

import numpy as np
import pandas as pd

from churn_core import config
from churn_core.bands import LABELS, assign_band

LOW_MEDIUM_QUANTILE = 0.40      # the lowest 40% of customers by score are "Low"
VERY_HIGH_QUANTILE = 0.85       # the top 15% by score are "Very high"

test = pd.read_csv(config.reports_dir() / "test_predictions.csv")
scores = pd.read_csv(config.reports_dir() / "scores.csv")
oof = scores[scores.score_kind == "oof"]

cutoff = config.default_cutoff()
low_medium = round(float(np.quantile(test.p_cal, LOW_MEDIUM_QUANTILE)), 2)
very_high = round(float(np.quantile(test.p_cal, VERY_HIGH_QUANTILE)), 2)
edges = (low_medium, cutoff, very_high)
assert low_medium < cutoff < very_high, f"edges not increasing: {edges}"


def table(df: pd.DataFrame) -> list:
    band = assign_band(df.p_cal.to_numpy(), edges)
    g = df.assign(band=band).groupby("band").agg(
        customers=("p_cal", "size"), observed_churn_rate=("actual_churn", "mean"),
        mean_predicted=("p_cal", "mean")).reindex(LABELS)
    g["share"] = g.customers / g.customers.sum()
    return [{"band": b, "customers": int(r.customers), "share": round(float(r.share), 4),
             "observed_churn_rate": round(float(r.observed_churn_rate), 4),
             "mean_predicted": round(float(r.mean_predicted), 4)} for b, r in g.iterrows()]


observed = {"test": table(test), "oof": table(oof)}

a = config.default_assumptions()
out = {
    "labels": list(LABELS),
    "edges": {"low_medium": low_medium, "cutoff": cutoff, "high_very_high": very_high},
    "quantiles_used": {"low_medium": LOW_MEDIUM_QUANTILE, "high_very_high": VERY_HIGH_QUANTILE},
    "assumptions": {"r": a.r, "d": a.d, "months": a.months},
    "note": "Cutoff edge derived from the business assumptions. Other edges are quantiles of "
            "test-set scores (no outcomes used). 'oof' = out-of-fold scores of training customers.",
    "observed": observed,
}
(config.models_dir() / "risk_bands.json").write_text(json.dumps(out, indent=2))

rows = [{"sample": s, **r} for s, rs in observed.items() for r in rs]
pd.DataFrame(rows).to_csv(config.reports_dir() / "band_table.csv", index=False)

print(f"Edges: low/medium {low_medium}, cutoff {cutoff:.4f}, high/very-high {very_high}")
for s in ("test", "oof"):
    print(f"\n{s}:")
    print(pd.DataFrame(observed[s]).to_string(index=False))