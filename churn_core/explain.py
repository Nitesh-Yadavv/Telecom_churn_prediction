"""
Plain-language reasons for a prediction. HEAVY (uses the model): API only.

How it works
  * LightGBM can return each feature's contribution to one prediction (SHAP-style values) with
    predict(..., pred_contrib=True), so the API does not need the shap package. A test checks the
    values equal shap's TreeExplainer.
  * Contributions are in the RAW model's log-odds, not in calibrated probability. They say which
    features pushed this customer's risk up or down, and by how much relative to each other.
  * Columns that describe the same fact are added together (the three contract columns become one
    "Contract type" driver), using the same grouping as the SHAP importance plot in notebook 04.

The sentences describe what the model uses. They are NOT causes: nothing here says that changing
a feature would change whether the customer leaves. Retention levers are untested hypotheses.
"""
import numpy as np
import pandas as pd

from churn_core import model
from churn_core.preprocess import build_features

# group key -> (label shown to the user, customer field that holds the value to display)
GROUP_INFO = {
    "internet": ("Internet service", "internet_service"),
    "contract": ("Contract type", "contract"),
    "payment_method": ("Payment method", "payment_method"),
    "phone_service": ("Phone service", "phone_service"),
    "senior_citizen": ("Senior citizen", "senior_citizen"),
    "partner": ("Has a partner", "partner"),
    "dependents": ("Has dependents", "dependents"),
    "tenure": ("Time as a customer", "tenure"),
    "paperless_billing": ("Paperless billing", "paperless_billing"),
    "monthly_charges": ("Monthly bill", "monthly_charges"),
    "total_charges": ("Total paid so far", "total_charges"),
    "gender_Male": ("Gender", "gender"),
    "multiple_lines_Yes": ("Multiple phone lines", "multiple_lines"),
    "online_security_Yes": ("Online security", "online_security"),
    "online_backup_Yes": ("Online backup", "online_backup"),
    "device_protection_Yes": ("Device protection", "device_protection"),
    "tech_support_Yes": ("Tech support", "tech_support"),
    "streaming_tv_Yes": ("Streaming TV", "streaming_tv"),
    "streaming_movies_Yes": ("Streaming movies", "streaming_movies"),
}

# Shown only next to a driver that RAISES risk. Hypotheses to test, not recommendations.
LEVERS = {
    "contract": "Offer a discount for a longer-term contract",
    "payment_method": "Encourage automatic payment",
    "internet": "Review price and service quality for this internet plan",
    "tech_support_Yes": "Offer a tech-support trial",
    "online_security_Yes": "Offer an online-security trial",
    "tenure": "Add an early-life check-in call",
    "monthly_charges": "Review whether the plan fits this customer's usage and price",
}
LEVER_STATUS = "untested hypothesis"
NOTE = ("Effects are in the raw model's log-odds and show which features pushed this customer's risk "
        "up or down. They describe the model, not causes. The probability shown elsewhere is calibrated.")


def group_columns(columns) -> dict:
    """Group feature columns that describe one fact (same grouping as notebook 04's SHAP plot)."""
    cols = list(columns)
    no_internet = ["internet_service_No"] + [c for c in cols if c.endswith("No internet service")]
    groups = {
        "internet": ["internet_service_Fiber optic"] + no_internet,
        "contract": [c for c in cols if c.startswith("contract_")],
        "payment_method": [c for c in cols if c.startswith("payment_method_")],
        "phone_service": ["phone_service", "multiple_lines_No phone service"],
    }
    used = {c for members in groups.values() for c in members}
    groups.update({c: [c] for c in cols if c not in used})
    return groups


def contributions(features: pd.DataFrame) -> tuple:
    """(per-feature contributions, base value), both in raw log-odds. One row per customer."""
    booster = model.load_models()["raw"].booster_
    contrib = booster.predict(features, pred_contrib=True)      # last column = the base value
    return contrib[:, :-1], contrib[:, -1]


def _format_value(field: str, value) -> str:
    if field == "tenure":
        return f"{int(value)} months"
    if field in ("monthly_charges", "total_charges"):
        return f"{float(value):,.2f}"
    if isinstance(value, bool | np.bool_):
        return "Yes" if value else "No"
    return str(value)


def explain(df: pd.DataFrame, top_k: int = 3) -> list:
    """For each CLEAN-format customer: the top drivers in plain language and the full waterfall."""
    features = build_features(df)
    contrib, base = contributions(features)
    cols = list(features.columns)
    groups = group_columns(cols)
    grouped = {g: contrib[:, [cols.index(c) for c in members]].sum(axis=1) for g, members in groups.items()}

    results = []
    for i in range(len(df)):
        row = df.iloc[i]
        items = []
        for g, effects in grouped.items():
            label, field = GROUP_INFO[g]
            items.append({"feature": g, "label": label, "value": _format_value(field, row[field]),
                          "effect": float(effects[i])})
        items.sort(key=lambda it: abs(it["effect"]), reverse=True)
        waterfall = [it for it in items if abs(it["effect"]) > 1e-12]

        drivers = []
        for it in waterfall[:top_k]:
            raises = it["effect"] > 0
            drivers.append({
                **it,
                "direction": "raises" if raises else "lowers",
                "sentence": f"{it['label']} ({it['value']}) {'raises' if raises else 'lowers'} the risk",
                "lever": ({"text": LEVERS[it["feature"]], "status": LEVER_STATUS}
                          if raises and it["feature"] in LEVERS else None),
            })
        results.append({"base_logit": float(base[i]), "raw_logit": float(base[i] + sum(g[i] for g in grouped.values())),
                        "drivers": drivers, "waterfall": waterfall, "note": NOTE})
    return results