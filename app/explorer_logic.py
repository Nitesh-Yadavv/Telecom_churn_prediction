"""
app/explorer_logic.py

Pure business and data processing functions for the Customer Risk Explorer.
Includes customer filtering, expected gain ranking, spreadsheet formula injection defense,
and batch scoring output formatting.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from churn_core.economics import cutoff_from, ev_per_customer

DANGEROUS_CSV_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def neutralise_value(val: Any) -> Any:
    """Prepend a single quote if string starts with spreadsheet formula characters."""
    if isinstance(val, str) and len(val) > 0 and val.startswith(DANGEROUS_CSV_PREFIXES):
        return f"'{val}"
    return val


def neutralise_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the DataFrame with all dangerous string prefixes neutralized."""
    safe_df = df.copy()
    for col in safe_df.columns:
        if safe_df[col].dtype == object or pd.api.types.is_string_dtype(safe_df[col]):
            safe_df[col] = safe_df[col].apply(neutralise_value)
    return safe_df


def neutralise_csv(df: pd.DataFrame) -> str:
    """Export DataFrame to a CSV string defended against CSV formula injection."""
    return neutralise_frame(df).to_csv(index=False)


def filter_customers(
    df: pd.DataFrame,
    population: str = "Held-out customers only (1,409)",
    bands: list[str] | None = None,
    contracts: list[str] | None = None,
    tenure_bands: list[str] | None = None,
    min_prob: float = 0.0,
) -> pd.DataFrame:
    """Apply interactive filters to dashboard_data.csv."""
    filtered = df.copy()

    # 1. Population split
    if "Held-out" in population:
        filtered = filtered[filtered["split"] == "test"]

    # 2. Risk bands
    if bands:
        filtered = filtered[filtered["band"].isin(bands)]

    # 3. Contract types
    if contracts:
        filtered = filtered[filtered["contract"].isin(contracts)]

    # 4. Tenure bands
    if tenure_bands:
        filtered = filtered[filtered["tenure_band"].isin(tenure_bands)]

    # 5. Minimum probability
    if min_prob > 0.0:
        filtered = filtered[filtered["probability"] >= min_prob]

    return filtered.reset_index(drop=True)


def rank_by_expected_gain(
    df: pd.DataFrame,
    r: float = 0.5,
    d: float = 0.2,
    months: float = 12.0,
    c: float = 0.0,
) -> pd.DataFrame:
    """Compute expected net gain per customer and contact flag, sorting descending by gain."""
    if df.empty:
        out = df.copy()
        out["expected_gain"] = []
        out["contact"] = []
        return out

    p = df["probability"].to_numpy(dtype=float)
    bill = df["monthly_charges"].to_numpy(dtype=float)
    cutoff = cutoff_from(r, d)

    ev = ev_per_customer(p, bill, r=r, d=d, months=months, c=c)

    ranked = df.copy()
    ranked["expected_gain"] = ev
    ranked["contact"] = ranked["probability"] >= cutoff

    return ranked.sort_values("expected_gain", ascending=False).reset_index(drop=True)


def compute_explorer_summary(ranked_df: pd.DataFrame) -> dict[str, Any]:
    """Calculate summary statistics for the filtered & ranked customer segment."""
    n_matching = len(ranked_df)
    if n_matching == 0:
        return {
            "n_matching": 0,
            "n_contact": 0,
            "total_gain": 0.0,
            "formatted_matching": "0",
            "formatted_contact": "0",
            "formatted_gain": "$0",
        }

    contacted = ranked_df[ranked_df["contact"]]
    n_contact = len(contacted)
    total_gain = float(contacted["expected_gain"].sum())

    return {
        "n_matching": n_matching,
        "n_contact": n_contact,
        "total_gain": total_gain,
        "formatted_matching": f"{n_matching:,}",
        "formatted_contact": f"{n_contact:,}",
        "formatted_gain": f"${round(total_gain):,}",
    }



def format_batch_results(batch_resp: dict[str, Any]) -> dict[str, Any]:
    """Parse both legacy and current /predict_batch response formats."""
    summary = batch_resp.get("summary", {})

    n_scored = int(
        batch_resp.get("n_scored", summary.get("rows_valid", summary.get("rows", 0)))
    )
    n_errors = int(
        batch_resp.get("n_errors", summary.get("rows_with_errors", 0))
    )
    n_warnings = int(
        batch_resp.get("n_warnings", summary.get("rows_with_warnings", 0))
    )

    # Parse issues
    raw_issues = batch_resp.get("issues", [])
    issues_df = pd.DataFrame(raw_issues)

    if not issues_df.empty:
        level_order = {"error": 0, "warning": 1}
        issues_df["_sort"] = issues_df["level"].map(
            lambda x: level_order.get(str(x).lower(), 2)
        )
        issues_df = (
            issues_df.sort_values(["_sort", "row"])
            .drop(columns=["_sort"])
            .reset_index(drop=True)
        )

    # Support both legacy "scored" and current "results" keys
    raw_scored = batch_resp.get("scored", batch_resp.get("results", []))
    scored_df = pd.DataFrame(raw_scored)

    n_contact = 0
    total_gain = 0.0
    band_counts: dict[str, int] = {
        "Low": 0,
        "Medium": 0,
        "High": 0,
        "Very high": 0,
    }

    if not scored_df.empty:
        if "flag" in scored_df.columns:
            contact_mask = scored_df["flag"].fillna(False).astype(bool)
            n_contact = int(contact_mask.sum())

            if "expected_gain" in scored_df.columns:
                total_gain = float(
                    scored_df.loc[contact_mask, "expected_gain"].sum()
                )

        if "band" in scored_df.columns:
            counts = scored_df["band"].value_counts().to_dict()
            for band in band_counts:
                band_counts[band] = int(counts.get(band, 0))

    return {
        "n_scored": n_scored,
        "n_errors": n_errors,
        "n_warnings": n_warnings,
        "n_contact": n_contact,
        "total_gain": total_gain,
        "issues_df": issues_df,
        "scored_df": scored_df,
        "band_counts": band_counts,
        "issues_truncated": bool(
            batch_resp.get(
                "issues_truncated",
                summary.get("issues_truncated", False),
            )
        ),
    }
