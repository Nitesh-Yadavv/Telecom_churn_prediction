"""
app/performance_logic.py

Pure data transformation and calculation functions for the Model Performance dashboard page.
Ensures deterministic, easily testable logic separated from Streamlit presentation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def get_performance_kpis(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract and format the headline performance metrics from metrics.json."""
    auc_pr = float(metrics.get("auc_pr", 0.0))
    roc_auc = float(metrics.get("roc_auc", 0.0))
    brier_uncal = float(metrics.get("brier_uncalibrated", 0.0))
    brier_sig = float(metrics.get("brier_sigmoid", 0.0))
    recall = float(metrics.get("recall", 0.0))
    precision = float(metrics.get("precision", 0.0))
    flagged = int(metrics.get("flagged", 0))
    n_test = int(metrics.get("n_test", 1409))
    flag_rate = (flagged / n_test) if n_test > 0 else 0.0

    return {
        "auc_pr": {
            "value": auc_pr,
            "display": f"{auc_pr:.3f}",
            "caption": "Primary metric (churners are 26.5% minority)",
        },
        "roc_auc": {
            "value": roc_auc,
            "display": f"{roc_auc:.3f}",
            "caption": "Ranking discrimination across all cutoffs",
        },
        "brier": {
            "value": brier_sig,
            "display": f"{brier_uncal:.3f} → {brier_sig:.3f}",
            "caption": "Sigmoid calibration improves probability accuracy",
        },
        "recall_precision": {
            "recall": recall,
            "precision": precision,
            "display": f"{recall * 100:.1f}% / {precision * 100:.1f}%",
            "caption": f"At cutoff {metrics.get('cutoff', 1/3):.3f}: {recall*100:.1f}% caught, {precision*100:.1f}% precision",
        },
        "flagged": {
            "value": flagged,
            "n_test": n_test,
            "display": f"{flagged:,} of {n_test:,}",
            "caption": f"{flag_rate * 100:.1f}% contact rate on held-out customers",
        },
    }


def get_model_comparison_data(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Format and sort CV results for pipeline comparison, flagging finalists."""
    df = cv_df.copy()
    finalists = {"LogisticRegression_ClassWeight", "LightGBM_ClassWeight"}
    df["is_finalist"] = df["pipeline"].isin(finalists)
    df["clean_name"] = df["pipeline"].str.replace("_", " ")
    return df.sort_values("auc_pr", ascending=True).reset_index(drop=True)


def format_band_table(band_df: pd.DataFrame, sample: str = "test") -> pd.DataFrame:
    """Filter and format the band table for a specific sample ('test' or 'oof')."""
    order = ["Low", "Medium", "High", "Very high"]
    sub = band_df[band_df["sample"] == sample].copy()
    sub["band_cat"] = pd.Categorical(sub["band"], categories=order, ordered=True)
    sub = sub.sort_values("band_cat").drop(columns=["band_cat"])
    return sub.reset_index(drop=True)


def check_band_ordering(band_df: pd.DataFrame) -> dict[str, bool]:
    """Verify that observed churn increases monotonically across bands in both samples."""
    results = {}
    for sample in ["test", "oof"]:
        sub = format_band_table(band_df, sample=sample)
        churn_rates = sub["observed_churn_rate"].to_numpy()
        is_increasing = bool(np.all(np.diff(churn_rates) > 0))
        results[sample] = is_increasing
    return results


def find_weakest_segment(seg_df: pd.DataFrame) -> dict[str, Any]:
    """Identify the weakest segment by recall from segment_metrics.csv."""
    # Restrict to contract and tenure_band segments
    focus = seg_df[seg_df["group_by"].isin(["contract", "tenure_band"])].copy()
    if focus.empty:
        focus = seg_df.copy()
    weakest_row = focus.loc[focus["recall"].idxmin()]
    precision_val = (
        float(weakest_row["precision"])
        if pd.notna(weakest_row["precision"])
        else 0.0
    )
    return {
        "group": str(weakest_row["group"]),
        "dimension": str(weakest_row["group_by"]),
        "n": int(weakest_row["n"]),
        "churners": int(weakest_row["churners"]),
        "caught": int(weakest_row["caught"]),
        "recall": float(weakest_row["recall"]),
        "precision": precision_val,
        "churn_rate": float(weakest_row["churn_rate"]),
    }


def get_fairness_tables(seg_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract and format fairness metrics for gender and senior citizen status."""
    gender_rows = seg_df[seg_df["group_by"] == "gender_Male"].copy()
    gender_map = {"0.0": "Female", "1.0": "Male", 0.0: "Female", 1.0: "Male", "0": "Female", "1": "Male"}
    gender_rows["Subgroup"] = gender_rows["group"].map(lambda x: gender_map.get(str(x), str(x)))

    senior_rows = seg_df[seg_df["group_by"] == "senior_citizen"].copy()
    senior_map = {"0": "Non-senior", "1": "Senior citizen", 0: "Non-senior", 1: "Senior citizen"}
    senior_rows["Subgroup"] = senior_rows["group"].map(lambda x: senior_map.get(str(x), str(x)))

    cols = ["Subgroup", "n", "churn_rate", "flag_rate", "recall", "precision"]
    gender_clean = gender_rows[cols].rename(
        columns={
            "n": "Customers",
            "churn_rate": "Churn rate",
            "flag_rate": "Flag rate",
            "recall": "Recall",
            "precision": "Precision",
        }
    ).reset_index(drop=True)

    senior_clean = senior_rows[cols].rename(
        columns={
            "n": "Customers",
            "churn_rate": "Churn rate",
            "flag_rate": "Flag rate",
            "recall": "Recall",
            "precision": "Precision",
        }
    ).reset_index(drop=True)

    return gender_clean, senior_clean


def get_top_shap_features(shap_df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Format and clean SHAP feature importances, returning the top N features."""
    df = shap_df.copy()
    if "feature" not in df.columns or "importance" not in df.columns:
        df = df.rename(columns={df.columns[0]: "feature", df.columns[1]: "importance"})
    df = df.dropna(subset=["importance"]).sort_values("importance", ascending=False)
    # Clean feature names for readable UI labels
    df["display_feature"] = (
        df["feature"]
        .str.replace("_", " ")
        .str.replace("contract ", "Contract: ")
        .str.replace("internet service ", "Internet: ")
        .str.replace("payment method ", "Payment: ")
        .str.capitalize()
    )
    return df.head(top_n).reset_index(drop=True)
