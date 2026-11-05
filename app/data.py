"""
app/data.py

Cached data loaders for the Telecom Churn Prediction dashboard.
Reads precomputed evaluation results, curves, and summaries from reports/, models/,
and outputs/ using path resolution from churn_core.config. All loaders use @st.cache_data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from churn_core import config


def _load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame, raising a descriptive error if missing or corrupt."""
    if not path.is_file():
        st.error(f"Missing required data file: '{path.name}' (expected at: {path})")
        raise FileNotFoundError(f"Missing required data file: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        st.error(f"Data file is empty: '{path.name}' (at: {path})")
        raise ValueError(f"Data file is empty: {path}") from exc
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        st.error(f"Corrupt or unreadable CSV file '{path.name}': {exc}")
        raise ValueError(f"Corrupt or unreadable CSV file: {path} ({exc})") from exc

    if df.empty:
        st.error(f"Data file contains no rows: '{path.name}' (at: {path})")
        raise ValueError(f"Data file contains no rows: {path}")
    return df


def _load_json(path: Path) -> dict[str, Any]:
    """Read a JSON file into a dictionary, raising a descriptive error if missing or corrupt."""
    if not path.is_file():
        st.error(f"Missing required data file: '{path.name}' (expected at: {path})")
        raise FileNotFoundError(f"Missing required data file: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        st.error(f"Corrupt or unreadable JSON file '{path.name}': {exc}")
        raise ValueError(f"Corrupt or unreadable JSON file: {path} ({exc})") from exc
    return data


@st.cache_data
def load_dashboard_data() -> pd.DataFrame:
    """Load the full 7,043 customer dataset with calibrated scores and actual churn."""
    return _load_csv(config.reports_dir() / "dashboard_data.csv")


@st.cache_data
def load_test_predictions() -> pd.DataFrame:
    """Load predictions and expected gains for the 1,409 held-out test customers."""
    return _load_csv(config.reports_dir() / "test_predictions.csv")


@st.cache_data
def load_metrics() -> dict[str, Any]:
    """Load headline model evaluation metrics, cutoff, and Brier scores."""
    return _load_json(config.reports_dir() / "metrics.json")


@st.cache_data
def load_overview_numbers() -> dict[str, Any]:
    """Load executive overview metrics including monthly revenue and revenue at risk."""
    return _load_json(config.reports_dir() / "overview_numbers.json")


@st.cache_data
def load_segment_revenue() -> pd.DataFrame:
    """Load revenue lost and revenue at risk broken down by customer segment."""
    return _load_csv(config.reports_dir() / "segment_revenue.csv")


@st.cache_data
def load_segment_metrics() -> pd.DataFrame:
    """Load test-set performance metrics (recall, precision, flag rate) by segment."""
    return _load_csv(config.reports_dir() / "segment_metrics.csv")


@st.cache_data
def load_band_table() -> pd.DataFrame:
    """Load observed churn rate and mean predicted probability per risk band."""
    return _load_csv(config.reports_dir() / "band_table.csv")


@st.cache_data
def load_cv_results() -> pd.DataFrame:
    """Load cross-validation comparison metrics across the evaluated ML pipelines."""
    return _load_csv(config.reports_dir() / "cv_results.csv")


@st.cache_data
def load_pr_curve() -> pd.DataFrame:
    """Load precision and recall coordinates for the calibrated model on the test set."""
    return _load_csv(config.reports_dir() / "pr_curve.csv")


@st.cache_data
def load_roc_curve() -> pd.DataFrame:
    """Load false positive rate and true positive rate coordinates for the ROC curve."""
    return _load_csv(config.reports_dir() / "roc_curve.csv")


@st.cache_data
def load_gains_curve() -> pd.DataFrame:
    """Load cumulative gains curve coordinates for the test set."""
    return _load_csv(config.reports_dir() / "gains_curve.csv")


@st.cache_data
def load_calibration_curve() -> pd.DataFrame:
    """Load reliability diagram data for uncalibrated and calibrated models."""
    return _load_csv(config.reports_dir() / "calibration_curve.csv")


@st.cache_data
def load_cutoff_sweep() -> pd.DataFrame:
    """Load savings, recall, and precision across decision cutoffs."""
    return _load_csv(config.reports_dir() / "cutoff_sweep.csv")


@st.cache_data
def load_shap_importance(path: Path | None = None) -> pd.DataFrame:
    """Load feature-level SHAP importance values with schema validation."""
    target_path = path if path is not None else config.home() / "outputs" / "shap_importance.csv"
    df = _load_csv(target_path)
    if len(df.columns) < 2:
        st.error(f"SHAP importance file has invalid schema (expected >= 2 columns, got {len(df.columns)})")
        raise ValueError(f"SHAP importance file requires at least 2 columns, got {len(df.columns)}")

    # Standardize column names if header does not explicitly have feature and importance
    if "feature" not in df.columns or "importance" not in df.columns:
        first_col, second_col = df.columns[0], df.columns[1]
        df = df.rename(columns={first_col: "feature", second_col: "importance"})

    # Ensure importance column is numeric
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
    if df["importance"].isna().all():
        st.error("SHAP importance values could not be parsed as numeric")
        raise ValueError("SHAP importance column contains no valid numeric values")

    return df


@st.cache_data
def load_shap_plot_path(plots_dir: Path | None = None) -> Path:
    """Resolve the path to the primary SHAP importance plot or fallback to grouped plot."""
    target_dir = plots_dir if plots_dir is not None else config.home() / "outputs" / "plots"
    candidates = [
        target_dir / "shap_importance.png",
        target_dir / "shap_grouped_importance.png",
        target_dir / "shap_summary.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    missing_desc = ", ".join(f"'{p.name}'" for p in candidates)
    st.error(f"Missing SHAP plot file. None of the candidate files exist: {missing_desc} in {target_dir}")
    raise FileNotFoundError(f"No SHAP plot file found in {target_dir} (checked {missing_desc})")


@st.cache_data
def load_risk_bands() -> dict[str, Any]:
    """Load risk band definitions, thresholds, and observed distribution data."""
    return _load_json(config.models_dir() / "risk_bands.json")


@st.cache_data
def load_model_config() -> dict[str, Any]:
    """Load saved model configuration, feature order, and training metadata."""
    return _load_json(config.models_dir() / "model_config.json")
