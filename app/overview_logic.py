"""
app/overview_logic.py

Pure business calculation functions for the Business Overview dashboard page.
Keeps data processing deterministic and cleanly testable.
"""
from __future__ import annotations

import pandas as pd

from churn_core.economics import policy_ranked, realized_value


def get_overview_kpis(overview_dict: dict) -> dict[str, dict[str, str | float]]:
    """Extract and format the four primary KPI metrics from overview_numbers.json."""
    tot_rev = float(overview_dict.get("total_monthly_revenue", 0.0))
    rev_lost = float(overview_dict.get("revenue_lost_monthly", 0.0))
    lost_share = float(overview_dict.get("revenue_lost_share", 0.0))
    n_curr = int(overview_dict.get("n_current_customers", 0))
    curr_rev = float(overview_dict.get("current_monthly_revenue", 0.0))
    exp_risk = float(overview_dict.get("expected_revenue_at_risk_monthly", 0.0))
    risk_share = (exp_risk / curr_rev) if curr_rev > 0 else 0.0

    return {
        "total_revenue": {
            "value": tot_rev,
            "formatted": f"${tot_rev:,.0f}",
            "caption": "All 7,043 historical & current accounts",
        },
        "revenue_lost": {
            "value": rev_lost,
            "formatted": f"${rev_lost:,.0f}",
            "caption": f"{lost_share * 100:.1f}% of total monthly revenue",
        },
        "current_customers": {
            "value": n_curr,
            "formatted": f"{n_curr:,}",
            "caption": f"monthly revenue: ${curr_rev:,.0f}",
        },
        "expected_at_risk": {
            "value": exp_risk,
            "formatted": f"${exp_risk:,.0f}",
            "caption": f"{risk_share * 100:.1f}% of current monthly revenue",
        },
    }


def get_segment_data(segment_df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Filter segment_revenue for a dimension, sorted descending by revenue_lost."""
    dim_df = segment_df[segment_df["dimension"] == dimension].copy()
    dim_df = dim_df.sort_values("revenue_lost", ascending=False).reset_index(drop=True)
    return dim_df


def get_top_segment_insight(dim_df: pd.DataFrame, dimension_name: str) -> str:
    """Auto-generate an insight sentence for the leading segment in a dimension."""
    if dim_df.empty:
        return ""
    top = dim_df.iloc[0]
    seg_name = str(top["segment"])
    share_pct = float(top["share_of_lost"]) * 100

    if dimension_name.lower() == "contract":
        return f"{seg_name} contracts account for {share_pct:.1f}% of lost revenue."
    if dimension_name.lower() == "tenure band":
        return f"{seg_name} months tenure accounts for {share_pct:.1f}% of lost revenue."
    return f"{seg_name} accounts for {share_pct:.1f}% of lost revenue."


def compute_q1_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Compute churn rate by contract."""
    res = (
        df.groupby("contract", observed=False)["actual_churn"]
        .agg(customers="count", churners="sum", churn_rate="mean")
        .reindex(["Month-to-month", "One year", "Two year"])
        .reset_index()
    )
    return res


def compute_q2_quartiles(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Compute churn rate by monthly charges quartile with explicit interval edges."""
    _, bins = pd.qcut(df["monthly_charges"], q=4, retbins=True, precision=2)
    labels = [f"${bins[i]:.2f}–${bins[i+1]:.2f}" for i in range(4)]
    temp = df[["actual_churn"]].copy()
    temp["quartile"] = pd.qcut(df["monthly_charges"], q=4, labels=labels)
    res = (
        temp.groupby("quartile", observed=False)["actual_churn"]
        .agg(customers="count", churners="sum", churn_rate="mean")
        .reset_index()
    )
    return res, labels


def compute_q3_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """Compute churn rate by tenure band."""
    order = ["0-11", "12-35", "36-72"]
    res = (
        df.groupby("tenure_band", observed=False)["actual_churn"]
        .agg(customers="count", churners="sum", churn_rate="mean")
        .reindex(order)
        .reset_index()
    )
    return res


def compute_q4_services(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute churn rates for internet service and payment method."""
    internet = (
        df.groupby("internet_service", observed=False)["actual_churn"]
        .agg(customers="count", churners="sum", churn_rate="mean")
        .sort_values("churn_rate", ascending=False)
        .reset_index()
    )
    payment = (
        df.groupby("payment_method", observed=False)["actual_churn"]
        .agg(customers="count", churners="sum", churn_rate="mean")
        .sort_values("churn_rate", ascending=False)
        .reset_index()
    )
    return {"internet": internet, "payment": payment}


def compute_q5_net_gain(
    df: pd.DataFrame,
    r: float = 0.5,
    d: float = 0.2,
    months: float = 12.0,
    c: float = 0.0,
) -> float:
    """Compute net gain of Model ranked by value on held-out customers at default assumptions."""
    test_df = df[df["split"] == "test"].copy()
    p = test_df["probability"].to_numpy(dtype=float)
    bill = test_df["monthly_charges"].to_numpy(dtype=float)
    y = test_df["actual_churn"].to_numpy(dtype=int)

    flag = policy_ranked(p, bill, r=r, d=d, months=months, c=c)
    gain = realized_value(flag, y, bill, r=r, d=d, months=months, c=c)
    return float(gain)
