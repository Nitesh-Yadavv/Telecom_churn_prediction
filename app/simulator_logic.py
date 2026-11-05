"""
app/simulator_logic.py

Pure computational logic for the Retention Campaign Simulator.
Zero Streamlit imports. Evaluates strategies, budget curves, band breakdowns,
paired bootstrap intervals, and sensitivity to the offer success rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from churn_core.economics import (
    cutoff_from,
    ev_per_customer,
    policy_ranked,
    policy_threshold,
    realized_per_customer,
    sensitivity,
)


def evaluate_strategies(
    df: pd.DataFrame,
    r: float,
    d: float,
    months: float,
    c: float = 0.0,
    fixed_cost: float = 0.0,
    capacity: int | None = None,
    use_expected_value: bool = False,
) -> list[dict[str, object]]:
    """Evaluate all 7 retention strategies against the chosen population."""
    n_pop = len(df)
    if n_pop == 0:
        return []

    p = df["probability"].to_numpy(dtype=float)
    bill = df["monthly_charges"].to_numpy(dtype=float)

    if use_expected_value:
        val_vec = ev_per_customer(p, bill, r, d, months, c)
    else:
        y = df["actual_churn"].to_numpy(dtype=int)
        val_vec = realized_per_customer(y, bill, r, d, months, c)

    cutoff = cutoff_from(r, d)

    # 1. Model, ranked by value
    flag_ranked = policy_ranked(
        p, bill, r, d, months, c=c, fixed_cost=fixed_cost, capacity=capacity
    )
    n_ranked = int(flag_ranked.sum())
    val_ranked = float(val_vec[flag_ranked].sum() - (fixed_cost if n_ranked > 0 else 0.0))

    # 2. Model, plain cutoff (with capacity if set: top-K by probability among flagged)
    flag_cutoff = policy_threshold(p, cutoff)
    if capacity is not None and flag_cutoff.sum() > capacity:
        idx_cand = np.where(flag_cutoff)[0]
        # Top-K by probability among threshold-flagged
        top_k_among_cand = idx_cand[np.argsort(-p[idx_cand])[: int(capacity)]]
        flag_cutoff = np.zeros(n_pop, dtype=bool)
        flag_cutoff[top_k_among_cand] = True
    n_cutoff = int(flag_cutoff.sum())
    val_cutoff = float(val_vec[flag_cutoff].sum() - (fixed_cost if n_cutoff > 0 else 0.0))

    # 3. Naive rule (Month-to-month and tenure < 12)
    mask_naive = (df["contract"] == "Month-to-month") & (df["tenure"] < 12)
    n_naive_total = int(mask_naive.sum())
    if capacity is not None and capacity < n_naive_total:
        n_naive = int(capacity)
        val_naive = float(
            (n_naive * val_vec[mask_naive].mean() if n_naive_total > 0 else 0.0)
            - (fixed_cost if n_naive > 0 else 0.0)
        )
    else:
        n_naive = n_naive_total
        val_naive = float(val_vec[mask_naive].sum() - (fixed_cost if n_naive > 0 else 0.0))

    # 4. All month-to-month
    mask_m2m = df["contract"] == "Month-to-month"
    n_m2m_total = int(mask_m2m.sum())
    if capacity is not None and capacity < n_m2m_total:
        n_m2m = int(capacity)
        val_m2m = float(
            (n_m2m * val_vec[mask_m2m].mean() if n_m2m_total > 0 else 0.0)
            - (fixed_cost if n_m2m > 0 else 0.0)
        )
    else:
        n_m2m = n_m2m_total
        val_m2m = float(val_vec[mask_m2m].sum() - (fixed_cost if n_m2m > 0 else 0.0))

    # 5. Random customers (match n_ranked count)
    n_random = n_ranked
    val_random = float(
        (n_random * val_vec.mean() if n_pop > 0 else 0.0) - (fixed_cost if n_random > 0 else 0.0)
    )

    # 6. Contact everyone (or capacity K)
    n_everyone = n_pop if capacity is None else min(n_pop, int(capacity))
    val_everyone = float(
        (n_everyone * val_vec.mean() if n_pop > 0 else 0.0)
        - (fixed_cost if n_everyone > 0 else 0.0)
    )

    # 7. No action
    val_none = 0.0
    n_none = 0

    strategies = [
        {"strategy": "Model, ranked by value", "contacted": n_ranked, "net_gain": val_ranked},
        {"strategy": "Model, plain cutoff", "contacted": n_cutoff, "net_gain": val_cutoff},
        {"strategy": "Naive rule (month-to-month, tenure < 1 yr)", "contacted": n_naive, "net_gain": val_naive},
        {"strategy": "All month-to-month", "contacted": n_m2m, "net_gain": val_m2m},
        {"strategy": "Random customers", "contacted": n_random, "net_gain": val_random},
        {"strategy": "Contact everyone", "contacted": n_everyone, "net_gain": val_everyone},
        {"strategy": "No action", "contacted": n_none, "net_gain": val_none},
    ]

    for s in strategies:
        gain = float(s["net_gain"])
        s["per_1000"] = (gain / n_pop * 1000.0) if n_pop > 0 else 0.0
        s["vs_no_action"] = gain - val_none

    return strategies


def compute_budget_curves(
    df: pd.DataFrame,
    r: float,
    d: float,
    months: float,
    c: float = 0.0,
    fixed_cost: float = 0.0,
    use_expected_value: bool = False,
    n_points: int = 140,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Compute (x_contacts, y_net_gain) curves across the full contact range 0..N."""
    n_pop = len(df)
    if n_pop == 0:
        return {}

    p = df["probability"].to_numpy(dtype=float)
    bill = df["monthly_charges"].to_numpy(dtype=float)

    if use_expected_value:
        val_vec = ev_per_customer(p, bill, r, d, months, c)
    else:
        y = df["actual_churn"].to_numpy(dtype=int)
        val_vec = realized_per_customer(y, bill, r, d, months, c)

    # Generate evenly spaced contact points from 0 to n_pop
    x = np.unique(np.linspace(0, n_pop, min(n_points + 1, n_pop + 1), dtype=int))

    # 1. Model ranked by value
    ev = ev_per_customer(p, bill, r, d, months, c)
    order_val = np.argsort(-ev, kind="stable")
    cum_val_ranked = np.cumsum(val_vec[order_val])
    y_ranked = np.zeros(len(x), dtype=float)
    for idx, k in enumerate(x):
        if k == 0:
            y_ranked[idx] = 0.0
        else:
            y_ranked[idx] = cum_val_ranked[k - 1] - fixed_cost

    # 2. Model by probability (top-K)
    order_p = np.argsort(-p, kind="stable")
    cum_val_p = np.cumsum(val_vec[order_p])
    y_prob = np.zeros(len(x), dtype=float)
    for idx, k in enumerate(x):
        if k == 0:
            y_prob[idx] = 0.0
        else:
            y_prob[idx] = cum_val_p[k - 1] - fixed_cost

    # 3. Naive rule (Month-to-month & tenure < 12)
    mask_naive = (df["contract"] == "Month-to-month") & (df["tenure"] < 12)
    n_naive = int(mask_naive.sum())
    mean_naive_val = float(val_vec[mask_naive].mean()) if n_naive > 0 else 0.0
    tot_naive_val = float(val_vec[mask_naive].sum())
    y_naive = np.zeros(len(x), dtype=float)
    for idx, k in enumerate(x):
        if k == 0:
            y_naive[idx] = 0.0
        elif k <= n_naive:
            y_naive[idx] = k * mean_naive_val - fixed_cost
        else:
            y_naive[idx] = tot_naive_val - fixed_cost

    # 4. All month-to-month
    mask_m2m = df["contract"] == "Month-to-month"
    n_m2m = int(mask_m2m.sum())
    mean_m2m_val = float(val_vec[mask_m2m].mean()) if n_m2m > 0 else 0.0
    tot_m2m_val = float(val_vec[mask_m2m].sum())
    y_m2m = np.zeros(len(x), dtype=float)
    for idx, k in enumerate(x):
        if k == 0:
            y_m2m[idx] = 0.0
        elif k <= n_m2m:
            y_m2m[idx] = k * mean_m2m_val - fixed_cost
        else:
            y_m2m[idx] = tot_m2m_val - fixed_cost

    # 5. Random customers
    mean_pop_val = float(val_vec.mean())
    y_random = np.zeros(len(x), dtype=float)
    for idx, k in enumerate(x):
        if k == 0:
            y_random[idx] = 0.0
        else:
            y_random[idx] = k * mean_pop_val - fixed_cost

    return {
        "x": x,
        "model_ranked": y_ranked,
        "model_prob": y_prob,
        "naive": y_naive,
        "month_to_month": y_m2m,
        "random": y_random,
    }


def compute_band_breakdown(
    df: pd.DataFrame,
    r: float,
    d: float,
    months: float,
    c: float = 0.0,
    use_expected_value: bool = False,
) -> list[dict[str, object]]:
    """Calculate the net value and customer count of contacting each risk band."""
    p = df["probability"].to_numpy(dtype=float)
    bill = df["monthly_charges"].to_numpy(dtype=float)

    if use_expected_value:
        val_vec = ev_per_customer(p, bill, r, d, months, c)
    else:
        y = df["actual_churn"].to_numpy(dtype=int)
        val_vec = realized_per_customer(y, bill, r, d, months, c)

    results = []
    bands_order = ["Low", "Medium", "High", "Very high"]
    for b in bands_order:
        mask = df["band"] == b
        count = int(mask.sum())
        gain = float(val_vec[mask].sum()) if count > 0 else 0.0
        results.append({"band": b, "count": count, "net_gain": gain})
    return results


def run_paired_bootstrap(
    df: pd.DataFrame,
    r: float,
    d: float,
    months: float,
    c: float = 0.0,
    fixed_cost: float = 0.0,
    capacity: int | None = None,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Run a paired bootstrap of (model ranked minus naive) over real customer outcomes."""
    n_pop = len(df)
    if n_pop == 0:
        return {"point": 0.0, "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    y = df["actual_churn"].to_numpy(dtype=int)
    p = df["probability"].to_numpy(dtype=float)
    bill = df["monthly_charges"].to_numpy(dtype=float)
    val_vec = realized_per_customer(y, bill, r, d, months, c)

    flag_ranked = policy_ranked(
        p, bill, r, d, months, c=c, fixed_cost=fixed_cost, capacity=capacity
    )
    mask_naive = ((df["contract"] == "Month-to-month") & (df["tenure"] < 12)).to_numpy()

    # Paired customer-level difference vector
    diff_vec = np.where(flag_ranked, val_vec, 0.0) - np.where(mask_naive, val_vec, 0.0)
    point_est = float(diff_vec.sum())

    rng = np.random.default_rng(seed)
    resamples = rng.choice(diff_vec, size=(n_resamples, n_pop), replace=True).sum(axis=1)
    ci_low, ci_high = np.percentile(resamples, [2.5, 97.5])

    return {
        "point": point_est,
        "mean": float(resamples.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def compute_sensitivity_table(
    df: pd.DataFrame,
    sidebar_r: float,
    d: float,
    months: float,
    r_values: list[float] | np.ndarray | None = None,
) -> tuple[pd.DataFrame, float | None]:
    """Calculate sensitivity across offer success rates and find the break-even r."""
    if r_values is None:
        r_values = np.round(np.arange(0.1, 1.05, 0.05), 2)

    p = df["probability"].to_numpy(dtype=float)
    y = df["actual_churn"].to_numpy(dtype=int)
    bill = df["monthly_charges"].to_numpy(dtype=float)
    base_cutoff = cutoff_from(sidebar_r, d)

    rows = sensitivity(p, y, bill, r_values, d, months, base_cutoff)
    sens_df = pd.DataFrame(rows)

    # Linear interpolation to find break-even r where saving_kept crosses 0
    break_even_r: float | None = None
    for i in range(len(sens_df) - 1):
        r1, s1 = sens_df.loc[i, "r"], sens_df.loc[i, "saving_kept"]
        r2, s2 = sens_df.loc[i + 1, "r"], sens_df.loc[i + 1, "saving_kept"]
        if (s1 <= 0 and s2 >= 0) or (s1 >= 0 and s2 <= 0):
            if s2 != s1:
                break_even_r = float(r1 + (0.0 - s1) / (s2 - s1) * (r2 - r1))
            else:
                break_even_r = float(r1)
            break

    return sens_df, break_even_r
