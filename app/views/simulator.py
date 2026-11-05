"""
app/views/simulator.py

Retention campaign simulator.
Answers whether a retention campaign earns or loses money under user-defined
business assumptions (r, d, months, contact cost, fixed cost, and contact capacity).
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data import load_dashboard_data
from app.simulator_logic import (
    compute_band_breakdown,
    compute_budget_curves,
    compute_sensitivity_table,
    evaluate_strategies,
    run_paired_bootstrap,
)
from app.ui import (
    BAND_COLORS,
    DEEP_TEAL,
    PRIMARY_TEAL,
    apply_plotly_theme,
    card,
    format_currency,
    kpi,
)
from churn_core.economics import policy_ranked

# ----------------- Read Sidebar Business Assumptions -----------------
r = float(st.session_state.get("r", 0.5))
d = float(st.session_state.get("d", 0.2))
months = float(st.session_state.get("months", 12))

st.title("Retention campaign simulator")
st.markdown(
    "**Move the sliders in the sidebar:** this page shows whether a retention campaign "
    "would earn or lose money under your assumptions."
)

# ----------------- Controls Row -----------------
with card("sim_controls"):
    view_option = st.radio(
        "Evaluation view",
        options=[
            "Held-out customers, real outcomes",
            "Held-out customers, model's expectation",
            "All customers, real outcomes (secondary)",
        ],
        index=0,
        key="sim_view",
        horizontal=True,
    )

    c_left, c_mid, c_right = st.columns([1, 1, 1.6])
    with c_left:
        c = st.number_input(
            "Contact cost per customer",
            min_value=0.0,
            value=0.0,
            step=1.0,
            key="sim_c",
            help="in the same currency units as the bill",
        )
    with c_mid:
        fixed_cost = st.number_input(
            "Fixed campaign cost",
            min_value=0.0,
            value=0.0,
            step=100.0,
            key="sim_fixed",
            help="Fixed setup or campaign fee incurred once if any customer is contacted",
        )
    with c_right:
        limit_capacity = st.checkbox(
            "Limit the number of customers contacted",
            value=False,
            key="sim_limit",
        )

# ----------------- Data Population -----------------
full_df = load_dashboard_data()
if "Held-out" in view_option:
    pop_df = full_df[full_df["split"] == "test"].copy().reset_index(drop=True)
else:
    pop_df = full_df.copy().reset_index(drop=True)
    st.info("Results on training customers use out-of-fold scores and are slightly optimistic.")

n_pop = len(pop_df)
capacity: int | None = None
if limit_capacity:
    capacity = int(
        st.slider(
            "Contact capacity (max customers to reach)",
            min_value=10,
            max_value=n_pop,
            value=min(300, n_pop),
            step=10,
            key="sim_capacity",
        )
    )

use_expected = "model's expectation" in view_option

# ----------------- Evaluate Strategies -----------------
strategies = evaluate_strategies(
    pop_df,
    r=r,
    d=d,
    months=months,
    c=c,
    fixed_cost=fixed_cost,
    capacity=capacity,
    use_expected_value=use_expected,
)

strat_map = {s["strategy"]: s for s in strategies}
ranked_model = strat_map["Model, ranked by value"]
naive_rule = strat_map["Naive rule (month-to-month, tenure < 1 yr)"]

n_model = int(ranked_model["contacted"])
net_gain = float(ranked_model["net_gain"])
per_1000 = float(ranked_model["per_1000"])
vs_naive = net_gain - float(naive_rule["net_gain"])
naive_base = abs(float(naive_rule["net_gain"]))
rel_pct = (vs_naive / naive_base * 100.0) if naive_base > 0 else 0.0

# ----------------- 4 KPI Cards -----------------
col1, col2, col3, col4 = st.columns(4)

with col1, card("kpi_contacts"):
    if not use_expected:
        p_arr = pop_df["probability"].to_numpy(dtype=float)
        bill_arr = pop_df["monthly_charges"].to_numpy(dtype=float)
        flag_arr = policy_ranked(
            p_arr, bill_arr, r, d, months, c=c, fixed_cost=fixed_cost, capacity=capacity
        )
        y_arr = pop_df["actual_churn"].to_numpy(dtype=int)
        real_churners = int(y_arr[flag_arr].sum())
        prec = (real_churners / n_model) if n_model > 0 else 0.0
        retained = round(r * real_churners)
        caption_kpi1 = (
            f"{n_model / n_pop:.1%} of customers; about {real_churners:,} of them are real "
            f"churners (precision {prec:.1%}); about {retained:,} retained at your success rate"
        )
    else:
        caption_kpi1 = (
            f"{n_model / n_pop:.1%} of customers; scored by the model's expected probabilities"
        )
    kpi("Customers to contact", f"{n_model:,}", caption=caption_kpi1)

with col2, card("kpi_gain"):
    kpi(
        "Net gain",
        format_currency(net_gain),
        caption=f"over {int(months)} months, in currency units; view: {view_option.split(',')[0]}",
    )

with col3, card("kpi_per_1000"):
    kpi(
        "Gain per 1,000 customers",
        format_currency(per_1000),
        caption=f"per 1,000 customers in this population ({n_pop:,} total)",
    )

with col4, card("kpi_vs_naive"):
    kpi(
        "Versus the naive rule",
        f"{vs_naive:+,.0f}",
        caption=(
            f"{rel_pct:+.0f}% relative difference compared with "
            "targeting month-to-month customers with under 12 months of tenure"
        ),
        help="The naive rule contacts month-to-month customers with under 12 months of tenure.",
    )

st.markdown("")

# ----------------- Chart 1: Budget Curves -----------------
curves = compute_budget_curves(
    pop_df,
    r=r,
    d=d,
    months=months,
    c=c,
    fixed_cost=fixed_cost,
    use_expected_value=use_expected,
)

fig_budget = go.Figure()
fig_budget.add_trace(
    go.Scatter(
        x=curves["x"],
        y=curves["model_ranked"],
        mode="lines",
        name="Model, ranked by value",
        line={"color": PRIMARY_TEAL, "width": 3},
    )
)
fig_budget.add_trace(
    go.Scatter(
        x=curves["x"],
        y=curves["model_prob"],
        mode="lines",
        name="Model by probability (top-K)",
        line={"color": "#3F73D6", "width": 2, "dash": "dash"},
    )
)
fig_budget.add_trace(
    go.Scatter(
        x=curves["x"],
        y=curves["naive"],
        mode="lines",
        name="Naive rule (random K)",
        line={"color": "#E5710C", "width": 2},
    )
)
fig_budget.add_trace(
    go.Scatter(
        x=curves["x"],
        y=curves["month_to_month"],
        mode="lines",
        name="All month-to-month (random K)",
        line={"color": "#9B51E0", "width": 2},
    )
)
fig_budget.add_trace(
    go.Scatter(
        x=curves["x"],
        y=curves["random"],
        mode="lines",
        name="Random customers",
        line={"color": "#828282", "width": 1.5, "dash": "dot"},
    )
)

# Current selection marker
if n_model > 0:
    fig_budget.add_trace(
        go.Scatter(
            x=[n_model],
            y=[net_gain],
            mode="markers",
            name="Current selection",
            marker={"size": 11, "color": DEEP_TEAL, "symbol": "diamond", "line": {"width": 2, "color": "#FFFFFF"}},
            hovertemplate="Current plan: %{x:,} contacted, %{y:,.0f} gain<extra></extra>",
        )
    )

fig_budget.update_layout(
    xaxis_title="Customers contacted",
    yaxis_title="Net gain (currency units)",
    height=420,
)
apply_plotly_theme(fig_budget)

st.subheader("Net gain by number of customers contacted")
st.plotly_chart(
    fig_budget,
    use_container_width=True,
    theme=None,
)
st.caption(
    "The curves rise, flatten and then fall: contacting customers beyond the break-even loses money. "
    "Do not choose a budget from the peak of one test set; it would overfit that sample."
)

st.markdown("")

# ----------------- Two Columns: Band Breakdown & Strategy Table -----------------
col_chart, col_tbl = st.columns([1, 1.25])

with col_chart:
    st.subheader("Net gain by risk band")
    bands = compute_band_breakdown(
        pop_df, r=r, d=d, months=months, c=c, use_expected_value=use_expected
    )

    band_names = [b["band"] for b in bands]
    band_gains = [b["net_gain"] for b in bands]
    band_counts = [b["count"] for b in bands]
    colors = [BAND_COLORS.get(b, PRIMARY_TEAL) for b in band_names]

    fig_bands = go.Figure(
        go.Bar(
            x=band_names,
            y=band_gains,
            marker_color=colors,
            text=[f"{cnt:,} cust<br>{format_currency(gn)}" for cnt, gn in zip(band_counts, band_gains)],
            textposition="auto",
            hovertemplate="Band: %{x}<br>Net gain: %{y:,.0f}<br>Customers: %{text}<extra></extra>",
        )
    )
    fig_bands.update_layout(yaxis_title="Net gain if band contacted", height=380)
    apply_plotly_theme(fig_bands)
    st.plotly_chart(fig_bands, use_container_width=True)

    low_med_lose = (band_gains[0] < 0) and (band_gains[1] < 0)
    high_earn = (band_gains[2] > 0) or (band_gains[3] > 0)
    if low_med_lose and high_earn:
        caption_bands = "Low and Medium customers lose money on average at your assumptions; High and Very high earn it."
    else:
        caption_bands = "Net gain breakdown by customer risk band under your current economic assumptions."
    st.caption(caption_bands)

with col_tbl:
    st.subheader("Comparison of retention strategies")
    rows_display = []
    for s in strategies:
        rows_display.append(
            {
                "Strategy": s["strategy"],
                "Customers contacted": f"{int(s['contacted']):,}",
                "Net gain": format_currency(float(s["net_gain"])),
                "Per 1,000": format_currency(float(s["per_1000"])),
                "Vs no action": format_currency(float(s["vs_no_action"])),
            }
        )
    table_df = pd.DataFrame(rows_display)
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    # Export strategies to CSV
    csv_bytes = table_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download table (CSV)",
        data=csv_bytes,
        file_name="strategy_comparison.csv",
        mime="text/csv",
        key="sim_dl_table",
    )

st.markdown("")

# ----------------- Section 5: Uncertainty Card & Sensitivity -----------------
if not use_expected:
    st.subheader("Gain versus the naive rule with uncertainty")
    with card("kpi_bootstrap"):
        boot = run_paired_bootstrap(
            pop_df, r=r, d=d, months=months, c=c, fixed_cost=fixed_cost, capacity=capacity
        )
        st.markdown(
            f"**Paired customer-level bootstrap:** Model ranked by value earns **{boot['point']:+,.0f}** "
            f"more than the naive rule.\n\n"
            f"- **95% confidence interval:** `[{boot['ci_low']:+,.0f} to {boot['ci_high']:+,.0f}]` "
            f"(2,000 resamples, seeded for reproducibility)\n"
            f"- **Bootstrap resample mean:** `{boot['mean']:+,.0f}`"
        )
else:
    st.info(
        "Paired bootstrap uncertainty requires real observed outcomes and is disabled in the expected-value view."
    )

st.markdown("")

# ----------------- Chart 3: Sensitivity to Success Rate -----------------
if not use_expected:
    st.subheader("If the offer success rate is not what you assumed")
    sens_df, break_even_r = compute_sensitivity_table(pop_df, sidebar_r=r, d=d, months=months)

    fig_sens = go.Figure()
    fig_sens.add_trace(
        go.Scatter(
            x=sens_df["r"],
            y=sens_df["saving_adapted"],
            mode="lines",
            name="Cutoff re-derived for each r",
            line={"color": PRIMARY_TEAL, "width": 2.5},
        )
    )
    fig_sens.add_trace(
        go.Scatter(
            x=sens_df["r"],
            y=sens_df["saving_kept"],
            mode="lines",
            name=f"Cutoff kept at current value ({d / (d + r * (1 - d)):.1%})",
            line={"color": "#E5710C", "width": 2.5, "dash": "dash"},
        )
    )
    # Zero line
    fig_sens.add_hline(y=0, line_dash="dot", line_color="#828282", annotation_text="Break-even (0 gain)")
    # Vertical marker at current r
    fig_sens.add_vline(x=r, line_dash="dash", line_color=PRIMARY_TEAL, annotation_text=f"Current r ({r:.0%})")

    fig_sens.update_layout(
        xaxis_title="True offer success rate (r)",
        yaxis_title="Net saving (currency units)",
        height=380,
    )
    apply_plotly_theme(fig_sens)
    st.plotly_chart(fig_sens, use_container_width=True)

    if break_even_r is not None:
        st.caption(
            f"Break-even success rate: about **{break_even_r:.1%}**. "
            "Below this success rate, keeping the current cutoff loses money on real outcomes."
        )

st.markdown("")

# ----------------- Limitations Box -----------------
st.info(
    "The model predicts who is likely to leave, not who would respond to an offer, so one success "
    "rate is assumed for everyone. The success rate, discount and months are assumptions: this data "
    "has no offers or dates. The only way to learn the real success rate is a controlled test of the offer. "
    "Results on training customers use out-of-fold scores and are slightly optimistic."
)

# ----------------- Settings Download (JSON) -----------------
settings_payload = {
    "evaluation_view": view_option,
    "assumptions": {"r": r, "d": d, "months": months},
    "campaign_costs": {"contact_cost_c": c, "fixed_cost_F": fixed_cost},
    "capacity_limit": capacity,
    "headline_results": {
        "customers_contacted": n_model,
        "net_gain": net_gain,
        "gain_per_1000": per_1000,
        "versus_naive_rule": vs_naive,
    },
}
st.download_button(
    "Download these settings (JSON)",
    data=json.dumps(settings_payload, indent=2),
    file_name="simulator_settings.json",
    mime="application/json",
    key="sim_dl_settings",
)