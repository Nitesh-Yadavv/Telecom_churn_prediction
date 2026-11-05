"""
app/views/overview.py

Business overview dashboard page.
Answers how much revenue is at stake and which customers account for it.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app.data import (
    load_dashboard_data,
    load_overview_numbers,
    load_segment_revenue,
)
from app.overview_logic import (
    compute_q1_contract,
    compute_q2_quartiles,
    compute_q3_tenure,
    compute_q4_services,
    compute_q5_net_gain,
    get_overview_kpis,
    get_segment_data,
    get_top_segment_insight,
)
from app.ui import (
    BORDER_COLOR,
    DEEP_TEAL,
    MINT_TINT,
    MUTED_TEXT,
    PRIMARY_TEAL,
    PRIMARY_TEXT,
    apply_plotly_theme,
    card,
    format_currency,
    format_pct,
    kpi,
)

# ----------------- Page Title -----------------
st.title("Business overview")
st.caption("Revenue already lost, expected revenue at risk, and which customer segments account for them.")

# ----------------- Load Data -----------------
overview_numbers = load_overview_numbers()
segment_revenue = load_segment_revenue()
dashboard_df = load_dashboard_data()

# ----------------- 1. Four KPI Cards -----------------
kpis = get_overview_kpis(overview_numbers)

col1, col2, col3, col4 = st.columns(4)
with col1:
    kpi(
        label="Total monthly revenue",
        value=kpis["total_revenue"]["value"],
        caption=str(kpis["total_revenue"]["caption"]),
    )
with col2:
    kpi(
        label="Revenue already lost",
        value=kpis["revenue_lost"]["value"],
        caption=str(kpis["revenue_lost"]["caption"]),
    )
with col3:
    kpi(
        label="Current customers",
        value=kpis["current_customers"]["value"],
        caption=str(kpis["current_customers"]["caption"]),
    )
with col4:
    kpi(
        label="Expected revenue at risk",
        value=kpis["expected_at_risk"]["value"],
        caption=str(kpis["expected_at_risk"]["caption"]),
    )

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# ----------------- 2. Dual Revenue Definitions Callout -----------------
defs = overview_numbers.get("definitions", {})
rev_lost_def = defs.get(
    "revenue_lost",
    "Sum of monthly charges of customers marked churned in the data (already left).",
)
exp_at_risk_def = defs.get(
    "expected_at_risk",
    "Sum of calibrated churn probability x monthly charge over customers still active. Uses out-of-fold scores for training rows. Assumes the churn process continues.",
)

st.markdown(
    f"""
    <div style="background-color: {MINT_TINT}; border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 16px 20px; margin: 8px 0 24px 0;">
        <div style="display: flex; gap: 24px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 280px;">
                <strong style="color: {PRIMARY_TEXT}; font-size: 15px;">Revenue already lost</strong>
                <p style="margin: 6px 0 0 0; color: {PRIMARY_TEXT}; font-size: 13.5px; line-height: 1.5;">{rev_lost_def}</p>
            </div>
            <div style="flex: 1; min-width: 280px;">
                <strong style="color: {PRIMARY_TEXT}; font-size: 15px;">Expected revenue at risk</strong>
                <p style="margin: 6px 0 0 0; color: {PRIMARY_TEXT}; font-size: 13.5px; line-height: 1.5;">{exp_at_risk_def}</p>
            </div>
        </div>
        <div style="margin-top: 14px; border-top: 1px solid #C8DDD8; padding-top: 10px; font-size: 13px; color: {MUTED_TEXT}; font-style: italic;">
            Expected revenue at risk applies the model's churn probability to customers who stayed in this snapshot. It assumes the same churn process continues; it is not a forecast.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------- 3. Segment Explorer -----------------
st.subheader("Segment explorer")

dim_options = ["Contract", "Internet service", "Payment method", "Tenure band"]
dim_keys = {
    "Contract": "contract",
    "Internet service": "internet_service",
    "Payment method": "payment_method",
    "Tenure band": "tenure_band",
}

selected_dim_label = st.radio(
    "Dimension",
    options=dim_options,
    index=0,
    key="overview_dim",
    horizontal=True,
)
selected_dim_key = dim_keys[selected_dim_label]

dim_df = get_segment_data(segment_revenue, selected_dim_key)
insight_text = get_top_segment_insight(dim_df, selected_dim_label)

st.markdown(f"**{insight_text}**")

c_chart, c_table = st.columns([1.1, 1.3])

with c_chart:
    # Horizontal bar chart sorted descending (Plotly y-axis renders bottom to top by default, so reverse)
    plot_df = dim_df.iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=plot_df["revenue_lost"],
            y=plot_df["segment"],
            orientation="h",
            text=[f"{s * 100:.1f}%" for s in plot_df["share_of_lost"]],
            textposition="auto",
            marker_color=PRIMARY_TEAL,
            hoverinfo="x+y",
        )
    )
    fig.update_layout(
        title="Revenue lost per segment",
        xaxis_title="Revenue lost ($)",
        yaxis_title="",
        margin={"l": 20, "r": 20, "t": 40, "b": 30},
        height=300,
    )
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

with c_table:
    # Display formatted table
    table_display = dim_df.copy()
    table_display["churn_rate"] = table_display["churn_rate"].apply(lambda v: format_pct(v, 1))
    table_display["revenue_lost"] = table_display["revenue_lost"].apply(format_currency)
    table_display["share_of_lost"] = table_display["share_of_lost"].apply(lambda v: format_pct(v, 1))
    table_display["expected_at_risk"] = table_display["expected_at_risk"].apply(format_currency)
    table_display = table_display[
        ["segment", "customers", "churn_rate", "revenue_lost", "share_of_lost", "expected_at_risk"]
    ].rename(
        columns={
            "segment": "Segment",
            "customers": "Customers",
            "churn_rate": "Churn rate",
            "revenue_lost": "Revenue lost",
            "share_of_lost": "Share of lost",
            "expected_at_risk": "Expected at risk",
        }
    )
    st.dataframe(table_display, use_container_width=True, hide_index=True)

st.caption("Shares add to 100% within a dimension, not across dimensions: the dimensions overlap.")
st.caption("These figures describe where revenue was lost, not why. A segment is not a cause.")

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# ----------------- 4. Five Business Questions -----------------
st.subheader("Five key business questions")

row1_left, row1_right = st.columns(2)

with row1_left, card("q1_card"):
    st.markdown("**Q1: Do month-to-month customers churn more than annual-contract customers?**")
    st.write(
        "Yes. Month-to-month customers churn at 42.7%, compared to 11.3% for one-year and 2.8% for two-year contracts."
    )
    q1_df = compute_q1_contract(dashboard_df)
    fig_q1 = go.Figure(
        go.Bar(
            x=q1_df["contract"],
            y=q1_df["churn_rate"],
            text=[f"{r * 100:.1f}%" for r in q1_df["churn_rate"]],
            textposition="auto",
            marker_color=[DEEP_TEAL, PRIMARY_TEAL, "#4F8A85"],
        )
    )
    fig_q1.update_layout(
        yaxis_tickformat=".0%",
        yaxis_title="Churn rate",
        margin={"l": 10, "r": 10, "t": 20, "b": 20},
        height=220,
    )
    apply_plotly_theme(fig_q1)
    st.plotly_chart(fig_q1, use_container_width=True)

with row1_right, card("q2_card"):
    st.markdown("**Q2: Does a higher monthly bill go with higher churn?**")
    q2_df, q2_labels = compute_q2_quartiles(dashboard_df)
    st.write(
        f"Yes. Churn peaks at 37.5% in quartile 3 ({q2_labels[2]}) and 32.9% in quartile 4 ({q2_labels[3]}), roughly triple quartile 1 ({q2_labels[0]}) at 11.2%."
    )
    fig_q2 = go.Figure(
        go.Bar(
            x=[f"Q{i+1}<br>{lbl}" for i, lbl in enumerate(q2_labels)],
            y=q2_df["churn_rate"],
            text=[f"{r * 100:.1f}%" for r in q2_df["churn_rate"]],
            textposition="auto",
            marker_color=PRIMARY_TEAL,
        )
    )
    fig_q2.update_layout(
        yaxis_tickformat=".0%",
        yaxis_title="Churn rate",
        margin={"l": 10, "r": 10, "t": 20, "b": 20},
        height=220,
    )
    apply_plotly_theme(fig_q2)
    st.plotly_chart(fig_q2, use_container_width=True)

row2_left, row2_right = st.columns(2)

with row2_left, card("q3_card"):
    st.markdown("**Q3: How much does tenure reduce churn risk?**")
    st.write(
        "Significantly. Customers in their first year churn at 48.3%, falling to 26.1% in years 2–3, and 12.1% after 3 years."
    )
    q3_df = compute_q3_tenure(dashboard_df)
    fig_q3 = go.Figure(
        go.Bar(
            x=[f"{b} mos" for b in q3_df["tenure_band"]],
            y=q3_df["churn_rate"],
            text=[f"{r * 100:.1f}%" for r in q3_df["churn_rate"]],
            textposition="auto",
            marker_color=[DEEP_TEAL, PRIMARY_TEAL, "#4F8A85"],
        )
    )
    fig_q3.update_layout(
        yaxis_tickformat=".0%",
        yaxis_title="Churn rate",
        margin={"l": 10, "r": 10, "t": 20, "b": 20},
        height=220,
    )
    apply_plotly_theme(fig_q3)
    st.plotly_chart(fig_q3, use_container_width=True)


with row2_right, card("q4_card"):
    st.markdown("**Q4: Which service features go with retention?**")
    st.write(
        "Fiber optic and Electronic check customers have the highest churn "
        "rates. DSL, customers without internet service, and automatic "
        "payment methods show lower churn rates."
    )

    q4_data = compute_q4_services(dashboard_df)
    q4_internet = q4_data["internet"]
    q4_payment = q4_data["payment"]

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.caption("Churn rate by internet service")
        fig_q4_internet = go.Figure(
            go.Bar(
                x=q4_internet["internet_service"],
                y=q4_internet["churn_rate"],
                text=[f"{r * 100:.1f}%" for r in q4_internet["churn_rate"]],
                textposition="auto",
                marker_color=PRIMARY_TEAL,
            )
        )
        fig_q4_internet.update_layout(
            yaxis_tickformat=".0%",
            yaxis_title="Churn rate",
            margin={"l": 10, "r": 10, "t": 20, "b": 20},
            height=260,
        )
        apply_plotly_theme(fig_q4_internet)
        st.plotly_chart(
            fig_q4_internet, use_container_width=True
        )

    with chart_col2:
        st.caption("Churn rate by payment method")
        fig_q4_payment = go.Figure(
            go.Bar(
                x=q4_payment["payment_method"],
                y=q4_payment["churn_rate"],
                text=[f"{r * 100:.1f}%" for r in q4_payment["churn_rate"]],
                textposition="auto",
                marker_color=PRIMARY_TEAL,
            )
        )
        fig_q4_payment.update_layout(
            yaxis_tickformat=".0%",
            yaxis_title="Churn rate",
            margin={"l": 10, "r": 10, "t": 20, "b": 20},
            height=260,
        )
        apply_plotly_theme(fig_q4_payment)
        st.plotly_chart(
            fig_q4_payment, use_container_width=True
        )


with card("q5_card"):
    st.markdown("**Q5: If we retain 50% of predicted churners with a discount, how much is saved?**")
    q5_gain = compute_q5_net_gain(dashboard_df, r=0.5, d=0.2, months=12.0, c=0.0)
    st.write(
        f"A targeted retention campaign using the model ranked by value on held-out customers saves a net **${format_currency(q5_gain)}** under default assumptions (50% success rate, 20% discount, 12 months)."
    )
    st.info("Open the simulator to change the assumptions (retention rate, discount, contact costs, capacity).")

# ----------------- 5. Footnote -----------------
st.caption("These figures describe monthly revenue, not yearly totals.")