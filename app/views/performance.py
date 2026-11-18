"""
app/views/performance.py

Model performance dashboard page.
Evaluates ranking quality, probability calibration, risk bands, segmentation weaknesses,
fairness across subgroups, feature importance, and the formal model card.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data import (
    load_band_table,
    load_calibration_curve,
    load_cv_results,
    load_gains_curve,
    load_metrics,
    load_pr_curve,
    load_roc_curve,
    load_segment_metrics,
    load_shap_importance,
)
from app.performance_logic import (
    find_weakest_segment,
    format_band_table,
    get_fairness_tables,
    get_model_comparison_data,
    get_performance_kpis,
    get_top_shap_features,
)
from app.ui import (
    BAND_COLORS,
    DEEP_TEAL,
    MINT_TINT,
    MUTED_TEXT,
    PRIMARY_TEAL,
    PRIMARY_TEXT,
    apply_plotly_theme,
    card,
    format_pct,
    kpi,
)
from churn_core import config

# ----------------- Page Title & Caption -----------------
st.title("Model performance")
st.caption(
    "How good the model is, how well calibrated, and where it is weakest. "
    "This page ignores the sidebar assumptions, except where noted: all figures on the 1,409 held-out customers unless stated."
)

# ----------------- Load Data -----------------
metrics = load_metrics()
cv_df = load_cv_results()
pr_df = load_pr_curve()
gains_df = load_gains_curve()
roc_df = load_roc_curve()
calib_df = load_calibration_curve()
band_df = load_band_table()
seg_df = load_segment_metrics()
shap_df = load_shap_importance()

# ----------------- 1. Top KPI Row -----------------
kpis = get_performance_kpis(metrics)
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    kpi(label="AUC-PR", value=kpis["auc_pr"]["display"], caption=str(kpis["auc_pr"]["caption"]))
with c2:
    kpi(label="ROC-AUC", value=kpis["roc_auc"]["display"], caption=str(kpis["roc_auc"]["caption"]))
with c3:
    kpi(label="Brier score", value=kpis["brier"]["display"], caption=str(kpis["brier"]["caption"]))
with c4:
    kpi(label="Recall / Precision", value=kpis["recall_precision"]["display"], caption=str(kpis["recall_precision"]["caption"]))
with c5:
    kpi(label="Flagged", value=kpis["flagged"]["display"], caption=str(kpis["flagged"]["caption"]))

st.caption("All figures on the 1,409 held-out customers.")
st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# ----------------- 2. Model Comparison -----------------
st.subheader("Model comparison (cross-validation)")
comp_df = get_model_comparison_data(cv_df)

metric_view = st.radio(
    "Metric",
    options=["AUC-PR (primary)", "ROC-AUC"],
    index=0,
    horizontal=True,
    key="perf_comp_metric",
)

target_col = "auc_pr" if "AUC-PR" in metric_view else "roc_auc"
target_std = f"{target_col}_std"

fig_comp = go.Figure(
    go.Bar(
        y=comp_df["clean_name"],
        x=comp_df[target_col],
        orientation="h",
        error_x={"type": "data", "array": comp_df[target_std], "visible": True, "color": MUTED_TEXT},
        marker_color=[PRIMARY_TEAL if is_fin else "#94A3B8" for is_fin in comp_df["is_finalist"]],
        text=[f"{v:.3f}" for v in comp_df[target_col]],
        textposition="inside",
        hoverinfo="x+y",
    )
)
fig_comp.update_layout(
    title=f"5-Fold Cross-Validation {target_col.upper().replace('_', '-')} by Pipeline",
    xaxis_title=target_col.upper().replace("_", "-"),
    yaxis_title="",
    margin={"l": 20, "r": 20, "t": 40, "b": 30},
    height=340,
)
apply_plotly_theme(fig_comp)
st.plotly_chart(fig_comp, use_container_width=True)

st.caption("Cross-validation on the training data only. The test set is used once, for the final model.")
st.markdown(
    f"""
    <div style="background-color: {MINT_TINT}; border-left: 3px solid {PRIMARY_TEAL}; padding: 10px 14px; margin-bottom: 24px; font-size: 13.5px; color: {PRIMARY_TEXT};">
        A class-weighted logistic regression scored about the same as untuned LightGBM, so the signal is largely additive.
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------- 3. Ranking Quality -----------------
st.subheader("Ranking quality")
c_pr, c_gains = st.columns(2)

with c_pr, card("pr_card"):
    st.markdown("**Precision-Recall Curve**")
    fig_pr = go.Figure()
    fig_pr.add_trace(
        go.Scatter(
            x=pr_df["recall"],
            y=pr_df["precision"],
            mode="lines",
            line={"color": PRIMARY_TEAL, "width": 2.5},
            name="LightGBM (Calibrated)",
        )
    )
    # Baseline no-skill line (test set churn rate: 374 / 1409 = 0.2654)
    base_rate = metrics["n_churners"] / metrics["n_test"]
    fig_pr.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[base_rate, base_rate],
            mode="lines",
            line={"color": "#94A3B8", "dash": "dash"},
            name=f"Baseline ({base_rate:.1%})",
        )
    )
    # Operating cutoff point marker
    fig_pr.add_trace(
        go.Scatter(
            x=[metrics["recall"]],
            y=[metrics["precision"]],
            mode="markers+text",
            marker={"size": 10, "color": DEEP_TEAL},
            text=["Cutoff (p=0.33)"],
            textposition="top right",
            name="Cutoff",
        )
    )
    fig_pr.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis_range=[0, 1.02],
        yaxis_range=[0, 1.02],
        margin={"l": 15, "r": 15, "t": 20, "b": 20},
        height=300,
    )
    apply_plotly_theme(fig_pr)
    st.plotly_chart(fig_pr, use_container_width=True)

with c_gains, card("gains_card"):
    st.markdown("**Cumulative Gains Curve**")
    fig_gains = go.Figure()
    fig_gains.add_trace(
        go.Scatter(
            x=gains_df["frac_customers"],
            y=gains_df["frac_churners"],
            mode="lines",
            line={"color": PRIMARY_TEAL, "width": 2.5},
            name="Model",
        )
    )
    # Random diagonal
    fig_gains.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"color": "#94A3B8", "dash": "dash"},
            name="Random",
        )
    )
    # Operating cutoff marker
    flag_frac = metrics["flagged"] / metrics["n_test"]
    fig_gains.add_trace(
        go.Scatter(
            x=[flag_frac],
            y=[metrics["recall"]],
            mode="markers+text",
            marker={"size": 10, "color": DEEP_TEAL},
            text=[f"Top {flag_frac:.0%}: {metrics['recall']:.0%} caught"],
            textposition="bottom right",
            name="Cutoff",
        )
    )
    fig_gains.update_layout(
        xaxis_title="Fraction of Customers Contacted",
        yaxis_title="Fraction of Churners Caught",
        xaxis_range=[0, 1.02],
        yaxis_range=[0, 1.02],
        margin={"l": 15, "r": 15, "t": 20, "b": 20},
        height=300,
    )
    apply_plotly_theme(fig_gains)
    st.plotly_chart(fig_gains, use_container_width=True)

with st.expander("Receiver Operating Characteristic (ROC) curve"):
    fig_roc = go.Figure()
    fig_roc.add_trace(
        go.Scatter(
            x=roc_df["fpr"],
            y=roc_df["tpr"],
            mode="lines",
            line={"color": PRIMARY_TEAL, "width": 2.5},
            name=f"ROC (AUC = {metrics['roc_auc']:.3f})",
        )
    )
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"color": "#94A3B8", "dash": "dash"},
            name="Random (AUC = 0.50)",
        )
    )
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis_range=[0, 1.02],
        yaxis_range=[0, 1.02],
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=320,
    )
    apply_plotly_theme(fig_roc)
    st.plotly_chart(fig_roc, use_container_width=True)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# ----------------- 4. Calibration -----------------
st.subheader("Probability calibration")

c_calib_plot, c_calib_text = st.columns([1.3, 1.1])

with c_calib_plot:
    fig_calib = go.Figure()
    # Diagonal reference line
    fig_calib.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"color": "#94A3B8", "dash": "dash"},
            name="Perfect calibration",
        )
    )
    # Uncalibrated line
    uncal_df = calib_df[calib_df["model"] == "uncalibrated"]
    fig_calib.add_trace(
        go.Scatter(
            x=uncal_df["mean_pred"],
            y=uncal_df["frac_pos"],
            mode="lines+markers",
            line={"color": "#E06D53", "width": 2, "dash": "dot"},
            marker={"size": 6},
            name=f"Before calibration (Brier: {metrics['brier_uncalibrated']:.3f})",
        )
    )
    # Calibrated (sigmoid) line
    sig_df = calib_df[calib_df["model"] == "sigmoid"]
    fig_calib.add_trace(
        go.Scatter(
            x=sig_df["mean_pred"],
            y=sig_df["frac_pos"],
            mode="lines+markers",
            line={"color": PRIMARY_TEAL, "width": 2.5},
            marker={"size": 7},
            name=f"After calibration (Brier: {metrics['brier_sigmoid']:.3f})",
        )
    )
    fig_calib.update_layout(
        title="Reliability diagram",
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed fraction of positives",
        xaxis_range=[0, 1.0],
        yaxis_range=[0, 1.0],
        margin={"l": 20, "r": 20, "t": 35, "b": 20},
        height=320,
    )
    apply_plotly_theme(fig_calib)
    st.plotly_chart(fig_calib, use_container_width=True)

with c_calib_text, card("calib_info"):
    st.markdown("**Why calibration matters**")
    st.write(
        "Class weights make the model good at ranking customers but inflate its probabilities. "
        "Calibration fixes that, so a probability of 40% really means about 40 in 100. "
        "This matters because the money calculation multiplies by these probabilities."
    )
    st.markdown(
        f"- **Uncalibrated Brier score**: `{metrics['brier_uncalibrated']:.3f}`\n"
        f"- **Sigmoid Brier score**: `{metrics['brier_sigmoid']:.3f}` *(lower is better)*\n"
        f"- **Optimal cutoff formula**: $d / (d + r(1-d)) = 1/3$"
    )

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# ----------------- 5. Risk Bands -----------------
st.subheader("Risk bands")
st.write(
    "Observed churn rises with every band, so the bands mean something. "
    "The edges at 0.33 comes from the cost rule; the other two edges are score percentiles."
)

c_band_chart, c_band_tables = st.columns([1.1, 1.3])

with c_band_chart:
    test_bands = format_band_table(band_df, sample="test")
    band_palette = [BAND_COLORS.get(b, PRIMARY_TEAL) for b in test_bands["band"]]
    fig_bands = go.Figure(
        go.Bar(
            x=test_bands["band"],
            y=test_bands["observed_churn_rate"],
            text=[f"{r:.1%}<br>({c:,})" for r, c in zip(test_bands["observed_churn_rate"], test_bands["customers"])],
            textposition="auto",
            marker_color=band_palette,
            hoverinfo="x+y",
        )
    )
    fig_bands.update_layout(
        title="Observed Churn Rate by Risk Band (Held-out Test)",
        yaxis_tickformat=".0%",
        yaxis_title="Observed Churn Rate",
        xaxis_title="Risk Band",
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        height=320,
    )
    apply_plotly_theme(fig_bands)
    st.plotly_chart(fig_bands, use_container_width=True)

with c_band_tables:
    st.markdown("**Held-out test sample (N=1,409)**")
    t_test = test_bands.copy()
    t_test["share"] = t_test["share"].apply(lambda v: format_pct(v, 1))
    t_test["observed_churn_rate"] = t_test["observed_churn_rate"].apply(lambda v: format_pct(v, 1))
    t_test["mean_predicted"] = t_test["mean_predicted"].apply(lambda v: format_pct(v, 1))
    st.dataframe(
        t_test.rename(
            columns={
                "band": "Band",
                "customers": "Customers",
                "share": "Share",
                "observed_churn_rate": "Observed churn",
                "mean_predicted": "Mean predicted",
            }
        ).drop(columns=["sample"], errors="ignore"),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Out-of-fold training sample (N=5,634)**")
    t_oof = format_band_table(band_df, sample="oof")
    t_oof["share"] = t_oof["share"].apply(lambda v: format_pct(v, 1))
    t_oof["observed_churn_rate"] = t_oof["observed_churn_rate"].apply(lambda v: format_pct(v, 1))
    t_oof["mean_predicted"] = t_oof["mean_predicted"].apply(lambda v: format_pct(v, 1))
    st.dataframe(
        t_oof.rename(
            columns={
                "band": "Band",
                "customers": "Customers",
                "share": "Share",
                "observed_churn_rate": "Observed churn",
                "mean_predicted": "Mean predicted",
            }
        ).drop(columns=["sample"], errors="ignore"),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# ----------------- 6. Where the Model is Weaker -----------------
st.subheader("Where the model is weaker")

weakest_info = find_weakest_segment(seg_df)
weak_contract = seg_df[seg_df["group_by"] == "contract"].copy()
weak_tenure = seg_df[seg_df["group_by"] == "tenure_band"].copy()

c_weak_cont, c_weak_ten = st.columns(2)

with c_weak_cont, card("weak_cont_card"):
    st.markdown("**By Contract Type**")
    fig_wc = go.Figure()
    fig_wc.add_trace(
        go.Bar(
            name="Recall",
            x=weak_contract["group"],
            y=weak_contract["recall"],
            text=[f"{r:.1%}" if pd.notna(r) else "—" for r in weak_contract["recall"]],
            textposition="auto",
            marker_color=PRIMARY_TEAL,
        )
    )
    fig_wc.add_trace(
        go.Bar(
            name="Precision",
            x=weak_contract["group"],
            y=weak_contract["precision"],
            text=[f"{p:.1%}" if pd.notna(p) else "—" for p in weak_contract["precision"]],
            textposition="auto",
            marker_color="#94A3B8",
        )
    )
    fig_wc.update_layout(
        barmode="group",
        yaxis_tickformat=".0%",
        yaxis_title="Score",
        margin={"l": 10, "r": 10, "t": 20, "b": 20},
        height=240,
    )
    apply_plotly_theme(fig_wc)
    st.plotly_chart(fig_wc, use_container_width=True)

with c_weak_ten, card("weak_ten_card"):
    st.markdown("**By Tenure Band**")
    fig_wt = go.Figure()
    fig_wt.add_trace(
        go.Bar(
            name="Recall",
            x=[f"{b} mos" for b in weak_tenure["group"]],
            y=weak_tenure["recall"],
            text=[f"{r:.1%}" if pd.notna(r) else "—" for r in weak_tenure["recall"]],
            textposition="auto",
            marker_color=PRIMARY_TEAL,
        )
    )
    fig_wt.add_trace(
        go.Bar(
            name="Precision",
            x=[f"{b} mos" for b in weak_tenure["group"]],
            y=weak_tenure["precision"],
            text=[f"{p:.1%}" if pd.notna(p) else "—" for p in weak_tenure["precision"]],
            textposition="auto",
            marker_color="#94A3B8",
        )
    )
    fig_wt.update_layout(
        barmode="group",
        yaxis_tickformat=".0%",
        yaxis_title="Score",
        margin={"l": 10, "r": 10, "t": 20, "b": 20},
        height=240,
    )
    apply_plotly_theme(fig_wt)
    st.plotly_chart(fig_wt, use_container_width=True)

st.caption(
    f"Weakest segment: {weakest_info['group']} ({weakest_info['dimension']}) with recall of "
    f"{weakest_info['recall']:.1%} ({weakest_info['caught']} of {weakest_info['churners']} churners caught)."
)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# ----------------- 7. Fairness Check -----------------
st.subheader("Fairness check")

gender_table, senior_table = get_fairness_tables(seg_df)
c_fair_gen, c_fair_sen = st.columns(2)

with c_fair_gen, card("fair_gender_card"):
    st.markdown("**By Gender**")
    disp_gen = gender_table.copy()
    disp_gen["Churn rate"] = disp_gen["Churn rate"].apply(lambda v: format_pct(v, 1))
    disp_gen["Flag rate"] = disp_gen["Flag rate"].apply(lambda v: format_pct(v, 1))
    disp_gen["Recall"] = disp_gen["Recall"].apply(lambda v: format_pct(v, 1))
    disp_gen["Precision"] = disp_gen["Precision"].apply(lambda v: format_pct(v, 1))
    st.dataframe(disp_gen, use_container_width=True, hide_index=True)

with c_fair_sen, card("fair_senior_card"):
    st.markdown("**By Senior Citizen Status**")
    disp_sen = senior_table.copy()
    disp_sen["Churn rate"] = disp_sen["Churn rate"].apply(lambda v: format_pct(v, 1))
    disp_sen["Flag rate"] = disp_sen["Flag rate"].apply(lambda v: format_pct(v, 1))
    disp_sen["Recall"] = disp_sen["Recall"].apply(lambda v: format_pct(v, 1))
    disp_sen["Precision"] = disp_sen["Precision"].apply(lambda v: format_pct(v, 1))
    st.dataframe(disp_sen, use_container_width=True, hide_index=True)

st.caption(
    "Gender is a model input, but its measured effect on predictions is zero. "
    "Senior citizen is used by the model, so the table shows how its flags and recall differ. "
    "A difference in flag rate is not automatically unfair: it can reflect different churn rates. "
    "Check the churn-rate column too."
)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# ----------------- 8. What Drives the Model (SHAP) -----------------
st.subheader("What drives the model")

top_shap = get_top_shap_features(shap_df, top_n=15).iloc[::-1]

fig_shap = go.Figure(
    go.Bar(
        x=top_shap["importance"],
        y=top_shap["display_feature"],
        orientation="h",
        marker_color=PRIMARY_TEAL,
        text=[f"{v:.3f}" for v in top_shap["importance"]],
        textposition="auto",
        hoverinfo="x+y",
    )
)
fig_shap.update_layout(
    title="Top 15 Features by Mean Absolute SHAP Value",
    xaxis_title="Mean |SHAP value| (log-odds impact)",
    yaxis_title="",
    margin={"l": 20, "r": 20, "t": 35, "b": 20},
    height=420,
)
apply_plotly_theme(fig_shap)
st.plotly_chart(fig_shap, use_container_width=True)

st.caption("Describes what the model learned to rely on, not causal drivers.")

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# ----------------- 9. Model Card -----------------
card_path = config.home() / "app" / "content" / "model_card.md"
model_card_text = ""
if card_path.is_file():
    model_card_text = card_path.read_text(encoding="utf-8")
else:
    model_card_text = "Model card documentation file is missing."

with st.expander("Model card", expanded=True):
    st.markdown(model_card_text)