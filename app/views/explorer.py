"""
app/views/explorer.py

Customer risk explorer dashboard page.
Features:
A. Ranked customer table with interactive filters and expected gain sorting.
B. Batch CSV file scoring via prediction API with data-health diagnostics and export.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app.api_client import ApiError, predict_batch, upload_template
from app.data import load_dashboard_data
from app.explorer_logic import (
    compute_explorer_summary,
    filter_customers,
    format_batch_results,
    neutralise_csv,
    rank_by_expected_gain,
)
from app.ui import (
    BAND_COLORS,
    PRIMARY_TEAL,
    apply_plotly_theme,
    card,
    format_currency,
    format_pct,
    kpi,
)

# ----------------- Sidebar Assumptions -----------------
r = float(st.session_state.get("r", 0.5))
d = float(st.session_state.get("d", 0.2))
months = float(st.session_state.get("months", 12))

# ----------------- Title -----------------
st.title("Customer risk explorer")
st.caption("A ranked list of customers, and scoring for a file you upload.")

tab_ranked, tab_upload = st.tabs(["A. Ranked customers", "B. Score a file"])

# =========================================================================
# TAB A: Ranked Customers
# =========================================================================
with tab_ranked:
    full_df = load_dashboard_data()

    # Filters row
    with card("filter_card"):
        f_col1, f_col2, f_col3 = st.columns([1.4, 1.2, 1.2])

        with f_col1:
            pop_choice = st.radio(
                "Population",
                options=[
                    "Held-out customers only (1,409)",
                    "All customers (7,043; training rows are out-of-fold)",
                ],
                index=0,
                key="exp_population",
            )
        with f_col2:
            bands_choice = st.multiselect(
                "Risk bands",
                options=["Low", "Medium", "High", "Very high"],
                default=["Low", "Medium", "High", "Very high"],
                key="exp_bands",
            )
            min_prob = float(
                st.slider(
                    "Minimum churn probability",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                    key="exp_min_prob",
                )
            )
        with f_col3:
            contracts_choice = st.multiselect(
                "Contract type",
                options=["Month-to-month", "One year", "Two year"],
                default=["Month-to-month", "One year", "Two year"],
                key="exp_contract",
            )
            tenure_choice = st.multiselect(
                "Tenure band",
                options=["0-11", "12-35", "36-72"],
                default=["0-11", "12-35", "36-72"],
                key="exp_tenure",
            )

    if "All customers" in pop_choice:
        st.info("Most customers here were used to train the model, so their scores are out-of-fold, a fair but slightly optimistic estimate.")

    # Filter & rank
    filtered_df = filter_customers(
        full_df,
        population=pop_choice,
        bands=bands_choice,
        contracts=contracts_choice,
        tenure_bands=tenure_choice,
        min_prob=min_prob,
    )

    ranked_df = rank_by_expected_gain(filtered_df, r=r, d=d, months=months)
    summary = compute_explorer_summary(ranked_df)

    # Summary strip
    s1, s2, s3 = st.columns(3)
    with s1:
        kpi("Customers matching", summary["formatted_matching"], caption="Filtered population")
    with s2:
        kpi(
            "Recommended to contact",
            summary["formatted_contact"],
            caption=f"Expected gain > $0 (p ≥ {d / (d + r * (1 - d)):.2f})",
        )
    with s3:
        kpi("Total expected net gain", summary["formatted_gain"], caption=f"At r={r:.0%}, d={d:.0%}, {months:.0f} mos")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Display Table
    n_display = min(len(ranked_df), 500)
    st.write(f"Showing top **{n_display:,}** of **{len(ranked_df):,}** matching customers, ranked by expected net gain.")

    if not ranked_df.empty:
        display_sub = ranked_df.head(500).copy()
        display_sub["Churn probability"] = display_sub["probability"].apply(lambda v: format_pct(v, 1))
        display_sub["Contact?"] = display_sub["contact"].apply(lambda v: "Yes" if v else "No")
        display_sub["Monthly bill"] = display_sub["monthly_charges"].apply(format_currency)
        display_sub["Tenure"] = display_sub["tenure"].apply(lambda v: f"{v} mos")
        display_sub["Expected net gain"] = display_sub["expected_gain"].apply(format_currency)
        display_sub["Actual (known)"] = display_sub["actual_churn"].apply(lambda v: "Churned" if v == 1 else "Stayed")

        cols_to_show = [
            "customer_id",
            "Churn probability",
            "band",
            "Contact?",
            "Monthly bill",
            "contract",
            "Tenure",
            "Expected net gain",
            "Actual (known)",
        ]
        col_names = {
            "customer_id": "Customer ID",
            "band": "Risk band",
            "contract": "Contract",
        }
        st.dataframe(
            display_sub[cols_to_show].rename(columns=col_names),
            use_container_width=True,
            hide_index=True,
        )

        csv_data = neutralise_csv(ranked_df)
        st.download_button(
            "Download filtered customers (CSV)",
            data=csv_data,
            file_name="filtered_customers.csv",
            mime="text/csv",
            help="Full filtered list (formula-injection neutralized)",
        )
    else:
        st.warning("No customers match the selected filter criteria.")

# =========================================================================
# TAB B: Score a File
# =========================================================================
with tab_upload:
    st.markdown("**Do not upload real customer data. Files are scored in memory and never stored.**")

    # Template download section
    try:
        template_text = upload_template()
        st.download_button(
            "Download CSV template",
            data=template_text,
            file_name="churn_upload_template.csv",
            mime="text/csv",
            help="Download the expected column format for batch scoring",
        )
    except ApiError as exc:
        st.warning(f"Could not load template from API: {exc}")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload customer CSV file (Max 2 MB, 5,000 rows)",
        type=["csv"],
        key="exp_file_upload",
    )

    include_reasons = st.checkbox(
        "Include the top reasons for each row (slower)",
        value=False,
        key="exp_include_reasons",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        if len(file_bytes) > 2_000_000:
            st.error("File is larger than 2 MB. Please reduce file size before uploading.")
        else:
            if st.button("Score this file", type="primary", key="exp_score_btn"):
                with st.spinner("Scoring batch file via prediction service..."):
                    try:
                        resp = predict_batch(
                            data=file_bytes,
                            filename=uploaded_file.name,
                            r=r,
                            d=d,
                            months=months,
                            reasons=include_reasons,
                        )
                        results = format_batch_results(resp)

                        # Check if any row was scored
                        if results["n_scored"] == 0:
                            st.error("No row could be scored. Please check the data-health errors below.")
                        else:
                            st.success(f"Successfully scored {results['n_scored']:,} customer rows.")

                        # Summary KPIs
                        b1, b2, b3, b4, b5 = st.columns(5)
                        with b1:
                            kpi("Rows scored", f"{results['n_scored']:,}")
                        with b2:
                            kpi("Rows with errors", f"{results['n_errors']:,}")
                        with b3:
                            kpi("Rows with warnings", f"{results['n_warnings']:,}")
                        with b4:
                            kpi("Customers to contact", f"{results['n_contact']:,}")
                        with b5:
                            kpi("Total expected gain", f"${round(results['total_gain']):,}")

                        # Data Health Panel
                        issues_df = results["issues_df"]
                        if not issues_df.empty:
                            with st.expander(f"Data-health report ({len(issues_df)} issues)", expanded=True):
                                st.caption("Errors: the row was not scored. Warnings: scored, but the value looks unusual.")
                                st.dataframe(issues_df, use_container_width=True, hide_index=True)
                                if results["issues_truncated"]:
                                    st.warning("Issue list truncated.")

                        # Results Table & Chart
                        scored_df = results["scored_df"]
                        if not scored_df.empty:
                            c_bchart, c_btable = st.columns([1, 1.4])
                            with c_bchart:
                                band_counts = results["band_counts"]
                                fig_batch = go.Figure(
                                    go.Bar(
                                        x=list(band_counts.keys()),
                                        y=list(band_counts.values()),
                                        marker_color=[BAND_COLORS.get(b, PRIMARY_TEAL) for b in band_counts],
                                        text=list(band_counts.values()),
                                        textposition="auto",
                                    )
                                )
                                fig_batch.update_layout(
                                    title="Customers by Risk Band",
                                    xaxis_title="Risk Band",
                                    yaxis_title="Count",
                                    margin={"l": 10, "r": 10, "t": 35, "b": 10},
                                    height=300,
                                )
                                apply_plotly_theme(fig_batch)
                                st.plotly_chart(fig_batch, use_container_width=True)

                            with c_btable:
                                st.markdown("**Scored customer predictions**")
                                disp_scored = scored_df.copy()
                                if "probability" in disp_scored.columns:
                                    disp_scored["probability"] = disp_scored["probability"].apply(lambda v: format_pct(v, 1))
                                if "expected_gain" in disp_scored.columns:
                                    disp_scored["expected_gain"] = disp_scored["expected_gain"].apply(format_currency)
                                if "flag" in disp_scored.columns:
                                    disp_scored["flag"] = disp_scored["flag"].apply(lambda v: "Yes" if v else "No")

                                st.dataframe(disp_scored, use_container_width=True, hide_index=True)

                            # Scored CSV Export
                            export_df = scored_df.copy()
                            export_df["model_version"] = resp.get("model_version", "1.0.0")
                            export_df["r"] = r
                            export_df["d"] = d
                            export_df["months"] = months

                            csv_out = neutralise_csv(export_df)
                            st.download_button(
                                "Download scored results (CSV)",
                                data=csv_out,
                                file_name="scored_customers.csv",
                                mime="text/csv",
                                help="Sanitized scored results with campaign assumptions",
                            )

                    except ApiError as exc:
                        st.error(f"API scoring failed: {exc}")