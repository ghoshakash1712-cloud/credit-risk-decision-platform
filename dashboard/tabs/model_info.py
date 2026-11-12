import streamlit as st
import pandas as pd
import altair as alt
from dashboard.client import get, safe

def render(params: dict, meta: dict):
    st.header("Model Performance, Lineage & Governance")
    st.markdown("Inspect global feature importance, discrimination and calibration statistics across splits, data lineage, and model operational limitations.")

    # Top lineage info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Version", meta.get("model_version", "N/A"))
    with col2:
        st.metric("Freddie Mac Data Release", f"Release {meta.get('data_release', 'N/A')}")
    with col3:
        st.metric("Target Variable", "default_24m (90+ DPD within 24m)")

    st.divider()

    # Global Feature Importance
    st.subheader("Global Feature Importance (TreeSHAP Mean |Contribution|)")
    imp_data, _ = safe(get, "/explain/global", top=20)
    
    if imp_data and imp_data.get("features"):
        df_imp = pd.DataFrame(imp_data["features"])
        chart = alt.Chart(df_imp).mark_bar(color="#4E79A7").encode(
            x=alt.X("mean_abs_contribution:Q", title="Mean Absolute TreeSHAP Contribution"),
            y=alt.Y("feature:N", sort="-x", title="Feature Name"),
            tooltip=["feature", alt.Tooltip("mean_abs_contribution:Q", format=".5f")]
        ).properties(
            title="Top 20 Features Driving Credit Risk Decisions Globally",
            height=380
        )
        st.altair_chart(chart, use_container_width=True)

        st.markdown("**Global Feature Importance Table**")
        st.dataframe(df_imp, use_container_width=True, hide_index=True)

    st.divider()

    # Model Performance across Splits
    st.subheader("Model Discrimination & Calibration Across Temporal Splits")
    rep, _ = safe(get, "/model-report")

    if rep:
        split_summary = []
        for s in ["valid", "oot", "stress"]:
            if s in rep:
                s_data = rep[s]
                split_summary.append({
                    "Split": s.upper(),
                    "Sample Size (N)": f"{s_data['n_loans']:,}",
                    "Defaults": f"{s_data['n_defaults']:,}",
                    "Default Rate": f"{s_data['default_rate']*100:.2f}%",
                    "AUC-ROC": f"{s_data['auc']:.4f}",
                    "Brier Score": f"{s_data['brier_score']:.4f}",
                    "KS Statistic": f"{s_data['ks_stat']:.4f}"
                })
        
        st.dataframe(pd.DataFrame(split_summary), use_container_width=True, hide_index=True)

        # Decile Performance Table
        st.subheader("OOT Decile Ordering (2019 Vintage)")
        if "oot" in rep and "deciles" in rep["oot"]:
            df_dec = pd.DataFrame(rep["oot"]["deciles"])
            df_disp = df_dec.rename(columns={
                "decile": "Decile",
                "loans": "Loans",
                "defaults": "Defaults",
                "empirical_default_rate": "Observed Default Rate",
                "mean_predicted_probability": "Mean Predicted Prob",
                "cum_defaults_captured": "Cum Defaults Captured",
                "cum_capture_rate": "Cum Capture Rate (Recall)"
            })
            df_disp["Observed Default Rate"] = df_disp["Observed Default Rate"].apply(lambda x: f"{x*100:.2f}%")
            df_disp["Mean Predicted Prob"] = df_disp["Mean Predicted Prob"].apply(lambda x: f"{x*100:.2f}%")
            df_disp["Cum Capture Rate (Recall)"] = df_disp["Cum Capture Rate (Recall)"].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(df_disp, use_container_width=True, hide_index=True)

    # Disclaimers and Limitations
    st.divider()
    st.subheader("Governance, Licensing & Model Limitations (SRS 1.4 & DT-09)")
    st.info(
        "📚 **Portfolio Project & Academic Demonstration:** This application is developed for educational and portfolio demonstration "
        "purposes. It does not constitute lending advice or credit underwriting software for live loan origination decisions."
    )
    st.caption(
        "• **Proxy Limitations (FA-01):** The dataset does not include direct protected demographic attributes (Race, Ethnicity, Sex, Age). "
        "Fairness evaluations use Freddie Mac single-family loan proxies (First-time homebuyer status, borrower count, occupancy, loan purpose). "
        "Disparities in proxies must not be interpreted as definitive proof of disparate impact without holistic demographic surveys.\n\n"
        "• **Freddie Mac Data Licensing Terms (DT-09):** The underlying loan-level data is provided by Freddie Mac under the Single-Family "
        "Loan-Level Dataset terms of use for academic and research purposes. Redistribution of raw loan records is strictly prohibited.\n\n"
        "• **Macroeconomic Regime Shifts:** Model scores were calibrated on fixed 2015–2018 origination vintages. In severe interest rate "
        "or unemployment shocks (such as 2020–2021 COVID forbearance), score distributions shift, requiring drift monitoring and periodic threshold recalibration."
    )
