import streamlit as st
import pandas as pd
import altair as alt
from dashboard.client import get, safe

def render(params: dict, meta: dict):
    st.header("Population Stability Index (PSI) Drift Monitor")
    st.markdown("Track distribution drift on output scores and input credit features relative to baseline validation distributions (DR-01 to DR-15).")

    v_col1, v_col2 = st.columns([1, 2])
    with v_col1:
        view_type = st.radio("Monitoring View", ["Real Vintages", "Simulated Shock"], horizontal=True, key="drift_view_type")

    is_simulated = (view_type == "Simulated Shock")

    if not is_simulated:
        with v_col2:
            real_window = st.selectbox("Evaluation Window", ["oot", "stress", "valid"], index=0, key="drift_real_window")
        drift_data, _ = safe(get, "/drift", source="real", window=real_window)
        window_title = real_window.upper()
    else:
        with v_col2:
            scen_col, sev_col = st.columns(2)
            with scen_col:
                scenario = st.selectbox(
                    "Simulated Scenario",
                    ["rate_shock", "debt_stress", "credit_deterioration", "leverage_squeeze"],
                    format_func=lambda s: s.replace("_", " ").title(),
                    key="drift_sim_scen"
                )
            with sev_col:
                severity = st.radio("Severity", ["mild", "moderate", "severe"], horizontal=True, key="drift_sim_sev")
        drift_data, _ = safe(get, "/drift", source="simulated", scenario=scenario, severity=severity)
        window_title = f"SIMULATED: {scenario.replace('_', ' ').upper()} ({severity.upper()})"

    if not drift_data:
        return

    st.divider()

    # Status Banner (UI-08, DR-15)
    status_str = drift_data.get("status", "unknown").capitalize()
    score_psi = drift_data.get("score_psi", 0.0)
    triggers = drift_data.get("triggers", [])
    
    triggers_text = ", ".join(triggers) if triggers else "None"
    
    # Determine DR-15 recommended action
    if score_psi >= 0.20:
        rec_action = "Retrain recommended. Evaluate macroeconomic regime shift and recalibrate credit decision threshold."
        st.error(
            f"🚨 **STATUS: RETRAIN RECOMMENDED (Significant Drift)** — Score PSI = {score_psi:.4f} (≥ 0.20)\n\n"
            f"• **Triggering Variables:** {triggers_text}\n\n"
            f"• **Recommended Action (DR-15):** {rec_action}"
        )
    elif score_psi >= 0.10:
        rec_action = "Warning. Increase monitoring cadence, investigate borrower mix shift, and inspect feature drift."
        st.warning(
            f"⚠️ **STATUS: WARNING (Moderate Drift)** — Score PSI = {score_psi:.4f} (0.10 – 0.20)\n\n"
            f"• **Triggering Variables:** {triggers_text}\n\n"
            f"• **Recommended Action (DR-15):** {rec_action}"
        )
    else:
        rec_action = "Portfolio distribution remains stable. Continue regular monitoring cadence."
        st.success(
            f"✅ **STATUS: STABLE (No Action Needed)** — Score PSI = {score_psi:.4f} (< 0.10)\n\n"
            f"• **Triggering Variables:** {triggers_text}\n\n"
            f"• **Recommended Action (DR-15):** {rec_action}"
        )

    if is_simulated:
        st.info("ℹ️ **Notice: Displaying synthetic counterfactual shift scenario.** All values and distributions represent simulated stress shocks.")

    st.divider()

    # Feature PSI Chart & Table
    st.subheader(f"Feature Stability & Drift ({window_title})")
    feat_data = drift_data.get("features", [])

    if feat_data:
        df_feats = pd.DataFrame(feat_data).sort_values("psi", ascending=False)
        
        # Color coding by band
        band_colors = alt.Scale(domain=["stable", "moderate", "significant"], range=["#59A14F", "#EDC949", "#E15759"])

        bar_chart = alt.Chart(df_feats).mark_bar().encode(
            x=alt.X("psi:Q", title="Population Stability Index (PSI)"),
            y=alt.Y("feature:N", sort="-x", title="Feature Name"),
            color=alt.Color("band:N", scale=band_colors, title="Drift Band"),
            tooltip=["feature", alt.Tooltip("psi:Q", format=".4f"), "band", "structural"]
        )

        rule10 = alt.Chart(pd.DataFrame([{"psi": 0.10}])).mark_rule(
            color="#EDC949", strokeDash=[4, 4], strokeWidth=2
        ).encode(x="psi:Q")
        rule20 = alt.Chart(pd.DataFrame([{"psi": 0.20}])).mark_rule(
            color="#E15759", strokeDash=[4, 4], strokeWidth=2
        ).encode(x="psi:Q")

        feat_chart = (bar_chart + rule10 + rule20).properties(
            title=f"Feature PSI Breakdown with 0.10 and 0.20 Warning Thresholds ({window_title})",
            height=380
        )
        st.altair_chart(feat_chart, use_container_width=True)

        st.markdown("**Feature Drift Metrics Table**")
        disp_f_df = df_feats.copy()
        disp_f_df["psi"] = disp_f_df["psi"].apply(lambda p: f"{p:.4f}")
        disp_f_df["structural"] = disp_f_df["structural"].apply(lambda s: "Yes" if s else "No")
        st.dataframe(disp_f_df, use_container_width=True, hide_index=True)

    st.divider()

    # Score Distribution Overlay
    st.subheader("Score Distribution: Baseline Reference vs Current Window")
    score_bins = drift_data.get("score_bins", [])
    if score_bins:
        overlay_rows = []
        for b in score_bins:
            overlay_rows.append({"Bin": b["bin"], "Distribution": "Reference (Validation Baseline)", "Share (%)": b["reference_share"] * 100.0})
            overlay_rows.append({"Bin": b["bin"], "Distribution": f"Current Window ({window_title})", "Share (%)": b["current_share"] * 100.0})
        
        df_overlay = pd.DataFrame(overlay_rows)
        overlay_chart = alt.Chart(df_overlay).mark_bar(opacity=0.75).encode(
            x=alt.X("Bin:O", title="Score Decile Bin (Validation-Fitted)"),
            y=alt.Y("Share (%):Q", title="Population Share (%)", scale=alt.Scale(domain=[0, max(df_overlay["Share (%)"].max() * 1.15, 25)])),
            color=alt.Color("Distribution:N", scale=alt.Scale(range=["#4E79A7", "#F28E2B"]), title="Population"),
            xOffset="Distribution:N",
            tooltip=["Bin", "Distribution", alt.Tooltip("Share (%):Q", format=".2f")]
        ).properties(
            title="Score Distribution Shift Overlay",
            height=320
        )
        st.altair_chart(overlay_chart, use_container_width=True)

    # Quarterly PSI Trend
    st.divider()
    st.subheader("Quarterly Historical Drift Trend (2018Q1 – 2021Q4)")
    q_data, _ = safe(get, "/drift", source="real", window="quarterly")
    if q_data and q_data.get("series"):
        df_q = pd.DataFrame(q_data["series"])
        
        q_line = alt.Chart(df_q).mark_line(point=True, color="#4E79A7", strokeWidth=2).encode(
            x=alt.X("quarter:N", title="Origination Quarter"),
            y=alt.Y("score_psi:Q", title="Score PSI", scale=alt.Scale(domain=[0, max(df_q["score_psi"].max() * 1.2, 0.25)])),
            tooltip=["quarter", alt.Tooltip("score_psi:Q", format=".4f"), "status", "n_loans"]
        )
        q_rule10 = alt.Chart(pd.DataFrame([{"y": 0.10}])).mark_rule(color="#EDC949", strokeDash=[4, 4]).encode(y="y:Q")
        q_rule20 = alt.Chart(pd.DataFrame([{"y": 0.20}])).mark_rule(color="#E15759", strokeDash=[4, 4]).encode(y="y:Q")
        
        q_chart = (q_line + q_rule10 + q_rule20).properties(
            title="Quarterly Score Drift Relative to Validation Baseline",
            height=300
        )
        st.altair_chart(q_chart, use_container_width=True)
