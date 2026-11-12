import streamlit as st
import pandas as pd
import altair as alt
from dashboard.client import get, safe

def render(params: dict, meta: dict):
    st.header("Fairness & Demographic Bias Audit")
    st.markdown("Audit approval decisions and model error rates across demographic and loan attribute groups using Fairlearn metrics and Wilson confidence intervals.")

    configured_attrs = meta.get("fairness", {}).get("attributes", [])
    attr_keys = [a["key"] if isinstance(a, dict) else str(a) for a in configured_attrs]
    attr_labels = {a["key"]: a.get("label", a["key"]) for a in configured_attrs if isinstance(a, dict)}

    sel_attr = st.selectbox(
        "Select Sensitive / Protected Attribute Proxy",
        options=attr_keys,
        format_func=lambda k: attr_labels.get(k, k),
        key="fairness_attr_select"
    )

    fair_res, _ = safe(get, "/fairness", **params, attribute=sel_attr)
    if not fair_res or not fair_res.get("attributes"):
        return

    audit = fair_res["attributes"][sel_attr]
    thresh_used = fair_res["threshold_used"]
    st.caption(f"Evaluated on split **{fair_res['split'].upper()}** at operating threshold **{thresh_used*100:.2f}%**")

    # Status check
    if audit.get("status") == "not_enough_rated_groups":
        st.warning("⚠️ Insufficient rated groups meeting minimum sample size criteria (500 loans, 30 defaults) to evaluate disparate impact.")
        return

    # Flags Panel (UI-05)
    flags = audit.get("flags", [])
    if flags:
        st.error(f"🚩 **{len(flags)} Fairness Flag(s) Triggered**")
        for f in flags:
            st.markdown(f"- **{f['type'].upper()}**: {f['message']}")
    else:
        st.success("✅ **No Fairness Flags Triggered**: Demographic parity and predictive equality ratios are within tolerance boundaries.")

    # FA-02 Caution Note (mandatory display next to flags)
    st.info(
        "⚖️ **Regulatory & Contextual Note (FA-02):** "
        + audit.get("notes", "Disparities reflect underlying economic differences. A flag is a prompt to investigate, not proof of bias.")
    )

    st.divider()

    # Groups Data Table
    st.subheader("Subgroup Error Rates & Parity Metrics")
    groups_data = audit.get("groups", [])
    table_rows = []
    
    for g in groups_data:
        dcr_ci = f"[{g['default_capture_rate_ci_lower']*100:.1f}%, {g['default_capture_rate_ci_upper']*100:.1f}%]"
        frr_ci = f"[{g['false_rejection_rate_ci_lower']*100:.1f}%, {g['false_rejection_rate_ci_upper']*100:.1f}%]"
        table_rows.append({
            "Group": g["group"],
            "Loans": g["n_loans"],
            "Defaults": g["n_defaults"],
            "Default Rate": f"{g['default_rate']*100:.2f}%",
            "Rejection Rate": f"{g['rejection_rate']*100:.1f}%",
            "Default Capture Rate": f"{g['default_capture_rate']*100:.1f}%",
            "Default Capture 95% CI": dcr_ci,
            "False Rejection Rate": f"{g['false_rejection_rate']*100:.1f}%",
            "False Rejection 95% CI": frr_ci,
            "Calibration Ratio": f"{g['calibration_ratio']:.2f}" if g.get("calibration_ratio") is not None else "N/A",
            "Disparate Impact Ratio": f"{g['disparate_impact_ratio']:.2f}" if g.get("disparate_impact_ratio") is not None else "N/A",
            "Rated": "Yes" if g.get("rated", True) else "No (Too small)"
        })

    df_groups = pd.DataFrame(table_rows)
    st.dataframe(df_groups, use_container_width=True, hide_index=True)

    st.divider()

    # Visualizations: Charts paired with tables (UI-05, UI-12)
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.subheader("1. Rejection Rate by Subgroup")
        df_rej = pd.DataFrame([
            {"Group": g["group"], "Rejection Rate (%)": g["rejection_rate"] * 100.0, "Rated": g.get("rated", True)}
            for g in groups_data
        ])
        
        rej_chart = alt.Chart(df_rej).mark_bar().encode(
            x=alt.X("Group:N", title="Group", sort=None),
            y=alt.Y("Rejection Rate (%):Q", title="Rejection Rate (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Rated:N", scale=alt.Scale(domain=[True, False], range=["#4E79A7", "#BAB0AC"]), title="Rated Group"),
            tooltip=["Group", "Rejection Rate (%)", "Rated"]
        ).properties(height=320, title="Subgroup Rejection Rate Comparison")
        st.altair_chart(rej_chart, use_container_width=True)

    with col_c2:
        st.subheader("2. Error Rates with Wilson 95% CIs")
        whisker_data = []
        for g in groups_data:
            if g.get("rated", True):
                whisker_data.append({
                    "Group": g["group"],
                    "Metric": "Default Capture Rate (Recall)",
                    "Rate": g["default_capture_rate"] * 100.0,
                    "CI_Lower": g["default_capture_rate_ci_lower"] * 100.0,
                    "CI_Upper": g["default_capture_rate_ci_upper"] * 100.0
                })
                whisker_data.append({
                    "Group": g["group"],
                    "Metric": "False Rejection Rate (1 - Spec)",
                    "Rate": g["false_rejection_rate"] * 100.0,
                    "CI_Lower": g["false_rejection_rate_ci_lower"] * 100.0,
                    "CI_Upper": g["false_rejection_rate_ci_upper"] * 100.0
                })
        
        if whisker_data:
            df_w = pd.DataFrame(whisker_data)
            
            bars = alt.Chart(df_w).mark_point(filled=True, size=60).encode(
                x=alt.X("Group:N", title="Group"),
                y=alt.Y("Rate:Q", title="Rate (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Metric:N", title="Metric"),
                tooltip=["Group", "Metric", alt.Tooltip("Rate:Q", format=".1f"), alt.Tooltip("CI_Lower:Q", format=".1f"), alt.Tooltip("CI_Upper:Q", format=".1f")]
            )
            
            ticks = alt.Chart(df_w).mark_errorbar().encode(
                x=alt.X("Group:N"),
                y=alt.Y("CI_Lower:Q", title="Rate (%)"),
                y2="CI_Upper:Q",
                color=alt.Color("Metric:N")
            )
            
            ci_chart = (bars + ticks).properties(height=320, title="Default Capture & False Rejection Rates (with Wilson CIs)")
            st.altair_chart(ci_chart, use_container_width=True)

    # Baseline vs Current Comparison
    st.divider()
    st.subheader("Baseline vs Current Threshold Movement")
    st.caption("How group metrics shifted between the reference default threshold and current tuned threshold:")
    
    base_data = fair_res.get("baseline", {}).get(sel_attr, {})
    if base_data and "groups" in base_data:
        comp_rows = []
        base_groups = {bg["group"]: bg for bg in base_data["groups"]}
        for g in groups_data:
            g_name = g["group"]
            bg = base_groups.get(g_name, {})
            b_rej = bg.get("rejection_rate")
            c_rej = g["rejection_rate"]
            comp_rows.append({
                "Group": g_name,
                "Baseline Rejection Rate": f"{b_rej*100:.1f}%" if b_rej is not None else "N/A",
                "Current Rejection Rate": f"{c_rej*100:.1f}%",
                "Rejection Rate Change": f"{(c_rej - (b_rej or 0))*100:+.1f}%",
                "Current Capture Rate": f"{g['default_capture_rate']*100:.1f}%"
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
