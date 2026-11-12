import streamlit as st
import pandas as pd
import altair as alt
from dashboard.client import get, safe

def render(params: dict, meta: dict):
    st.header("Business Threshold Optimizer")
    st.markdown("Tune business cost sliders in the sidebar to simulate balance-weighted net financial benefit across candidate approval thresholds.")

    opt_res, _ = safe(get, "/optimize", **params)
    if not opt_res:
        return

    # Check for reject_none or calibration warning
    if opt_res.get("reject_none"):
        st.info("ℹ️ **Rejecting nothing is optimal at these costs.** Model recommendations approve 100% of applicants.")

    if opt_res.get("calibration_warning"):
        st.warning(
            f"⚠️ **Calibration Warning:** Optimal threshold ({opt_res['threshold']:.4f}) deviates significantly "
            f"from the analytic threshold ({opt_res['analytic_threshold']:.4f}). Probabilities in this tail region may be miscalibrated."
        )

    # Headline metric cards (UI-04)
    m = opt_res["metrics"]
    t_val = opt_res["threshold"]
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        st.metric("Effective Threshold", f"{t_val*100:.2f}%", 
                  help="Operating probability threshold used for approval/rejection")
    with c2:
        st.metric("Net Financial Benefit", f"${m['net_benefit']:,.0f}", 
                  help="Balance-weighted net benefit over approving everyone")
    with c3:
        st.metric("Benefit vs Approve-All", f"${m['profit'] - m['approve_all_profit']:,.0f}",
                  help="Gain in portfolio profit compared to zero-rejection policy")
    with c4:
        st.metric("Default Capture Rate (Recall)", f"{m['recall']*100:.1f}%",
                  help="Percentage of defaulting loans successfully rejected")
    with c5:
        st.metric("Portfolio Rejection Rate", f"{m['rejection_rate']*100:.1f}%",
                  help="Percentage of total origination balance rejected")

    st.divider()

    # Net Benefit Curve with Altair
    st.subheader("Net Financial Benefit vs Candidate Decision Threshold")
    curve_data = pd.DataFrame(opt_res.get("curve", []))

    if not curve_data.empty:
        # Base line chart
        base_line = alt.Chart(curve_data).mark_line(color="#4E79A7", strokeWidth=2.5).encode(
            x=alt.X("threshold:Q", title="Decision Threshold", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("net_benefit:Q", title="Net Benefit (USD)", axis=alt.Axis(format="$,.0f")),
            tooltip=[
                alt.Tooltip("threshold:Q", format=".4f", title="Threshold"),
                alt.Tooltip("net_benefit:Q", format="$,.0f", title="Net Benefit")
            ]
        )

        # Optimum / Selected point
        point_df = pd.DataFrame([{"threshold": t_val, "net_benefit": m["net_benefit"]}])
        opt_point = alt.Chart(point_df).mark_point(
            color="#E15759", size=150, filled=True
        ).encode(
            x="threshold:Q",
            y="net_benefit:Q",
            tooltip=[
                alt.Tooltip("threshold:Q", format=".4f", title="Selected Threshold"),
                alt.Tooltip("net_benefit:Q", format="$,.0f", title="Selected Net Benefit")
            ]
        )

        # Rule line at selected threshold
        rule = alt.Chart(point_df).mark_rule(color="#E15759", strokeDash=[4, 4]).encode(x="threshold:Q")

        chart = (base_line + opt_point + rule).properties(
            title="Portfolio Net Benefit Curve (Evaluated on " + opt_res["split"].upper() + ")",
            height=380
        )
        st.altair_chart(chart, use_container_width=True)

    # Three-Split Comparison Table (OP-11)
    st.subheader("Performance Across Temporal & Stress Splits")
    st.caption("Comparison of policy performance applied consistently at the selected threshold:")

    splits_rows = []
    for s_entry in opt_res.get("splits", []):
        s_name = s_entry["split"]
        sm = s_entry["metrics"]
        splits_rows.append({
            "Split": s_name.upper(),
            "Threshold": f"{t_val*100:.2f}%",
            "Rejection Rate": f"{sm['rejection_rate']*100:.1f}%",
            "Recall (Default Capture)": f"{sm['recall']*100:.1f}%",
            "Precision": f"{sm['precision']*100:.1f}%",
            "Approved Default Rate": f"{sm['approved_default_rate']*100:.2f}%",
            "Net Benefit": f"${sm['net_benefit']:,.0f}",
            "Net Benefit / Loan": f"${sm['net_benefit_per_loan']:,.2f}",
            "Portfolio Profit": f"${sm['profit']:,.0f}"
        })

    df_splits = pd.DataFrame(splits_rows)
    st.dataframe(df_splits, use_container_width=True, hide_index=True)

    # Visible Caveats (OP-03, UI-04)
    st.markdown("---")
    st.markdown("### 📌 Optimization Assumptions & Model Limitations (OP-03)")
    st.caption(
        "• **Static Portfolio Assumption:** The what-if optimizer evaluates historical balance outcomes assuming borrower application volume "
        "and credit demand remain inelastic to underwriting criteria. It does not model borrower adverse selection or competitive market dynamics.\n\n"
        "• **Balance-Weighted Cost Model:** Loss from false approvals is modeled as `Original UPB × Loss Severity (25%)`. "
        "Loss from false rejections is modeled as `Original UPB × Net Interest Margin Lost (2%)`.\n\n"
        "• **Fixed Vintage Calibration:** Thresholds tuned on historical validation (2018) are subjected to out-of-time (2019) and "
        "macroeconomic stress (2020–2021) to verify policy robustness under regime shifts."
    )
