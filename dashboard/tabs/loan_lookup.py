import streamlit as st
import pandas as pd
import altair as alt
from dashboard.client import get, safe

def render(params: dict, meta: dict):
    st.header("Loan Lookup & Local Explanation")
    st.markdown("Search for loans in the active evaluation split, inspect risk predictions, and review local TreeSHAP explanations.")

    split = params.get("split", "valid")
    col_search, col_pick = st.columns([1, 2])
    
    with col_search:
        search_query = st.text_input("Search Loan ID Prefix", value="", placeholder="e.g. F118Q1...", key="loan_search_query")
    
    # Query /loans
    loans_resp, _ = safe(get, "/loans", split=split, limit=50, search=search_query if search_query else None)
    if not loans_resp or not loans_resp.get("items"):
        st.info("No loans found matching the criteria in split: " + split)
        return

    items = loans_resp["items"]
    loan_options = [it["loan_id"] for it in items]
    
    with col_pick:
        selected_loan_id = st.selectbox(
            f"Select Loan ({loans_resp['total']} total found in {split})",
            options=loan_options,
            key="selected_loan_id"
        )

    if not selected_loan_id:
        return

    # Fetch loan details, prediction, and explanation
    threshold_arg = params.get("threshold")
    detail, _ = safe(get, f"/loans/{selected_loan_id}")
    pred, _ = safe(get, f"/predict/{selected_loan_id}", threshold=threshold_arg)
    exp, _ = safe(get, f"/explain/{selected_loan_id}")

    if not detail or not pred or not exp:
        return

    st.divider()

    # Top summary metrics
    m1, m2, m3, m4 = st.columns(4)
    prob_pct = pred["probability"] * 100.0
    thresh_pct = pred["threshold"] * 100.0
    is_rejected = (pred["decision"] == "reject")
    
    with m1:
        st.metric("Default Probability", f"{prob_pct:.1f}%")
    with m2:
        st.metric(f"Decision ({pred['threshold_source'].title()} Thresh: {thresh_pct:.1f}%)", 
                  "REJECT" if is_rejected else "APPROVE")
    with m3:
        actual_str = "Defaulted (90+ DPD / Zero Bal)" if detail["actual_outcome"] == 1 else "Non-Default (Current / Paid)"
        st.metric("Historical 24m Outcome", actual_str)
    with m4:
        st.metric("Original Balance (UPB)", f"${detail['original_upb']:,.0f}")

    if is_rejected:
        st.error(f"Decision: **REJECT** — Model default probability ({prob_pct:.1f}%) meets or exceeds the decision threshold ({thresh_pct:.1f}%).")
    else:
        st.success(f"Decision: **APPROVE** — Model default probability ({prob_pct:.1f}%) is below the decision threshold ({thresh_pct:.1f}%).")

    # Feature details & TreeSHAP contributions
    tab_exp, tab_feats = st.tabs(["Local Explanation (TreeSHAP)", "Loan Attributes"])

    with tab_exp:
        st.subheader("Feature Risk Contributions")
        st.caption(f"Base Value (prior log-odds): {exp['base_value']:.4f} | Reconciled Probability: {exp['probability']*100:.1f}%")

        contrib_list = list(exp["contributions"])
        contrib_list.append({
            "feature": "Other Features (Combined)",
            "value": None,
            "contribution_log_odds": exp["other_contribution_log_odds"]
        })

        df_contrib = pd.DataFrame(contrib_list)
        df_contrib["direction"] = df_contrib["contribution_log_odds"].apply(
            lambda v: "Increases Risk" if v > 0 else "Decreases Risk"
        )

        chart = alt.Chart(df_contrib).mark_bar().encode(
            x=alt.X("contribution_log_odds:Q", title="Contribution (log-odds)"),
            y=alt.Y("feature:N", sort=None, title="Feature"),
            color=alt.Color("direction:N",
                            scale=alt.Scale(domain=["Increases Risk", "Decreases Risk"], range=["#E15759", "#59A14F"]),
                            title="Impact"),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("value", title="Feature Value"),
                alt.Tooltip("contribution_log_odds:Q", format="+.4f", title="Log-Odds Contribution"),
                alt.Tooltip("direction:N", title="Impact")
            ]
        ).properties(
            title="TreeSHAP Local Feature Contributions (Top 10 + Reconciled Remainder)",
            height=380
        )
        st.altair_chart(chart, use_container_width=True)

        st.markdown("**Contribution Data Table**")
        disp_df = df_contrib.copy()
        disp_df["contribution_log_odds"] = disp_df["contribution_log_odds"].apply(lambda x: f"{x:+.4f}")
        st.dataframe(disp_df, use_container_width=True, hide_index=True)

    with tab_feats:
        st.subheader("Origination Characteristics")
        feat_dict = detail.get("features", {})
        f_df = pd.DataFrame([{"Feature": k, "Value": ("N/A" if v is None else v)} for k, v in sorted(feat_dict.items())])
        st.dataframe(f_df, use_container_width=True, height=400, hide_index=True)
