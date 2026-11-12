import sys
from pathlib import Path

# Streamlit executes this file as a script, which otherwise places only
# ``dashboard/`` on sys.path. Add the repository root so absolute
# ``dashboard.*`` imports work for both local and Docker launches.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dashboard.client import get, safe
from dashboard.tabs import loan_lookup, optimizer, fairness, drift, model_info

st.set_page_config(
    page_title="Credit Risk Scoring System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header & Branding
st.sidebar.title("🏦 Credit Risk Scoring")
st.sidebar.caption("Freddie Mac Single-Family Loan-Level System")

# Fetch metadata and system health
meta, is_stale = safe(get, "/meta")

if not meta:
    st.error("Backend API is currently offline. Please launch the API server (`python -m api.app` or `docker compose up`).")
    if st.button("Retry Connection"):
        st.rerun()
    st.stop()

# Provenance Display (UI-02)
model_ver = meta.get("model_version", "v1.0")
rel = meta.get("data_release", "48")
st.sidebar.info(f"**Model:** `{model_ver}`\n\n**Data Release:** `{rel}`")

# Shared Interactive Controls (UI-02)
st.sidebar.subheader("Policy & Cost Parameters")

# Split selector
splits = meta.get("splits", ["valid", "oot", "stress"])
selected_split = st.sidebar.selectbox("Evaluation Split", options=splits, index=0, key="shared_split")

# Cost sliders (FP and FN)
default_fp = float(meta.get("default_fp_cost_pct", 2.0))
default_fn = float(meta.get("default_fn_cost_pct", 25.0))

slider_ranges = meta.get("slider_ranges", {})
fp_min = float(slider_ranges.get("fp_cost_pct", {}).get("min", 0.5))
fp_max = float(slider_ranges.get("fp_cost_pct", {}).get("max", 10.0))
fp_step = float(slider_ranges.get("fp_cost_pct", {}).get("step", 0.1))

fn_min = float(slider_ranges.get("fn_cost_pct", {}).get("min", 5.0))
fn_max = float(slider_ranges.get("fn_cost_pct", {}).get("max", 50.0))
fn_step = float(slider_ranges.get("fn_cost_pct", {}).get("step", 0.5))

fp_cost = st.sidebar.slider(
    "False Positive Cost (% of UPB)",
    min_value=fp_min,
    max_value=fp_max,
    value=default_fp,
    step=fp_step,
    help="Net interest margin foregone by rejecting a non-defaulting applicant",
    key="shared_fp_cost"
)

fn_cost = st.sidebar.slider(
    "False Negative Cost (% of UPB)",
    min_value=fn_min,
    max_value=fn_max,
    value=default_fn,
    step=fn_step,
    help="Net credit loss severity incurred from approving a defaulting borrower",
    key="shared_fn_cost"
)

# Threshold mode
mode = st.sidebar.radio("Threshold Mode", ["Optimal", "Manual"], index=0, key="shared_mode")

default_t = float(meta.get("default_threshold", 0.08))
manual_t = st.sidebar.number_input(
    "Manual Operating Threshold",
    min_value=0.001,
    max_value=0.999,
    value=default_t,
    step=0.005,
    disabled=(mode == "Optimal"),
    help="Explicit probability threshold used when manual mode is selected",
    key="shared_manual_threshold"
)

# Pack shared parameters for tabs (UI-02)
params = {
    "split": selected_split,
    "fp_cost_pct": fp_cost,
    "fn_cost_pct": fn_cost,
    "mode": mode.lower(),
    "threshold": manual_t if mode == "Manual" else None
}

# Main Application Tabs (UI-01)
tab_names = [
    "Loan Lookup",
    "Business Optimizer",
    "Fairness & Bias Audit",
    "Drift Monitor",
    "Model & Data"
]
tabs = st.tabs(tab_names)

with tabs[0]:
    loan_lookup.render(params, meta)

with tabs[1]:
    optimizer.render(params, meta)

with tabs[2]:
    fairness.render(params, meta)

with tabs[3]:
    drift.render(params, meta)

with tabs[4]:
    model_info.render(params, meta)
