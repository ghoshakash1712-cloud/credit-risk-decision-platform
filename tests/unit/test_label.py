import pandas as pd
import numpy as np
from src.common.config import load_config

def label_loans(orig: pd.DataFrame, perf: pd.DataFrame, cfg: dict):
    W = cfg["data"]["label_window_months"]
    p = perf.merge(orig[["loan_id", "first_payment_date"]], on="loan_id", how="inner")

    p_yr = p["period"].astype(str).str[:4].astype(int)
    p_mo = p["period"].astype(str).str[4:6].astype(int)
    fp_yr = p["first_payment_date"].astype(str).str[:4].astype(int)
    fp_mo = p["first_payment_date"].astype(str).str[4:6].astype(int)
    p["m"] = (p_yr - fp_yr) * 12 + (p_mo - fp_mo)
    in_window = p["m"].between(0, W)

    status = p["delinq_status"].astype(str).str.strip()
    status_num = pd.to_numeric(status, errors="coerce")
    is_90_plus = (status_num >= cfg["data"]["default_delinquency_status"]) | (status == "RA")

    zb = pd.to_numeric(p["zero_bal_code"], errors="coerce")
    is_credit_event = zb.isin(cfg["data"]["credit_event_zero_balance_codes"])

    default_series = (in_window & (is_90_plus | is_credit_event)).groupby(p["loan_id"]).any()

    has_defect = (zb.isin(cfg["data"]["exclude_zero_balance_codes"])).groupby(p["loan_id"]).any()
    no_perf = ~orig["loan_id"].isin(p["loan_id"])
    if "harp_indicator" in orig.columns:
        is_harp = orig.set_index("loan_id")["harp_indicator"].astype(str).str.strip().str.upper() == "Y"
    else:
        is_harp = pd.Series(False, index=orig["loan_id"])

    labels_df = pd.DataFrame({"loan_id": orig["loan_id"]})
    labels_df["default_24m"] = labels_df["loan_id"].map(default_series).fillna(False).astype(int)

    valid_mask = (~labels_df["loan_id"].map(has_defect).fillna(False)) & \
                 (~labels_df["loan_id"].isin(orig.loc[no_perf, "loan_id"])) & \
                 (~labels_df["loan_id"].map(is_harp).fillna(False))

    labels_df = labels_df[valid_mask].copy()

    reasons = {
        "total_orig": len(orig),
        "excluded_defect_96": int(has_defect.sum()),
        "excluded_no_perf": int(no_perf.sum()),
        "excluded_harp": int(is_harp.sum()),
        "final_cohort": len(labels_df)
    }
    return labels_df, reasons

def test_label_construction_and_exclusions():
    """TS-03: Tests synthetic performance records for labeling rules and exclusions."""
    cfg = load_config()

    orig = pd.DataFrame({
        "loan_id": ["A", "B", "C", "D", "E"],
        "first_payment_date": ["201801", "201801", "201801", "201801", "201801"],
        "harp_indicator": ["N", "N", "N", "N", "N"]
    })

    # Loan A: reaches status 3 at m=20 -> 1
    # Loan B: reaches status 3 at m=30 (outside window) -> 0
    # Loan C: modified at m=10 (loan_age resets), status 3 at m=22 -> 1
    # Loan D: zero_bal_code 96 -> excluded
    # Loan E: zero_bal_code 01 at m=12 (voluntary payoff) -> 0
    perf = pd.DataFrame({
        "loan_id": ["A", "B", "C", "D", "E"],
        "period": ["201909", "202007", "201911", "201906", "201901"],
        "delinq_status": ["03", "03", "03", "00", "00"],
        "loan_age": [20, 30, 2, 17, 12],
        "zero_bal_code": [np.nan, np.nan, np.nan, "96", "01"],
        "actual_loss": [0.0, 0.0, 0.0, 0.0, 0.0]
    })

    labels, reasons = label_loans(orig, perf, cfg)

    assert "D" not in labels["loan_id"].values, "Loan D with zero_bal_code 96 must be excluded"
    res = dict(zip(labels["loan_id"], labels["default_24m"]))
    assert res["A"] == 1, "Loan A reached 90+ days within window -> 1"
    assert res["B"] == 0, "Loan B reached 90+ days outside window -> 0"
    assert res["C"] == 1, "Loan C reached 90+ days at m=22 (ignoring loan_age reset) -> 1"
    assert res["E"] == 0, "Loan E voluntary payoff code 01 -> 0"
    assert reasons["excluded_defect_96"] == 1
