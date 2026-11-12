from math import sqrt
import numpy as np
import pandas as pd

ATTRIBUTES = ["first_time_homebuyer", "borrower_count", "occupancy", "loan_purpose"]
UNRATED = {"Unknown", "Other or unknown"}

LABELS = {
    "first_time_homebuyer": {"Y": "Yes", "N": "No"},
    "occupancy": {"P": "Primary residence", "O": "Primary residence", "I": "Investment property", "S": "Second home"},
    "loan_purpose": {"P": "Purchase", "C": "Refinance cash-out", "N": "Refinance no cash-out", "R": "Other or unknown"}
}

def add_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. First-time homebuyer
    if "first_time_homebuyer" in df.columns:
        fthb = df["first_time_homebuyer"].astype(str).str.strip().str.upper()
        df["grp_first_time_homebuyer"] = fthb.map(LABELS["first_time_homebuyer"]).fillna("Unknown")
    else:
        df["grp_first_time_homebuyer"] = "Unknown"

    # 2. Borrower count
    if "n_borrowers" in df.columns:
        nb = pd.to_numeric(df["n_borrowers"], errors="coerce")
        df["grp_borrower_count"] = np.where(nb.isna(), "Unknown",
                                            np.where(nb == 1, "1 borrower", "2 or more"))
    else:
        df["grp_borrower_count"] = "Unknown"

    # 3. Occupancy
    if "occupancy" in df.columns:
        occ = df["occupancy"].astype(str).str.strip().str.upper()
        df["grp_occupancy"] = occ.map(LABELS["occupancy"]).fillna("Unknown")
    else:
        df["grp_occupancy"] = "Unknown"

    # 4. Loan purpose
    if "loan_purpose" in df.columns:
        lp = df["loan_purpose"].astype(str).str.strip().str.upper()
        df["grp_loan_purpose"] = lp.map(LABELS["loan_purpose"]).fillna("Other or unknown")
    else:
        df["grp_loan_purpose"] = "Other or unknown"

    return df

def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """FA-05: 95% Wilson score interval for proportion k / n."""
    if n == 0 or np.isnan(n):
        return (None, None, None)
    p = float(k / n)
    denom = 1.0 + (z * z) / n
    centre = (p + (z * z) / (2.0 * n)) / denom
    half = (z * sqrt(p * (1.0 - p) / n + (z * z) / (4.0 * n * n))) / denom
    ci_low = max(0.0, float(centre - half))
    ci_high = min(1.0, float(centre + half))
    return (float(p), ci_low, ci_high)

def encode_groups(df: pd.DataFrame) -> dict:
    """Precompute integer codes per attribute for fast vectorized numpy binning."""
    result = {}
    for attr in ATTRIBUTES:
        col = f"grp_{attr}"
        if col not in df.columns:
            df = add_group_columns(df)
        series = df[col].astype(str)
        labels = sorted(series.unique())
        label_to_idx = {lab: i for i, lab in enumerate(labels)}
        codes = series.map(label_to_idx).values.astype(int)
        result[attr] = (codes, labels)
    return result

def audit_attribute(prob: np.ndarray, y: np.ndarray, codes: np.ndarray, labels: list, t: float, cfg_fairness: dict) -> dict:
    """FA-05 to FA-08, FA-14 to FA-16: Vectorized NumPy computation of fairness metrics."""
    prob = np.asarray(prob, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    reject = prob >= t
    G = len(labels)

    min_loans = cfg_fairness.get("size_rule", {}).get("min_loans", 500)
    min_defaults = cfg_fairness.get("size_rule", {}).get("min_defaults", 30)
    flags_cfg = cfg_fairness.get("flags", {})
    di_min = flags_cfg.get("di_ratio_min", 0.80)
    eod_max = flags_cfg.get("eod_max", 0.10)
    cal_range = flags_cfg.get("calibration_ratio_range", [0.80, 1.25])

    n = np.bincount(codes, minlength=G)
    d = np.bincount(codes, weights=y, minlength=G).astype(int)
    r = np.bincount(codes, weights=reject, minlength=G).astype(int)
    tp = np.bincount(codes, weights=reject & (y == 1), minlength=G).astype(int)
    fp = np.bincount(codes, weights=reject & (y == 0), minlength=G).astype(int)
    neg = n - d
    sum_p = np.bincount(codes, weights=prob, minlength=G)

    rows = []
    for g in range(G):
        grp_n = int(n[g])
        grp_d = int(d[g])
        grp_r = int(r[g])
        grp_app = grp_n - grp_r
        grp_tp = int(tp[g])
        grp_fp = int(fp[g])
        grp_neg = int(neg[g])

        mean_p = float(sum_p[g] / grp_n) if grp_n > 0 else 0.0

        p_obs, obs_lo, obs_hi = wilson(grp_d, grp_n)
        p_rej, rej_lo, rej_hi = wilson(grp_r, grp_n)
        p_app, app_lo, app_hi = wilson(grp_app, grp_n)

        # FA-14: Undefined rates
        if grp_d > 0:
            p_tpr, tpr_lo, tpr_hi = wilson(grp_tp, grp_d)
        else:
            p_tpr, tpr_lo, tpr_hi = (None, None, None)

        if grp_neg > 0:
            p_fpr, fpr_lo, fpr_hi = wilson(grp_fp, grp_neg)
        else:
            p_fpr, fpr_lo, fpr_hi = (None, None, None)

        is_rated = bool(grp_n >= min_loans and grp_d >= min_defaults and labels[g] not in UNRATED)

        rows.append({
            "group": labels[g],
            "loans": grp_n,
            "defaults": grp_d,
            "observed_default_rate": {"value": p_obs, "ci_lower": obs_lo, "ci_upper": obs_hi},
            "mean_predicted_probability": mean_p,
            "rejection_rate": {"value": p_rej, "ci_lower": rej_lo, "ci_upper": rej_hi},
            "approval_rate": {"value": p_app, "ci_lower": app_lo, "ci_upper": app_hi},
            "default_capture_rate": {"value": p_tpr, "ci_lower": tpr_lo, "ci_upper": tpr_hi},
            "false_rejection_rate": {"value": p_fpr, "ci_lower": fpr_lo, "ci_upper": fpr_hi},
            "rated": is_rated
        })

    rated_rows = [r for r in rows if r["rated"]]

    # FA-15: Too few groups
    if len(rated_rows) < 2:
        return {
            "status": "not_enough_rated_groups",
            "groups": rows,
            "summary": None,
            "flags": []
        }

    # Summary measures
    app_vals = [r["approval_rate"]["value"] for r in rated_rows]
    rej_vals = [r["rejection_rate"]["value"] for r in rated_rows]

    # FA-16: Reject-nothing threshold
    if max(rej_vals) == 0.0:
        di = 1.0
        dpd = 0.0
    else:
        max_app = max(app_vals)
        di = float(min(app_vals) / max_app) if max_app > 0 else 1.0
        dpd = float(max(rej_vals) - min(rej_vals))

    tpr_vals = [r["default_capture_rate"]["value"] for r in rated_rows if r["default_capture_rate"]["value"] is not None]
    fpr_vals = [r["false_rejection_rate"]["value"] for r in rated_rows if r["false_rejection_rate"]["value"] is not None]

    gap_tpr = (max(tpr_vals) - min(tpr_vals)) if len(tpr_vals) >= 2 else 0.0
    gap_fpr = (max(fpr_vals) - min(fpr_vals)) if len(fpr_vals) >= 2 else 0.0
    eod = float(max(gap_tpr, gap_fpr))

    cal_map = {}
    for r in rated_rows:
        if r["mean_predicted_probability"] > 0 and r["observed_default_rate"]["value"] is not None:
            cal_map[r["group"]] = float(r["observed_default_rate"]["value"] / r["mean_predicted_probability"])
        else:
            cal_map[r["group"]] = None

    flags = []

    # Disparate impact flag
    if di < di_min:
        min_g = rated_rows[int(np.argmin(app_vals))]["group"]
        max_g = rated_rows[int(np.argmax(app_vals))]["group"]
        flags.append({
            "type": "disparate_impact",
            "metric": round(di, 4),
            "threshold": di_min,
            "reason": f"Disparate impact ratio {di:.2f} is below 0.80 ({min_g} vs {max_g})"
        })

    # Equalized odds flag
    if eod > eod_max:
        if gap_tpr >= gap_fpr and len(tpr_vals) >= 2:
            min_g = rated_rows[int(np.argmin([r['default_capture_rate']['value'] for r in rated_rows]))]["group"]
            max_g = rated_rows[int(np.argmax([r['default_capture_rate']['value'] for r in rated_rows]))]["group"]
            detail = f"default capture rate gap between {max_g} and {min_g}"
        else:
            min_g = rated_rows[int(np.argmin([r['false_rejection_rate']['value'] for r in rated_rows]))]["group"]
            max_g = rated_rows[int(np.argmax([r['false_rejection_rate']['value'] for r in rated_rows]))]["group"]
            detail = f"false rejection rate gap between {max_g} and {min_g}"
        flags.append({
            "type": "equalized_odds",
            "metric": round(eod, 4),
            "threshold": eod_max,
            "reason": f"Equalized odds difference {eod:.2f} exceeds 0.10 ({detail})"
        })

    # Calibration ratio flag
    for grp, ratio in cal_map.items():
        if ratio is not None:
            grp_defaults = next((r["defaults"] for r in rated_rows if r["group"] == grp), 0)
            if grp_defaults >= min_defaults:
                if ratio < cal_range[0] or ratio > cal_range[1]:
                    flags.append({
                        "type": "calibration_ratio",
                        "metric": round(ratio, 4),
                        "group": grp,
                        "threshold": cal_range,
                        "reason": f"Calibration ratio {ratio:.2f} for {grp} is outside [{cal_range[0]}, {cal_range[1]}]"
                    })

    return {
        "status": "ok",
        "groups": rows,
        "summary": {
            "disparate_impact_ratio": round(di, 4),
            "demographic_parity_difference": round(dpd, 4),
            "equalized_odds_difference": round(eod, 4),
            "calibration_ratio": {k: (round(v, 4) if v is not None else None) for k, v in cal_map.items()}
        },
        "flags": flags
    }
