import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, roc_curve
from src.common.config import load_config, project_path
from src.common.features import feature_matrix, apply_categories
from src.common.groups import add_group_columns
from src.common import cost
from src.common.psi import build_spec

def compute_decile_table(prob: np.ndarray, y: np.ndarray) -> list:
    df = pd.DataFrame({"prob": prob, "y": y})
    try:
        df["decile"] = pd.qcut(df["prob"], 10, labels=False, duplicates="drop")
    except Exception:
        df["decile"] = 0
    grouped = df.groupby("decile")
    rows = []
    for d, g in grouped:
        n = len(g)
        defs = int(g["y"].sum())
        rate = float(defs / n) if n > 0 else 0.0
        mean_p = float(g["prob"].mean())
        rows.append({
            "decile": int(d) + 1,
            "loans": n,
            "defaults": defs,
            "observed_default_rate": round(rate, 4),
            "mean_predicted_probability": round(mean_p, 4)
        })
    return rows

def compute_calibration_table(prob: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    b_idx = np.digitize(prob, bins, right=True)
    rows = []
    for b in range(1, n_bins + 1):
        mask = (b_idx == b)
        n = int(np.sum(mask))
        if n > 0:
            defs = int(np.sum(y[mask]))
            rate = float(defs / n)
            mean_p = float(np.mean(prob[mask]))
        else:
            defs = 0
            rate = 0.0
            mean_p = float((bins[b-1] + bins[b]) / 2.0)
        rows.append({
            "bin": b,
            "bin_lower": round(float(bins[b-1]), 2),
            "bin_upper": round(float(bins[b]), 2),
            "loans": n,
            "defaults": defs,
            "observed_default_rate": round(rate, 4),
            "mean_predicted_probability": round(mean_p, 4)
        })
    return rows

def main():
    cfg = load_config()
    processed = project_path(cfg, "processed")
    artifacts = project_path(cfg, "artifacts")
    reports = project_path(cfg, "reports")
    reports.mkdir(parents=True, exist_ok=True)

    with open(artifacts / "categories.json", "r", encoding="utf-8") as f:
        categories = json.load(f)

    model_path = artifacts / "model.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    booster = lgb.Booster(model_file=str(model_path))

    print("Scoring holdout splits (valid, oot, stress)...")
    parts = []
    for split_name in ["valid", "oot", "stress"]:
        split_file = processed / f"{split_name}.parquet"
        df = apply_categories(pd.read_parquet(split_file), categories)
        X = feature_matrix(df, cfg)
        df["probability"] = booster.predict(X)
        df = add_group_columns(df)
        parts.append(df)

    holdout = pd.concat(parts, ignore_index=True)

    # Core columns to include in holdout table
    id_cols = ["loan_id", "split", "vintage", "vintage_quarter"]
    label_cols = ["probability", "orig_upb", "default_24m"]
    group_cols = ["grp_first_time_homebuyer", "grp_borrower_count", "grp_occupancy", "grp_loan_purpose"]
    feat_cols = cfg["features"]["numeric"] + cfg["features"]["categorical"]

    keep_cols = []
    for c in id_cols + label_cols + group_cols + feat_cols:
        if c in holdout.columns and c not in keep_cols:
            keep_cols.append(c)

    holdout_out = artifacts / "holdout_predictions.parquet"
    holdout[keep_cols].to_parquet(holdout_out, index=False)
    print(f"Holdout predictions written to {holdout_out} with {len(holdout)} rows")

    # MD-07, 4.6: Default Decision Threshold
    loss_sev_path = artifacts / "loss_severity.json"
    with open(loss_sev_path, "r", encoding="utf-8") as f:
        sev_data = json.load(f)
    sev_mean = sev_data.get("mean", 0.25)

    fn_min = cfg["cost"]["slider"]["fn"]["min"]
    fn_max = cfg["cost"]["slider"]["fn"]["max"]
    L_pct = float(np.clip(round(sev_mean * 100.0, 1), fn_min, fn_max))
    g_pct = float(cfg["cost"]["default_fp_cost_pct"])

    valid_df = holdout[holdout["split"] == "valid"]
    prep_val = cost.prepare(valid_df["probability"].values, valid_df["default_24m"].values, valid_df["orig_upb"].values)
    opt_val = cost.optimize(prep_val, g_pct / 100.0, L_pct / 100.0)
    default_threshold = round(float(opt_val.threshold), 4)

    print(f"Default threshold computed: {default_threshold} (g={g_pct}%, L={L_pct}%)")

    # Update metadata
    meta_path = artifacts / "metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta.update({
        "default_threshold": default_threshold,
        "default_fp_cost_pct": g_pct,
        "default_fn_cost_pct": L_pct
    })
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # MD-05, MD-06: Model Evaluation Report
    report = {}
    for split_name in ["valid", "oot", "stress"]:
        s = holdout[holdout["split"] == split_name]
        p = s["probability"].values
        y = s["default_24m"].values

        auc = float(roc_auc_score(y, p))
        gini = 2.0 * auc - 1.0
        fpr, tpr, _ = roc_curve(y, p)
        ks = float(np.max(tpr - fpr))
        pr_auc = float(average_precision_score(y, p))
        brier = float(brier_score_loss(y, p))
        ll = float(log_loss(y, p))

        report[split_name] = {
            "auc": round(auc, 4),
            "gini": round(gini, 4),
            "ks_statistic": round(ks, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "log_loss": round(ll, 4),
            "deciles": compute_decile_table(p, y),
            "calibration": compute_calibration_table(p, y)
        }

    report_json = artifacts / "model_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Write human-readable markdown report
    report_md = reports / "model_report.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Model Performance & Evaluation Report\n\n")
        f.write(f"**Model Version:** {meta.get('model_version', 'N/A')}  \n")
        f.write(f"**Default Threshold:** {default_threshold}  \n\n")
        f.write("## Overall Metrics Across Splits\n\n")
        f.write("| Metric | Validation (2018) | Out-of-Time (2019) | Stress (2020-21) |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for m_key in ["auc", "gini", "ks_statistic", "pr_auc", "brier_score", "log_loss"]:
            f.write(f"| **{m_key.upper()}** | {report['valid'][m_key]} | {report['oot'][m_key]} | {report['stress'][m_key]} |\n")

    print(f"Model report written to {report_json} and {report_md}")

    # MD-06: Sanity gates
    oot_auc = report["oot"]["auc"]
    val_auc = report["valid"]["auc"]
    floor = cfg["model"]["sanity"]["oot_auc_floor"]
    warn_drop = cfg["model"]["sanity"]["oot_auc_drop_warn"]

    if oot_auc < floor:
        raise ValueError(f"Sanity gate failed: OOT AUC {oot_auc:.4f} is below floor {floor}")
    if (val_auc - oot_auc) > warn_drop:
        print(f"Warning: OOT AUC {oot_auc:.4f} dropped more than {warn_drop} below Valid AUC {val_auc:.4f}")

    # MD-10: Global feature importance (TreeSHAP on 5,000 valid loans)
    print("Computing global TreeSHAP feature importance on 5,000 valid loans...")
    valid_sample = valid_df.sample(min(5000, len(valid_df)), random_state=cfg.get("seed", 42))
    X_sample = feature_matrix(valid_sample, cfg)
    contribs = booster.predict(X_sample, pred_contrib=True) # shape: (n_samples, n_features + 1)
    feat_contribs = np.abs(contribs[:, :-1]).mean(axis=0)

    feat_names = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    importance_list = [
        {"feature": feat, "mean_abs_contribution": round(float(imp), 5)}
        for feat, imp in sorted(zip(feat_names, feat_contribs), key=lambda x: x[1], reverse=True)
    ]

    imp_path = artifacts / "global_importance.json"
    with open(imp_path, "w", encoding="utf-8") as f:
        json.dump(importance_list, f, indent=2)
    print(f"Global importance written to {imp_path}")

    # DR-02, DR-04: Reference bins for PSI drift monitoring
    print("Computing reference bins (feature bins on train, score bins on validation)...")
    train_df = apply_categories(pd.read_parquet(processed / "train.parquet"), categories)
    ref_bins = {}
    floor = cfg.get("drift", {}).get("smoothing_floor", 0.0001)

    for f_name in cfg["features"]["numeric"]:
        ref_bins[f_name] = build_spec(train_df[f_name], "numeric", floor=floor)
    for f_name in cfg["features"]["categorical"]:
        ref_bins[f_name] = build_spec(train_df[f_name], "categorical", floor=floor)

    # Score reference is VALIDATION scores (DR-04)
    ref_bins["score"] = build_spec(pd.Series(valid_df["probability"].values), "numeric", floor=floor)

    ref_bins_path = artifacts / "reference_bins.json"
    with open(ref_bins_path, "w", encoding="utf-8") as f:
        json.dump(ref_bins, f, indent=2)
    print(f"Reference bins written to {ref_bins_path}")

if __name__ == "__main__":
    main()
