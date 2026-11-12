import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.common.config import load_config, project_path
from src.common.features import clean_origination, compute_quarter_median, add_derived, apply_categories
from src.common.groups import add_group_columns

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

    # Exclusions (DP-06):
    # 1. Defect: code 96
    has_defect = (zb.isin(cfg["data"]["exclude_zero_balance_codes"])).groupby(p["loan_id"]).any()
    # 2. No performance records
    no_perf = ~orig["loan_id"].isin(p["loan_id"])
    # 3. HARP: harp_indicator == 'Y'
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

def write_loss_severity(cohort: pd.DataFrame, cfg: dict, artifacts_dir: Path):
    """DP-11: Mean of Actual Loss divided by original UPB over defaulted training-vintage loans."""
    train_defaults = cohort[(cohort["split"] == "train") & (cohort["default_24m"] == 1)].copy()
    if "actual_loss" in train_defaults.columns:
        train_defaults["actual_loss_num"] = pd.to_numeric(train_defaults["actual_loss"], errors="coerce")
        train_defaults["orig_upb_num"] = pd.to_numeric(train_defaults["orig_upb"], errors="coerce")
        has_loss = train_defaults["actual_loss_num"].notna() & (train_defaults["actual_loss_num"] > 0)
        valid_losses = train_defaults[has_loss]
        if len(valid_losses) > 0:
            sev = (valid_losses["actual_loss_num"] / valid_losses["orig_upb_num"]).clip(0.0, 1.0)
            mean_sev = float(sev.mean())
            med_sev = float(sev.median())
            n_sev = len(sev)
        else:
            mean_sev = 0.25 # standard industry default
            med_sev = 0.25
            n_sev = 0
    else:
        mean_sev = 0.25
        med_sev = 0.25
        n_sev = 0

    loss_data = {
        "mean": round(mean_sev, 4),
        "median": round(med_sev, 4),
        "n": n_sev
    }
    loss_path = artifacts_dir / "loss_severity.json"
    with open(loss_path, "w", encoding="utf-8") as f:
        json.dump(loss_data, f, indent=2)
    print(f"Loss severity written to {loss_path}: {loss_data}")
    return loss_data

def main():
    cfg = load_config()
    interim = project_path(cfg, "interim")
    processed = project_path(cfg, "processed")
    artifacts = project_path(cfg, "artifacts")
    processed.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    vintages_cfg = cfg["data"]["vintages"]
    all_years = sorted(list(set(sum(vintages_cfg.values(), []))))

    year_to_split = {}
    for s_name, y_list in vintages_cfg.items():
        for y in y_list:
            year_to_split[y] = s_name

    frames = []
    exclusions = {}

    for year in all_years:
        print(f"Building dataset for vintage {year}...")
        orig = clean_origination(pd.read_parquet(interim / f"orig_{year}.parquet"), cfg)
        perf = pd.read_parquet(interim / f"perf_{year}.parquet")

        labels, reasons = label_loans(orig, perf, cfg)
        exclusions[str(year)] = reasons

        # Merge origination with labels and loss info
        merged = orig.merge(labels, on="loan_id", how="inner")
        if "actual_loss" in perf.columns:
            perf_sub = perf[["loan_id", "actual_loss"]].drop_duplicates("loan_id")
            merged = merged.merge(perf_sub, on="loan_id", how="left")

        frames.append(merged)

    cohort = pd.concat(frames, ignore_index=True)
    print(f"Total cohort size across all vintages: {len(cohort)}")

    # Compute quarter medians across the cohort before splitting (DP-09)
    quarter_median = compute_quarter_median(cohort)
    cohort = add_derived(cohort, quarter_median)

    # Assign split
    cohort["split"] = cohort["vintage"].map(year_to_split)
    cohort = add_group_columns(cohort)

    # Categories from TRAIN split only (MD-01)
    train_mask = cohort["split"] == "train"
    train_df = cohort[train_mask]
    categorical_cols = cfg["features"]["categorical"]
    categories = {}
    for col in categorical_cols:
        if col in train_df.columns:
            cats = sorted([str(x).strip() for x in train_df[col].dropna().unique() if str(x).strip() not in ("", "nan", "None", "<NA>")])
            categories[col] = cats

    # Save artifacts
    with open(artifacts / "quarter_median_rate.json", "w", encoding="utf-8") as f:
        json.dump({k: round(float(v), 4) for k, v in quarter_median.items()}, f, indent=2)

    with open(artifacts / "categories.json", "w", encoding="utf-8") as f:
        json.dump(categories, f, indent=2)

    default_rates = {}
    for split_name in ["train", "valid", "oot", "stress"]:
        part = cohort[cohort["split"] == split_name].copy()
        part = apply_categories(part, categories)
        out_file = processed / f"{split_name}.parquet"
        part.to_parquet(out_file, index=False)
        def_rate = float(part["default_24m"].mean()) if len(part) > 0 else 0.0
        default_rates[split_name] = round(def_rate, 4)
        print(f"  Split '{split_name}': {len(part)} loans, default rate {def_rate:.4f} -> {out_file}")

    write_loss_severity(cohort, cfg, artifacts)

    metadata_path = artifacts / "metadata.json"
    meta = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    meta.update({
        "data_release": str(cfg["data"]["release"]),
        "vintages": cfg["data"]["vintages"],
        "exclusions": exclusions,
        "default_rates": default_rates,
        "label_version": "v1-24m-90dpd-or-credit-event",
        "features": cfg["features"]
    })

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Dataset build complete. Metadata updated at {metadata_path}")

if __name__ == "__main__":
    main()
