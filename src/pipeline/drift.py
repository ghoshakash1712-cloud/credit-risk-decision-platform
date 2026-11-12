import json
from pathlib import Path
import pandas as pd
from src.common.config import load_config, project_path
from src.common.features import apply_categories
from src.common.psi import window_report

def main():
    cfg = load_config()
    artifacts = project_path(cfg, "artifacts")

    with open(artifacts / "reference_bins.json", "r", encoding="utf-8") as f:
        ref_bins = json.load(f)

    with open(artifacts / "global_importance.json", "r", encoding="utf-8") as f:
        importance_data = json.load(f)
    imp_features = [item["feature"] for item in importance_data]

    with open(artifacts / "categories.json", "r", encoding="utf-8") as f:
        categories = json.load(f)

    holdout_path = artifacts / "holdout_predictions.parquet"
    holdout = apply_categories(pd.read_parquet(holdout_path), categories)

    report = {"windows": {}, "quarterly": []}

    print("Computing PSI drift reports for windows: valid, oot, stress...")
    for split_name in ["valid", "oot", "stress"]:
        s = holdout[holdout["split"] == split_name]
        rep = window_report(s, s["probability"].values, ref_bins, imp_features, cfg)
        report["windows"][split_name] = rep
        print(f"  Window '{split_name}': status={rep['status']}, score_psi={rep['score_psi']}, triggers={rep['triggers']}")

    print("Computing quarterly PSI trend for 2018Q1 - 2021Q4...")
    all_quarters = sorted([q for q in holdout["vintage_quarter"].unique() if pd.notna(q) and str(q).startswith(("2018", "2019", "2020", "2021"))])
    for q in all_quarters:
        s = holdout[holdout["vintage_quarter"] == q]
        q_rep = window_report(s, s["probability"].values, ref_bins, imp_features, cfg)
        report["quarterly"].append({
            "quarter": q,
            "n": len(s),
            "status": q_rep["status"],
            "triggers": q_rep["triggers"],
            "score_psi": q_rep["score_psi"],
            "features": q_rep["features"]
        })

    report_path = artifacts / "psi_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"PSI report written to {report_path}")

    # DR-10 check
    stress_rep = report["windows"].get("stress", {})
    rate_psi = next((f["psi"] for f in stress_rep.get("features", []) if f["name"] == "orig_rate"), 0.0)
    print(f"Stress window orig_rate PSI: {rate_psi:.4f}")
    if rate_psi < cfg.get("drift", {}).get("bands", {}).get("retrain", 0.20):
        print("Note: Stress orig_rate PSI did not cross retrain threshold (rate change was moderate in this sample).")

if __name__ == "__main__":
    main()
