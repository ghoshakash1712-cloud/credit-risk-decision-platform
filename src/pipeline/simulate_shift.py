import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from src.common.config import load_config, project_path
from src.common.features import feature_matrix, apply_categories, add_derived
from src.common.psi import window_report

def main():
    cfg = load_config()
    processed = project_path(cfg, "processed")
    artifacts = project_path(cfg, "artifacts")

    with open(artifacts / "reference_bins.json", "r", encoding="utf-8") as f:
        ref_bins = json.load(f)

    with open(artifacts / "global_importance.json", "r", encoding="utf-8") as f:
        importance_data = json.load(f)
    imp_features = [item["feature"] for item in importance_data]

    with open(artifacts / "categories.json", "r", encoding="utf-8") as f:
        categories = json.load(f)

    with open(artifacts / "quarter_median_rate.json", "r", encoding="utf-8") as f:
        quarter_median = json.load(f)

    booster = lgb.Booster(model_file=str(artifacts / "model.txt"))

    print("Loading base OOT split for shock simulation...")
    base_raw = apply_categories(pd.read_parquet(processed / "oot.parquet"), categories)
    base_derived = add_derived(base_raw.copy(), quarter_median)
    base_probs = booster.predict(feature_matrix(base_derived, cfg))
    base_mean_prob = float(np.mean(base_probs))

    shocks_cfg = cfg.get("drift", {}).get("shocks", {})
    results = {}

    for scenario_name, spec in shocks_cfg.items():
        results[scenario_name] = {}
        for severity in ["mild", "moderate", "severe"]:
            shift_val = float(spec[severity])
            df_shocked = base_raw.copy()

            target_cols = spec.get("columns", [spec.get("column")])
            for col in target_cols:
                if col in df_shocked.columns:
                    shifted = pd.to_numeric(df_shocked[col], errors="coerce") + shift_val
                    if "clip" in spec:
                        lo, hi = spec["clip"]
                        shifted = shifted.clip(lower=lo, upper=hi)
                    if "cap" in spec:
                        shifted = shifted.clip(upper=spec["cap"])
                    df_shocked[col] = shifted

            # DR-12: Rebuild derived features using the SAME feature code
            df_shocked = add_derived(df_shocked, quarter_median)

            # Predict shocked probabilities
            shocked_probs = booster.predict(feature_matrix(df_shocked, cfg))
            mean_shocked_prob = float(np.mean(shocked_probs))

            # DR-14: Compute PSI report against baseline references
            rep = window_report(df_shocked, shocked_probs, ref_bins, imp_features, cfg)
            rep.update({
                "scenario": scenario_name,
                "severity": severity,
                "simulated": True,
                "mean_probability_baseline": round(base_mean_prob, 4),
                "mean_probability_shocked": round(mean_shocked_prob, 4),
                "mean_probability_change": round(mean_shocked_prob - base_mean_prob, 4)
            })

            results[scenario_name][severity] = rep
            print(f"  Scenario '{scenario_name}' [{severity}]: delta_mean_p={rep['mean_probability_change']:+.4f}, score_psi={rep['score_psi']:.4f}, status={rep['status']}")

    sim_path = artifacts / "psi_simulated.json"
    with open(sim_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Simulated shocks (12 scenarios) written to {sim_path}")

if __name__ == "__main__":
    main()
