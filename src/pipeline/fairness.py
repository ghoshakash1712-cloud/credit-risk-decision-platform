import json
from pathlib import Path
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, demographic_parity_ratio
from src.common.config import load_config, project_path
from src.common.groups import ATTRIBUTES, UNRATED, encode_groups, audit_attribute

def main():
    cfg = load_config()
    artifacts = project_path(cfg, "artifacts")
    
    holdout_path = artifacts / "holdout_predictions.parquet"
    meta_path = artifacts / "metadata.json"
    
    if not holdout_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Holdout predictions or metadata not found")

    holdout = pd.read_parquet(holdout_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    default_t = float(meta["default_threshold"])
    min_loans = cfg.get("fairness", {}).get("size_rule", {}).get("min_loans", 500)
    min_defaults = cfg.get("fairness", {}).get("size_rule", {}).get("min_defaults", 30)

    print(f"Computing Fairlearn baseline at default threshold {default_t:.4f}...")
    baseline_out = {}

    for split_name in ["valid", "oot", "stress"]:
        s = holdout[holdout["split"] == split_name]
        prob = s["probability"].values
        y = s["default_24m"].values
        reject = (prob >= default_t).astype(int)
        approval = 1 - reject

        baseline_out[split_name] = {}

        for attr in ATTRIBUTES:
            col = f"grp_{attr}"
            labels = s[col].astype(str)

            # Apply size rule filter (FA-07)
            grp_counts = s.groupby(col).agg(loans=("default_24m", "count"), defaults=("default_24m", "sum"))
            rated_groups = grp_counts[
                (grp_counts["loans"] >= min_loans) & 
                (grp_counts["defaults"] >= min_defaults) & 
                (~grp_counts.index.isin(UNRATED))
            ].index.tolist()

            mask = labels.isin(rated_groups).values

            if len(rated_groups) >= 2 and mask.sum() > 0:
                y_sub = y[mask]
                rej_sub = reject[mask]
                app_sub = approval[mask]
                sens_sub = labels[mask]

                mf = MetricFrame(
                    metrics={
                        "rejection_rate": selection_rate,
                        "default_capture_rate": true_positive_rate,
                        "false_rejection_rate": false_positive_rate
                    },
                    y_true=y_sub,
                    y_pred=rej_sub,
                    sensitive_features=sens_sub
                )

                # Approval is favorable outcome for disparate impact (FA-10)
                di_ratio = demographic_parity_ratio(y_sub, app_sub, sensitive_features=sens_sub)
                dp_diff = demographic_parity_difference(y_sub, rej_sub, sensitive_features=sens_sub)
                eo_diff = equalized_odds_difference(y_sub, rej_sub, sensitive_features=sens_sub)

                by_grp_dict = {}
                for k, v in mf.by_group.to_dict().items():
                    by_grp_dict[k] = {str(gk): round(float(gv), 4) for gk, gv in v.items()}

                baseline_out[split_name][attr] = {
                    "by_group": by_grp_dict,
                    "demographic_parity_difference": round(float(dp_diff), 4),
                    "equalized_odds_difference": round(float(eo_diff), 4),
                    "disparate_impact_ratio": round(float(di_ratio), 4),
                    "rated_groups": rated_groups
                }
            else:
                baseline_out[split_name][attr] = {
                    "status": "not_enough_rated_groups",
                    "rated_groups": rated_groups
                }

    baseline_path = artifacts / "fairness_baseline.json"
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline_out, f, indent=2)
    print(f"Fairlearn baseline written to {baseline_path}")

if __name__ == "__main__":
    main()
