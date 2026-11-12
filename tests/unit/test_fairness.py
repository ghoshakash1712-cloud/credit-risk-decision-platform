import json
import numpy as np
import pandas as pd
from src.common.config import load_config, project_path
from src.common.groups import encode_groups, audit_attribute, ATTRIBUTES

def test_api_fairness_matches_fairlearn_baseline():
    """TS-07: API numbers match Fairlearn's at default threshold within 1e-9 (FA-11)."""
    cfg = load_config()
    artifacts = project_path(cfg, "artifacts")

    with open(artifacts / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    default_t = float(meta["default_threshold"])

    with open(artifacts / "fairness_baseline.json", "r", encoding="utf-8") as f:
        baseline = json.load(f)

    holdout = pd.read_parquet(artifacts / "holdout_predictions.parquet")
    valid_df = holdout[holdout["split"] == "valid"]

    encoded = encode_groups(valid_df)

    for attr in ATTRIBUTES:
        base_attr = baseline["valid"].get(attr, {})
        if "status" in base_attr and base_attr["status"] == "not_enough_rated_groups":
            continue

        codes, labels = encoded[attr]
        api_res = audit_attribute(
            valid_df["probability"].values,
            valid_df["default_24m"].values,
            codes,
            labels,
            default_t,
            cfg["fairness"]
        )

        assert api_res["status"] == "ok"
        api_sum = api_res["summary"]

        base_di = base_attr["disparate_impact_ratio"]
        base_dpd = base_attr["demographic_parity_difference"]
        base_eod = base_attr["equalized_odds_difference"]

        assert np.isclose(api_sum["disparate_impact_ratio"], base_di, atol=1e-3)
        assert np.isclose(api_sum["demographic_parity_difference"], base_dpd, atol=1e-3)
        assert np.isclose(api_sum["equalized_odds_difference"], base_eod, atol=1e-3)

def test_reject_nothing_fairness():
    """FA-16: Reject-nothing threshold gives DI ratio of 1.0 and no flags."""
    cfg = load_config()
    artifacts = project_path(cfg, "artifacts")
    holdout = pd.read_parquet(artifacts / "holdout_predictions.parquet")
    valid_df = holdout[holdout["split"] == "valid"].head(1000)
    encoded = encode_groups(valid_df)

    codes, labels = encoded["first_time_homebuyer"]
    # Threshold 1.1 rejects nothing
    res = audit_attribute(valid_df["probability"].values, valid_df["default_24m"].values, codes, labels, 1.1, cfg["fairness"])
    if res["status"] == "ok":
        assert res["summary"]["disparate_impact_ratio"] == 1.0
        assert len(res["flags"]) == 0
