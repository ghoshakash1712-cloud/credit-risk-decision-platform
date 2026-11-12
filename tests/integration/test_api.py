import json
import shutil
import tempfile
from pathlib import Path
import pytest
import pandas as pd
from api.app import create_app
from api.loaders import Store

@pytest.fixture(scope="module")
def store():
    return Store()

@pytest.fixture(scope="module")
def client(store):
    app = create_app(store=store)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c

def test_ts08_health(client):
    res = client.get("/v1/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert "model_version" in data
    assert "data_release" in data
    assert "loans_per_split" in data
    assert "valid" in data["loans_per_split"]
    assert "oot" in data["loans_per_split"]
    assert "stress" in data["loans_per_split"]
    assert res.headers["Cache-Control"] == "no-store"
    assert "X-Request-ID" in res.headers

def test_ts08_meta(client):
    res = client.get("/v1/meta")
    assert res.status_code == 200
    data = res.get_json()
    for key in ["default_threshold", "default_fp_cost_pct", "default_fn_cost_pct", "slider_ranges", "splits", "fairness", "psi_bands", "features"]:
        assert key in data
    assert res.headers["Cache-Control"] == "max-age=60"

def test_ts08_loans_and_picker(client, store):
    # Missing required parameter split
    res = client.get("/v1/loans")
    assert res.status_code == 400
    err = res.get_json()["error"]
    assert err["code"] == "invalid_parameter"
    
    # Unknown parameter typo (AP-04)
    res = client.get("/v1/loans?split=valid&bogus=1")
    assert res.status_code == 400
    assert res.get_json()["error"]["code"] == "invalid_parameter"

    # Valid listing
    res = client.get("/v1/loans?split=valid&limit=10")
    assert res.status_code == 200
    data = res.get_json()
    assert data["total"] > 0
    assert len(data["items"]) == 10
    sample_id = data["items"][0]["loan_id"]
    
    # Prefix search
    prefix = sample_id[:5]
    res_search = client.get(f"/v1/loans?split=valid&search={prefix}&limit=5")
    assert res_search.status_code == 200
    data_search = res_search.get_json()
    assert len(data_search["items"]) > 0
    for item in data_search["items"]:
        assert item["loan_id"].startswith(prefix)

def test_ts08_loan_detail_predict_explain(client, store):
    valid_id = str(store.splits["valid"].df.index[0])
    
    # 1. Detail
    res = client.get(f"/v1/loans/{valid_id}")
    assert res.status_code == 200
    detail = res.get_json()
    assert detail["loan_id"] == valid_id
    assert "features" in detail
    assert "original_upb" in detail
    assert "actual_outcome" in detail
    
    # 2. Predict default threshold
    res_pred = client.get(f"/v1/predict/{valid_id}")
    assert res_pred.status_code == 200
    pdata = res_pred.get_json()
    assert 0.0 <= pdata["probability"] <= 1.0
    assert pdata["threshold_source"] == "default"
    assert pdata["decision"] in ["approve", "reject"]
    
    # Predict custom threshold (AP-10)
    res_pred_custom = client.get(f"/v1/predict/{valid_id}?threshold=0.01")
    assert res_pred_custom.status_code == 200
    pdata2 = res_pred_custom.get_json()
    assert pdata2["threshold"] == 0.01
    assert pdata2["threshold_source"] == "supplied"
    assert pdata2["decision"] == ("reject" if pdata2["probability"] >= 0.01 else "approve")
    
    # Predict out-of-range threshold
    res_bad_t = client.get(f"/v1/predict/{valid_id}?threshold=1.5")
    assert res_bad_t.status_code == 400
    assert res_bad_t.get_json()["error"]["code"] == "invalid_parameter"
    
    # 3. Explain (local TreeSHAP)
    res_exp = client.get(f"/v1/explain/{valid_id}")
    assert res_exp.status_code == 200
    edata = res_exp.get_json()
    assert "base_value" in edata
    assert "contributions" in edata
    assert len(edata["contributions"]) == 10
    assert "other_contribution_log_odds" in edata
    # Sum of contributions + base + other reconciles with sigmoid
    log_odds_total = edata["base_value"] + sum(c["contribution_log_odds"] for c in edata["contributions"]) + edata["other_contribution_log_odds"]
    import math
    reconciled_prob = 1.0 / (1.0 + math.exp(-log_odds_total))
    assert abs(reconciled_prob - edata["probability"]) < 1e-4

def test_ts08_unknown_and_training_loans_404(client):
    # Non-existent loan
    res = client.get("/v1/loans/NONEXISTENT_LOAN_123")
    assert res.status_code == 404
    assert res.get_json()["error"]["code"] == "loan_not_found"

    # Training loan must NOT be served (MD-14)
    train_df = pd.read_parquet("data/processed/train.parquet")
    train_id = str(train_df["loan_id"].iloc[0])
    
    res_tr_loan = client.get(f"/v1/loans/{train_id}")
    assert res_tr_loan.status_code == 404
    assert res_tr_loan.get_json()["error"]["code"] == "loan_not_found"
    
    res_tr_pred = client.get(f"/v1/predict/{train_id}")
    assert res_tr_pred.status_code == 404
    assert res_tr_pred.get_json()["error"]["code"] == "loan_not_found"

def test_ts08_optimize(client):
    # Optimal mode
    res = client.get("/v1/optimize?split=valid&fp_cost_pct=2.0&fn_cost_pct=25.0&mode=optimal")
    assert res.status_code == 200
    data = res.get_json()
    assert "threshold" in data
    assert "analytic_threshold" in data
    assert "calibration_warning" in data
    assert "metrics" in data
    assert "splits" in data
    assert len(data["splits"]) == 3
    assert "curve" in data
    assert len(data["curve"]) <= 200
    
    # Manual mode requiring threshold
    res_manual_missing = client.get("/v1/optimize?split=valid&mode=manual")
    assert res_manual_missing.status_code == 400
    assert res_manual_missing.get_json()["error"]["code"] == "invalid_parameter"
    
    res_manual = client.get("/v1/optimize?split=valid&mode=manual&threshold=0.08")
    assert res_manual.status_code == 200
    assert res_manual.get_json()["threshold"] == 0.08

def test_ts08_fairness(client):
    # Shared resolve logic with optimize (AP-09)
    res_opt = client.get("/v1/optimize?split=valid&fp_cost_pct=2.5&fn_cost_pct=20.0&mode=optimal")
    res_fair = client.get("/v1/fairness?split=valid&fp_cost_pct=2.5&fn_cost_pct=20.0&mode=optimal")
    
    assert res_opt.status_code == 200
    assert res_fair.status_code == 200
    opt_t = res_opt.get_json()["threshold"]
    fair_t = res_fair.get_json()["threshold_used"]
    assert abs(opt_t - fair_t) < 1e-9  # Exact match (FA-12, AP-09)

    fdata = res_fair.get_json()
    assert "baseline" in fdata
    assert "attributes" in fdata
    assert "first_time_homebuyer" in fdata["attributes"]
    
    # Specific attribute query
    res_attr = client.get("/v1/fairness?split=valid&attribute=borrower_count")
    assert res_attr.status_code == 200
    adata = res_attr.get_json()["attributes"]
    assert list(adata.keys()) == ["borrower_count"]

def test_ts08_drift_and_reports(client):
    # Real drift: oot
    res_drift_oot = client.get("/v1/drift?source=real&window=oot")
    assert res_drift_oot.status_code == 200
    d_oot = res_drift_oot.get_json()
    assert "status" in d_oot
    assert "score_psi" in d_oot
    assert "features" in d_oot
    assert res_drift_oot.headers["Cache-Control"] == "max-age=60"

    # Real drift: quarterly
    res_quarterly = client.get("/v1/drift?source=real&window=quarterly")
    assert res_quarterly.status_code == 200
    assert "series" in res_quarterly.get_json()

    # Simulated shock
    res_sim = client.get("/v1/drift?source=simulated&scenario=rate_shock&severity=severe")
    assert res_sim.status_code == 200
    sim_data = res_sim.get_json()
    assert sim_data["scenario"] == "rate_shock"
    assert sim_data["severity"] == "severe"
    assert "score_psi" in sim_data

    # Global importance
    res_imp = client.get("/v1/explain/global?top=15")
    assert res_imp.status_code == 200
    assert len(res_imp.get_json()["features"]) == 15
    assert res_imp.headers["Cache-Control"] == "max-age=60"

    # Model report
    res_rep = client.get("/v1/model-report")
    assert res_rep.status_code == 200
    rep_data = res_rep.get_json()
    assert "valid" in rep_data and "oot" in rep_data and "stress" in rep_data
    assert res_rep.headers["Cache-Control"] == "max-age=60"

def test_ts09_startup_self_check_tampered_model():
    """TS-09: Copy artifacts to temp dir, truncate model.txt, assert Store() raises SystemExit."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        artifacts_tmp = tmp_path / "artifacts"
        shutil.copytree("artifacts", artifacts_tmp)
        
        # Corrupt / truncate model.txt
        model_file = artifacts_tmp / "model.txt"
        with open(model_file, "w", encoding="utf-8") as f:
            f.write("corrupted lightgbm model header without tree data\n")
            
        with pytest.raises(SystemExit):
            Store(repo_root=str(tmp_path))
