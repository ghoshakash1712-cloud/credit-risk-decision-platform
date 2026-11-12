import math
from flask import Blueprint, request, current_app, jsonify
import numpy as np
import pandas as pd
from api.errors import ApiError
from api.params import parse_params
from src.common import cost
from src.common.groups import audit_attribute

v1 = Blueprint("v1", __name__)

def get_store():
    return current_app.extensions["store"]

def clean_json(val):
    if isinstance(val, dict):
        return {k: clean_json(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [clean_json(v) for v in val]
    elif isinstance(val, (np.floating, float)):
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)
    elif isinstance(val, (np.integer, int)):
        return int(val)
    elif isinstance(val, (np.bool_, bool)):
        return bool(val)
    elif pd.isna(val):
        return None
    return val

def ok(**body):
    store = get_store()
    meta = store.meta
    res = {
        "model_version": meta.get("model_version"),
        "data_release": meta.get("data_release")
    }
    res.update(body)
    return jsonify(clean_json(res))

SPEC_OPTIMIZE = {
    "split": {"type": str, "choices": ["valid", "oot", "stress"], "default": "valid"},
    "fp_cost_pct": {"type": float, "min": 0.1, "max": 20.0, "default": None},
    "fn_cost_pct": {"type": float, "min": 1.0, "max": 100.0, "default": None},
    "mode": {"type": str, "choices": ["optimal", "manual"], "default": "optimal"},
    "threshold": {"type": float, "min": 0.0, "max": 1.0, "default": None}
}

def resolve(args, extra_spec=None):
    store = get_store()
    rules = dict(SPEC_OPTIMIZE)
    if extra_spec:
        rules.update(extra_spec)
        
    p = parse_params(args, rules)
    if p["fp_cost_pct"] is None:
        p["fp_cost_pct"] = float(store.meta.get("default_cost_params", {}).get("fp_cost_pct", 2.0))
    if p["fn_cost_pct"] is None:
        p["fn_cost_pct"] = float(store.meta.get("default_cost_params", {}).get("fn_cost_pct", 25.0))
        
    if p["mode"] == "manual" and p["threshold"] is None:
        raise ApiError(
            code="invalid_parameter",
            status_code=400,
            message="threshold is required when mode is manual",
            details={"parameter": "threshold"}
        )
        
    g = p["fp_cost_pct"] / 100.0
    L = p["fn_cost_pct"] / 100.0
    split_name = p["split"]
    split_obj = store.splits[split_name]
    
    opt = cost.optimize(split_obj.prep, g, L)
    t = opt.threshold if p["mode"] == "optimal" else float(p["threshold"])
    return p, g, L, split_obj, opt, t

@v1.get("/health")
def health():
    store = get_store()
    counts = {s: int(len(store.splits[s].df)) for s in ["valid", "oot", "stress"]}
    return ok(status="ok", loans_per_split=counts)

@v1.get("/meta")
def meta():
    store = get_store()
    cfg = store.cfg
    default_cost = store.meta.get("default_cost_params", {})
    return ok(
        default_threshold=float(store.meta["default_threshold"]),
        default_fp_cost_pct=float(default_cost.get("fp_cost_pct", 2.0)),
        default_fn_cost_pct=float(default_cost.get("fn_cost_pct", 25.0)),
        slider_ranges=cfg.get("costs", {}).get("sliders", {}),
        splits=["valid", "oot", "stress"],
        fairness=cfg.get("fairness", {}),
        psi_bands=cfg.get("drift", {}).get("bands", {}),
        features=cfg.get("features", {}).get("numeric", []) + cfg.get("features", {}).get("categorical", [])
    )

@v1.get("/loans")
def list_loans():
    store = get_store()
    rules = {
        "split": {"type": str, "choices": ["valid", "oot", "stress"], "required": True},
        "limit": {"type": int, "min": 1, "max": 500, "default": 100},
        "offset": {"type": int, "min": 0, "default": 0},
        "search": {"type": str, "default": None}
    }
    p = parse_params(request.args, rules)
    df = store.splits[p["split"]].df
    if p["search"]:
        q = str(p["search"]).strip()
        df = df[df["loan_id"].astype(str).str.startswith(q)]
    total = len(df)
    sub = df.iloc[p["offset"] : p["offset"] + p["limit"]]
    items = [
        {
            "loan_id": str(r.loan_id),
            "vintage": int(r.vintage),
            "actual_outcome": int(r.default_24m)
        }
        for _, r in sub.iterrows()
    ]
    return ok(
        split=p["split"],
        total=total,
        limit=p["limit"],
        offset=p["offset"],
        items=items
    )

@v1.get("/loans/<loan_id>")
def get_loan(loan_id):
    store = get_store()
    if loan_id not in store.holdout.index:
        raise ApiError(
            code="loan_not_found",
            status_code=404,
            message=f"Loan {loan_id} not found in holdout splits",
            details={"loan_id": loan_id}
        )
    row = store.holdout.loc[loan_id]
    feature_cols = store.features.columns.tolist()
    feats = {col: (None if pd.isna(row[col]) else row[col]) for col in feature_cols}
    return ok(
        loan_id=str(row["loan_id"]),
        split=str(row["split"]),
        vintage=int(row["vintage"]),
        original_upb=float(row["orig_upb"]),
        actual_outcome=int(row["default_24m"]),
        features=feats
    )

@v1.get("/predict/<loan_id>")
def predict(loan_id):
    store = get_store()
    rules = {
        "threshold": {"type": float, "min": 0.0, "max": 1.0, "default": None}
    }
    p = parse_params(request.args, rules)
    if loan_id not in store.holdout.index:
        raise ApiError(
            code="loan_not_found",
            status_code=404,
            message=f"Loan {loan_id} not found in holdout splits",
            details={"loan_id": loan_id}
        )
    X = store.features.loc[[loan_id]]
    prob = float(store.booster.predict(X)[0])
    t = p["threshold"] if p["threshold"] is not None else float(store.meta["default_threshold"])
    t_src = "supplied" if p["threshold"] is not None else "default"
    decision = "reject" if prob >= t else "approve"
    return ok(
        loan_id=loan_id,
        probability=prob,
        threshold=t,
        threshold_source=t_src,
        decision=decision
    )

@v1.get("/explain/<loan_id>")
def explain(loan_id):
    store = get_store()
    if loan_id not in store.holdout.index:
        raise ApiError(
            code="loan_not_found",
            status_code=404,
            message=f"Loan {loan_id} not found in holdout splits",
            details={"loan_id": loan_id}
        )
    X = store.features.loc[[loan_id]]
    c = store.booster.predict(X, pred_contrib=True)[0]
    base_val = float(c[-1])
    vals = c[:-1]
    feature_names = store.features.columns.tolist()
    
    top_indices = np.argsort(-np.abs(vals))[:10]
    contributions = [
        {
            "feature": feature_names[i],
            "value": None if pd.isna(X.iloc[0, i]) else X.iloc[0, i],
            "contribution_log_odds": float(vals[i])
        }
        for i in top_indices
    ]
    other_contrib = float(vals.sum() - vals[top_indices].sum())
    prob = float(1.0 / (1.0 + math.exp(-float(c.sum()))))
    
    return ok(
        loan_id=loan_id,
        base_value=base_val,
        probability=prob,
        contributions=contributions,
        other_contribution_log_odds=other_contrib
    )

@v1.get("/explain/global")
def explain_global():
    store = get_store()
    rules = {"top": {"type": int, "min": 1, "max": 100, "default": 20}}
    p = parse_params(request.args, rules)
    raw = store.reports["importance"]
    feats = raw if isinstance(raw, list) else raw.get("features", [])
    return ok(features=feats[:p["top"]])

@v1.get("/optimize")
def optimize():
    store = get_store()
    p, g, L, split_obj, opt, t = resolve(request.args)
    
    analytic = cost.analytic_threshold(g, L)
    factor = float(store.cfg.get("costs", {}).get("calibration_warning_factor", 3.0))
    warn = cost.calibration_warning(opt.threshold, analytic, factor)
    metrics = cost.metrics_at(split_obj.prep, t, g, L)
    
    splits_comparison = []
    for s_name in ["valid", "oot", "stress"]:
        s_prep = store.splits[s_name].prep
        splits_comparison.append({
            "split": s_name,
            "metrics": cost.metrics_at(s_prep, t, g, L)
        })
        
    return ok(
        split=p["split"],
        inputs={
            "fp_cost_pct": p["fp_cost_pct"],
            "fn_cost_pct": p["fn_cost_pct"],
            "mode": p["mode"],
            **({"threshold": p["threshold"]} if p["mode"] == "manual" else {})
        },
        threshold=t,
        reject_none=bool(metrics["rejection_rate"] == 0.0),
        analytic_threshold=analytic,
        calibration_warning=warn,
        metrics=metrics,
        splits=splits_comparison,
        curve=opt.curve
    )

@v1.get("/fairness")
def fairness():
    store = get_store()
    configured_attr_list = store.cfg.get("fairness", {}).get("attributes", [])
    configured_keys = [a["key"] if isinstance(a, dict) else str(a) for a in configured_attr_list]
    fairness_spec = {
        "attribute": {"type": str, "choices": configured_keys, "default": None}
    }
    p, g, L, split_obj, opt, t = resolve(request.args, extra_spec=fairness_spec)
    
    attr_choice = p.get("attribute")
    attrs = [attr_choice] if attr_choice else configured_keys
    
    baseline_split = store.reports.get("fairness_baseline", {}).get(p["split"], {})
    baseline = {a: baseline_split.get(a) for a in attrs}
    notes_map = store.cfg.get("fairness", {}).get("notes", {})
    
    audit_results = {}
    for a in attrs:
        group_codes, labels = split_obj.groups[a]
        audit = audit_attribute(
            split_obj.prob,
            split_obj.y,
            group_codes,
            labels,
            t,
            store.cfg.get("fairness", {})
        )
        audit["notes"] = notes_map.get(a, "")
        audit_results[a] = audit
        
    return ok(
        split=p["split"],
        threshold_used=t,
        baseline=baseline,
        attributes=audit_results
    )

@v1.get("/drift")
def drift():
    store = get_store()
    source_rules = {
        "source": {"type": str, "choices": ["real", "simulated"], "default": "real"},
        "window": {"type": str, "choices": ["valid", "oot", "stress", "quarterly"], "default": "oot"},
        "scenario": {"type": str, "choices": ["debt_stress", "credit_deterioration", "rate_shock", "leverage_squeeze"], "default": None},
        "severity": {"type": str, "choices": ["mild", "moderate", "severe"], "default": None}
    }
    p = parse_params(request.args, source_rules)
    
    if p["source"] == "real":
        report = store.reports.get("psi_report", {})
        if p["window"] == "quarterly":
            return ok(
                source="real",
                window="quarterly",
                series=report.get("quarterly", [])
            )
        else:
            w_data = report.get("windows", {}).get(p["window"])
            if not w_data:
                raise ApiError(
                    code="not_computable",
                    status_code=422,
                    message=f"No PSI data for window {p['window']}",
                    details={"window": p["window"]}
                )
            return ok(
                source="real",
                window=p["window"],
                **w_data
            )
    else: # simulated
        if p["scenario"] is None or p["severity"] is None:
            raise ApiError(
                code="invalid_parameter",
                status_code=400,
                message="scenario and severity are required when source is simulated",
                details={"scenario": p["scenario"], "severity": p["severity"]}
            )
        sim_report = store.reports.get("psi_simulated", {})
        scen_data = sim_report.get(p["scenario"], {}).get(p["severity"])
        if not scen_data:
            raise ApiError(
                code="invalid_parameter",
                status_code=400,
                message=f"Unknown scenario/severity combination: {p['scenario']}/{p['severity']}",
                details={"scenario": p["scenario"], "severity": p["severity"]}
            )
        return ok(
            source="simulated",
            **scen_data
        )

@v1.get("/model-report")
def model_report():
    store = get_store()
    return ok(**store.reports["model_report"])
