import numpy as np
import pandas as pd

def build_spec(series: pd.Series, kind: str, floor: float = 0.0001) -> dict:
    if kind == "numeric":
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) == 0 or s.nunique() <= 1:
            edges = []
        else:
            quantiles = np.linspace(0, 1, 11)
            raw_edges = np.unique(np.quantile(s, quantiles)[1:-1])
            edges = raw_edges.tolist()
        spec = {"kind": "numeric", "edges": edges}
    else:
        # Categorical
        s = series.astype(str).str.strip()
        s_clean = s[~s.isin(["", "nan", "None", "<NA>", "NaN"])].dropna()
        cats = sorted(s_clean.unique().tolist())
        spec = {"kind": "categorical", "categories": cats}

    codes = assign_bins(series, spec)
    n_b = n_bins(spec)
    spec["reference_share"] = shares(codes, n_b).tolist()
    return spec

def n_bins(spec: dict) -> int:
    if spec["kind"] == "numeric":
        # len(edges) + 1 numeric bins + 1 missing bin
        return len(spec["edges"]) + 2
    else:
        # len(categories) + 1 unseen + 1 missing
        return len(spec["categories"]) + 2

def assign_bins(series: pd.Series, spec: dict) -> np.ndarray:
    if spec["kind"] == "numeric":
        s = pd.to_numeric(series, errors="coerce").values
        edges = np.array(spec["edges"], dtype=float)
        missing_bin = len(edges) + 1
        if len(edges) == 0:
            codes = np.zeros(len(s), dtype=int)
            codes[np.isnan(s)] = missing_bin
        else:
            codes = np.digitize(s, edges, right=True)
            codes[np.isnan(s)] = missing_bin
        return codes
    else:
        cats = spec["categories"]
        cat_to_idx = {c: i for i, c in enumerate(cats)}
        unseen_bin = len(cats)
        missing_bin = len(cats) + 1

        codes = np.empty(len(series), dtype=int)
        for i, val in enumerate(series):
            if pd.isna(val) or str(val).strip() in ("", "nan", "None", "<NA>", "NaN"):
                codes[i] = missing_bin
            else:
                v_str = str(val).strip()
                codes[i] = cat_to_idx.get(v_str, unseen_bin)
        return codes

def shares(codes: np.ndarray, k: int) -> np.ndarray:
    if len(codes) == 0:
        return np.zeros(k, dtype=float)
    return np.bincount(codes, minlength=k).astype(float) / len(codes)

def calculate_psi(ref_share: np.ndarray, cur_share: np.ndarray, floor: float = 0.0001) -> float:
    """DR-01, DR-03: Population Stability Index calculation with smoothing floor."""
    a = np.maximum(cur_share, floor)
    e = np.maximum(ref_share, floor)
    val = np.sum((a - e) * np.log(a / e))
    return float(val)

def band_of(v: float, cfg: dict) -> str:
    """DR-07: stable < 0.10, warning 0.10 to 0.20, retrain > 0.20."""
    bands = cfg.get("drift", {}).get("bands", {"warn": 0.10, "retrain": 0.20})
    warn = bands["warn"]
    retrain = bands["retrain"]
    if v < warn:
        return "stable"
    elif v <= retrain:
        return "warning"
    else:
        return "retrain"

def worst_band(bands: list) -> str:
    if "retrain" in bands:
        return "retrain"
    if "warning" in bands:
        return "warning"
    return "stable"

def window_report(df: pd.DataFrame, probs: np.ndarray, ref: dict, importance: list, cfg: dict) -> dict:
    min_sample = cfg.get("drift", {}).get("min_sample", 1000)
    floor = cfg.get("drift", {}).get("smoothing_floor", 0.0001)
    structural = set(cfg.get("drift", {}).get("structural_features", ["property_valuation_method"]))
    top_n = cfg.get("drift", {}).get("top_features", 10)

    n = len(df)
    if n < min_sample:
        return {
            "status": "insufficient_data",
            "n": n,
            "triggers": [],
            "score_psi": 0.0,
            "features": [],
            "score_bins": None
        }

    # Score PSI
    score_spec = ref["score"]
    score_codes = assign_bins(pd.Series(probs), score_spec)
    cur_score_share = shares(score_codes, n_bins(score_spec))
    score_psi = calculate_psi(np.array(score_spec["reference_share"]), cur_score_share, floor)

    feats = []
    psi_map = {}
    model_features = cfg["features"]["numeric"] + cfg["features"]["categorical"]

    for f in model_features:
        if f not in ref:
            continue
        f_spec = ref[f]
        s_data = df[f] if f in df.columns else pd.Series([np.nan] * n)
        cur_codes = assign_bins(s_data, f_spec)
        cur_share = shares(cur_codes, n_bins(f_spec))
        v = calculate_psi(np.array(f_spec["reference_share"]), cur_share, floor)
        is_const = bool(f_spec["kind"] == "numeric" and len(f_spec["edges"]) == 0)
        if is_const:
            v = 0.0
        psi_map[f] = v
        feats.append({
            "name": f,
            "psi": round(v, 4),
            "band": band_of(v, cfg),
            "structural": bool(f in structural),
            "constant": is_const
        })

    # Sort feats by PSI descending
    feats.sort(key=lambda x: x["psi"], reverse=True)

    # Status rules (DR-08, DR-09):
    # Status is worst band among score PSI and PSI of 10 most important non-structural features
    top_non_struct = [f for f in importance if f in psi_map and f not in structural][:top_n]
    judged = [("score", score_psi)] + [(f, psi_map[f]) for f in top_non_struct]

    bands = [band_of(v, cfg) for _, v in judged]
    status = worst_band(bands)
    triggers = [name for name, v in judged if band_of(v, cfg) != "stable"]

    return {
        "status": status,
        "n": n,
        "triggers": triggers,
        "score_psi": round(score_psi, 4),
        "features": feats,
        "score_bins": {
            "edges": score_spec["edges"],
            "reference": [round(x, 4) for x in score_spec["reference_share"]],
            "current": [round(float(x), 4) for x in cur_score_share]
        }
    }
