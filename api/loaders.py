from pathlib import Path
import json
from types import SimpleNamespace
import pandas as pd
import lightgbm as lgb
from src.common.config import load_config, get_repo_root
from src.common.features import apply_categories, feature_matrix
from src.common import cost
from src.common.groups import encode_groups

REPORT_FILES = {
    "model_report": "artifacts/model_report.json",
    "importance": "artifacts/global_importance.json",
    "fairness_baseline": "artifacts/fairness_baseline.json",
    "psi_report": "artifacts/psi_report.json",
    "psi_simulated": "artifacts/psi_simulated.json"
}

class Store:
    def __init__(self, cfg=None, repo_root=None):
        self.root = Path(repo_root) if repo_root else get_repo_root()
        self.cfg = cfg or load_config()
        
        meta_file = self.root / "artifacts" / "metadata.json"
        if not meta_file.exists():
            raise SystemExit("artifacts_unavailable: metadata.json not found")
        self.meta = self._read_json(meta_file)
        
        model_file = self.root / "artifacts" / "model.txt"
        if not model_file.exists():
            raise SystemExit("artifacts_unavailable: model.txt not found")
        try:
            self.booster = lgb.Booster(model_file=str(model_file))
        except Exception as e:
            raise SystemExit(f"artifact mismatch: failed to load model.txt: {e}")
            
        cats_file = self.root / "artifacts" / "categories.json"
        if not cats_file.exists():
            raise SystemExit("artifacts_unavailable: categories.json not found")
        cats = self._read_json(cats_file)
        self.cats = cats

        holdout_file = self.root / "artifacts" / "holdout_predictions.parquet"
        if not holdout_file.exists():
            raise SystemExit("artifacts_unavailable: holdout_predictions.parquet not found")
        
        raw_h = pd.read_parquet(holdout_file)
        h = apply_categories(raw_h, cats).set_index("loan_id", drop=False)
        self.holdout = h
        self.features = feature_matrix(h)
        
        self.splits = {}
        for name in ["valid", "oot", "stress"]:
            s = h[h["split"] == name]
            prob = s["probability"].to_numpy(dtype=float)
            y = s["default_24m"].to_numpy(dtype=int)
            upb = s["orig_upb"].to_numpy(dtype=float)
            prep = cost.prepare(prob, y, upb)
            self.splits[name] = SimpleNamespace(
                df=s,
                prep=prep,
                prob=prob,
                y=y,
                upb=upb,
                groups=encode_groups(s)
            )

        self.reports = {}
        for k, rel in REPORT_FILES.items():
            path = self.root / rel
            if not path.exists():
                raise SystemExit(f"artifacts_unavailable: {rel} not found")
            self.reports[k] = self._read_json(path)
            
        self.self_check()

    def _read_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def self_check(self):
        n_rows = self.cfg.get("api", {}).get("self_check_rows", 100)
        tol = float(self.cfg.get("api", {}).get("self_check_tolerance", 1e-4))
        seed = int(self.cfg.get("seed", 42))
        
        s = self.holdout.sample(n=min(n_rows, len(self.holdout)), random_state=seed)
        X_sub = self.features.loc[s.index]
        try:
            live = self.booster.predict(X_sub)
        except Exception as e:
            raise SystemExit(f"artifact mismatch: booster.predict failed: {e}")
            
        diff = abs(live - s["probability"].values).max()
        if diff > tol:
            raise SystemExit(f"artifact mismatch: model.txt does not reproduce stored probabilities (max_diff={diff:.6f} > {tol})")
