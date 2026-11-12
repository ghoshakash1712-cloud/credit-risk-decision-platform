import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import sklearn
from src.common.config import load_config, project_path
from src.common.features import feature_matrix, apply_categories

def now_utc():
    return datetime.now(timezone.utc).isoformat()

def short_hash(d: dict) -> str:
    s = json.dumps(d, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]

def main():
    cfg = load_config()
    processed = project_path(cfg, "processed")
    artifacts = project_path(cfg, "artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)

    with open(artifacts / "categories.json", "r", encoding="utf-8") as f:
        categories = json.load(f)

    print("Loading processed train and valid splits...")
    train_df = apply_categories(pd.read_parquet(processed / "train.parquet"), categories)
    valid_df = apply_categories(pd.read_parquet(processed / "valid.parquet"), categories)

    Xtr = feature_matrix(train_df, cfg)
    ytr = train_df["default_24m"].values
    Xva = feature_matrix(valid_df, cfg)
    yva = valid_df["default_24m"].values

    seed = int(cfg.get("seed", 42))
    model_cfg = cfg.get("model", {})
    n_iter = int(model_cfg.get("tuning_iterations", 20))
    cv_folds = int(model_cfg.get("cv_folds", 5))
    cv_n_estimators = int(model_cfg.get("cv_n_estimators", 300))
    max_estimators = int(model_cfg.get("max_estimators", 2000))
    early_stopping_rounds = int(model_cfg.get("early_stopping_rounds", 50))

    print(f"Running randomized hyperparameter search ({n_iter} iterations, {cv_folds}-fold CV on train)...")
    base = lgb.LGBMClassifier(
        n_estimators=cv_n_estimators,
        subsample_freq=1,
        random_state=seed,
        verbosity=-1,
        n_jobs=-1
    )

    space = {
        "num_leaves": randint(15, 127),
        "max_depth": [-1, 4, 6, 8],
        "learning_rate": loguniform(0.02, 0.1),
        "min_child_samples": randint(20, 200),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "reg_lambda": loguniform(1e-3, 10.0)
    }

    # MD-03: CV inside train split only. To make CV fast, sample 30,000 train rows for CV search if train is large
    if len(Xtr) > 30000:
        cv_sample_idx = np.random.default_rng(seed).choice(len(Xtr), 30000, replace=False)
        X_cv, y_cv = Xtr.iloc[cv_sample_idx], ytr[cv_sample_idx]
    else:
        X_cv, y_cv = Xtr, ytr

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        base,
        space,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        refit=False,
        random_state=seed,
        n_jobs=-1
    )
    search.fit(X_cv, y_cv)

    best_params = search.best_params_
    best_score = float(search.best_score_)
    print(f"Best CV ROC AUC: {best_score:.4f} with params: {best_params}")

    # MD-03: Refit on full training split with early stopping on validation split
    print(f"Refitting final model on full train split ({len(Xtr)} loans) with early stopping on valid...")
    final = lgb.LGBMClassifier(
        **best_params,
        n_estimators=max_estimators,
        subsample_freq=1,
        random_state=seed,
        verbosity=-1,
        n_jobs=-1
    )

    final.fit(
        Xtr,
        ytr,
        eval_set=[(Xva, yva)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
    )

    best_iter = int(final.best_iteration_)
    print(f"Training completed. Best iteration: {best_iter}")

    # MD-01, DT-08: Save in LightGBM native text format (never pickle)
    model_path = artifacts / "model.txt"
    final.booster_.save_model(str(model_path))
    print(f"Model saved in native text format to {model_path}")

    # MD-04: Reproducibility and metadata
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    model_version = f"{today_str}-{short_hash(best_params)}"

    metadata_path = artifacts / "metadata.json"
    meta = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    meta.update({
        "seed": seed,
        "model_version": model_version,
        "best_cv_auc": round(best_score, 4),
        "best_params": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in best_params.items()},
        "best_iteration": best_iter,
        "library_versions": {
            "lightgbm": lgb.__version__,
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__
        },
        "trained_at": now_utc(),
        "category_levels": categories
    })

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Metadata updated at {metadata_path} with model_version: {model_version}")

if __name__ == "__main__":
    main()
