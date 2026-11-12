import numpy as np
import pandas as pd
import lightgbm as lgb
from src.common.config import load_config
from src.common.features import apply_categories, feature_matrix
import json

def test_ts10_reproducibility():
    """TS-10: Verify training determinism with fixed seed."""
    cfg = load_config()
    seed = cfg.get("seed", 42)
    
    # Load sample from processed train
    cats = json.load(open("artifacts/categories.json"))
    train_df = pd.read_parquet("data/processed/train.parquet").head(1000)
    train_df = apply_categories(train_df, cats)
    
    X = feature_matrix(train_df)
    y = train_df["default_24m"].to_numpy(dtype=int)
    
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "deterministic": True,
        "verbosity": -1,
        "n_jobs": 1
    }
    
    dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
    
    m1 = lgb.train(params, dtrain, num_boost_round=10)
    p1 = m1.predict(X)
    
    m2 = lgb.train(params, dtrain, num_boost_round=10)
    p2 = m2.predict(X)
    
    # Must produce identical probabilities
    assert np.allclose(p1, p2, atol=1e-10)
