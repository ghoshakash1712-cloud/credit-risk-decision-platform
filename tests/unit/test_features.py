from src.common.config import load_config, load_schema
from src.common.features import clean_origination, add_derived
import pandas as pd
import numpy as np

def test_leakage_control():
    """TS-04: The feature list contains no performance-file field (DP-09)."""
    cfg = load_config()
    perf_cols = set(load_schema("performance"))
    features = set(cfg["features"]["numeric"] + cfg["features"]["categorical"])

    # Ensure no intersection with performance columns
    intersection = features & perf_cols
    assert not intersection, f"Features contain performance-file fields: {intersection}"

def test_borrower_cap():
    """SRS 5.9: n_borrowers is capped at 2."""
    cfg = load_config()
    df = pd.DataFrame({
        "loan_id": ["L1", "L2", "L3"],
        "n_borrowers": ["1", "2", "4"]
    })
    cleaned = clean_origination(df, cfg)
    assert list(cleaned["n_borrowers"]) == [1.0, 2.0, 2.0]

def test_not_available_sentinels():
    """DP-05: Sentinel codes converted to NaN."""
    cfg = load_config()
    df = pd.DataFrame({
        "loan_id": ["L1", "L2"],
        "fico": ["9999", "720"],
        "occupancy": ["9", "P"],
        "orig_dti": ["999", "35"]
    })
    cleaned = clean_origination(df, cfg)
    assert np.isnan(cleaned["fico"].iloc[0])
    assert cleaned["fico"].iloc[1] == 720.0
    assert np.isnan(cleaned["occupancy"].iloc[0])
    assert cleaned["occupancy"].iloc[1] == "P"
    assert np.isnan(cleaned["orig_dti"].iloc[0])
    assert cleaned["orig_dti"].iloc[1] == 35.0
