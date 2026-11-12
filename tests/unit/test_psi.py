import numpy as np
import pandas as pd
from src.common.psi import calculate_psi, build_spec, assign_bins, shares, n_bins

def test_psi_identical_data_is_zero():
    """TS-05: Identical data gives PSI of 0."""
    s = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    spec = build_spec(s, "numeric")
    ref_share = np.array(spec["reference_share"])
    val = calculate_psi(ref_share, ref_share)
    assert np.isclose(val, 0.0, atol=1e-6)

def test_psi_handworked_example():
    """TS-05: Hand-worked two-bin example matches."""
    # Two bins: ref = [0.5, 0.5], cur = [0.4, 0.6]
    # expected = (0.4 - 0.5) * ln(0.4 / 0.5) + (0.6 - 0.5) * ln(0.6 / 0.5)
    ref = np.array([0.5, 0.5])
    cur = np.array([0.4, 0.6])
    expected = (0.4 - 0.5) * np.log(0.4 / 0.5) + (0.6 - 0.5) * np.log(0.6 / 0.5)
    assert np.isclose(calculate_psi(ref, cur), expected)

def test_psi_empty_bin_stays_finite():
    """TS-05: An empty reference or current bin does not produce infinity."""
    ref = np.array([1.0, 0.0])
    cur = np.array([0.0, 1.0])
    val = calculate_psi(ref, cur, floor=0.0001)
    assert np.isfinite(val)
    assert val > 0
