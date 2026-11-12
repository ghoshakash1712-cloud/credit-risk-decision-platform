import numpy as np
import pytest
from src.common import cost

def test_sweep_matches_bruteforce():
    """TS-01: The sweep returns the same threshold and net benefit as brute-force search."""
    rng = np.random.default_rng(0)
    p = rng.choice(rng.random(200), 1000) # repeated values force ties
    y = rng.integers(0, 2, 1000)
    b = rng.integers(50_000, 600_000, 1000)

    prep = cost.prepare(p, y, b)

    for g, L in [(0.02, 0.25), (0.10, 0.05), (0.005, 0.60)]:
        # Brute force search over every unique threshold, plus reject nothing (t > max(p))
        unique_p = np.unique(p)
        candidates = [1.0] + list(unique_p)
        best_brute_nb = -float("inf")
        best_brute_t = 1.0

        for t in candidates:
            reject = p >= t
            nb = L * b[reject & (y == 1)].sum() - g * b[reject & (y == 0)].sum()
            # If tie, fewest rejected (which means higher threshold)
            if nb > best_brute_nb or (np.isclose(nb, best_brute_nb) and reject.sum() < (p >= best_brute_t).sum()):
                best_brute_nb = nb
                best_brute_t = t

        opt = cost.optimize(prep, g, L)
        assert np.isclose(opt.net_benefit, best_brute_nb, rtol=1e-5, atol=1e-2)

def test_cost_identity():
    """TS-02: Net benefit equals profit minus approve_all_profit."""
    rng = np.random.default_rng(42)
    p = rng.random(500)
    y = rng.integers(0, 2, 500)
    b = rng.integers(50_000, 500_000, 500)

    prep = cost.prepare(p, y, b)
    g, L = 0.02, 0.25

    for t in [0.05, 0.1, 0.2, 0.5, 0.9]:
        m = cost.metrics_at(prep, t, g, L)
        assert np.isclose(m["net_benefit"], m["profit"] - m["approve_all_profit"])

def test_analytic_threshold_and_calibration():
    """OP-09: Analytic threshold and calibration warning check."""
    g, L = 0.02, 0.25
    analytic = cost.analytic_threshold(g, L)
    assert np.isclose(analytic, 0.02 / 0.27)
    assert not cost.calibration_warning(analytic, analytic, factor=2.0)
    assert cost.calibration_warning(analytic * 2.1, analytic, factor=2.0)
    assert cost.calibration_warning(analytic * 0.4, analytic, factor=2.0)
