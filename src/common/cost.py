from dataclasses import dataclass
import numpy as np

class NotComputableError(Exception):
    """Raised when a split has no defaulting loans or no good loans."""
    pass

@dataclass
class Prepared:
    k: np.ndarray
    threshold: np.ndarray
    cum_tp: np.ndarray
    cum_fp: np.ndarray
    total_default_upb: float
    total_good_upb: float
    n_defaults: int
    n_good: int
    prob: np.ndarray
    y: np.ndarray
    upb: np.ndarray

@dataclass
class Optimum:
    threshold: float
    reject_none: bool
    net_benefit: float
    best_index: int
    curve: list

def prepare(prob: np.ndarray, y: np.ndarray, upb: np.ndarray) -> Prepared:
    prob = np.asarray(prob, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    upb = np.asarray(upb, dtype=np.float64)

    order = np.argsort(-prob, kind="stable")
    p = prob[order]
    yy = y[order]
    b = upb[order]

    cum_tp = np.cumsum(b * yy)
    cum_fp = np.cumsum(b * (1 - yy))

    if len(p) == 0:
        ends = np.array([], dtype=bool)
        idx = np.array([], dtype=int)
    else:
        ends = np.concatenate([p[:-1] != p[1:], [True]])
        idx = np.flatnonzero(ends)

    # Candidate 0 = reject nothing (threshold 1.0)
    # Candidate i+1 = reject everything down to the end of group i
    k = np.concatenate([[0], idx + 1]) if len(idx) > 0 else np.array([0])
    threshold = np.concatenate([[1.0], p[idx]]) if len(idx) > 0 else np.array([1.0])
    c_tp = np.concatenate([[0.0], cum_tp[idx]]) if len(idx) > 0 else np.array([0.0])
    c_fp = np.concatenate([[0.0], cum_fp[idx]]) if len(idx) > 0 else np.array([0.0])

    return Prepared(
        k=k,
        threshold=threshold,
        cum_tp=c_tp,
        cum_fp=c_fp,
        total_default_upb=float((b * yy).sum()),
        total_good_upb=float((b * (1 - yy)).sum()),
        n_defaults=int(yy.sum()),
        n_good=int((1 - yy).sum()),
        prob=prob,
        y=y,
        upb=upb
    )

def optimize(prep: Prepared, g: float, L: float, curve_n: int = 200) -> Optimum:
    if prep.n_defaults == 0 or prep.n_good == 0:
        raise NotComputableError("Split has no defaults or no good loans")

    net_benefit = L * prep.cum_tp - g * prep.cum_fp
    best_i = int(np.argmax(net_benefit)) # first max = fewest rejections (OP-08)
    reject_none = (best_i == 0)
    best_threshold = float(prep.threshold[best_i])
    curve = curve_points(prep, net_benefit, best_i, n=curve_n)

    return Optimum(
        threshold=best_threshold,
        reject_none=reject_none,
        net_benefit=float(net_benefit[best_i]),
        best_index=best_i,
        curve=curve
    )

def safe_div(num, denom):
    if denom == 0 or np.isnan(denom):
        return None
    return float(num / denom)

def metrics_at(prep: Prepared, t: float, g: float, L: float) -> dict:
    prob = prep.prob
    y = prep.y
    upb = prep.upb
    n = len(prob)

    reject = prob >= t
    tp = reject & (y == 1)
    fp = reject & (y == 0)
    fn = (~reject) & (y == 1)
    tn = (~reject) & (y == 0)

    tp_count = int(np.sum(tp))
    fp_count = int(np.sum(fp))
    tn_count = int(np.sum(tn))
    fn_count = int(np.sum(fn))

    b_tp = float(np.sum(upb[tp]))
    b_fp = float(np.sum(upb[fp]))
    b_fn = float(np.sum(upb[fn]))
    b_tn = float(np.sum(upb[tn]))

    profit = float(g * prep.total_good_upb - g * b_fp - L * b_fn)
    approve_all = float(g * prep.total_good_upb - L * prep.total_default_upb)
    net_benefit = float(L * b_tp - g * b_fp)

    return {
        "tp": tp_count,
        "fp": fp_count,
        "tn": tn_count,
        "fn": fn_count,
        "precision": safe_div(tp_count, tp_count + fp_count),
        "recall": safe_div(tp_count, tp_count + fn_count),
        "rejection_rate": float(np.mean(reject)) if n > 0 else 0.0,
        "approved_default_rate": safe_div(fn_count, fn_count + tn_count),
        "profit": profit,
        "approve_all_profit": approve_all,
        "net_benefit": net_benefit,
        "net_benefit_per_loan": float(net_benefit / n) if n > 0 else 0.0
    }

def analytic_threshold(g: float, L: float) -> float:
    return float(g / (g + L))

def calibration_warning(swept: float, analytic: float, factor: float = 2.0) -> bool:
    if analytic <= 0:
        return False
    return bool(swept > factor * analytic or swept < analytic / factor)

def curve_points(prep: Prepared, net_benefit: np.ndarray, best_i: int, n: int = 200) -> list:
    total_pts = len(net_benefit)
    if total_pts <= n:
        indices = np.arange(total_pts)
    else:
        sample_idx = np.linspace(0, total_pts - 1, n).astype(int)
        closest = np.argmin(np.abs(sample_idx - best_i))
        sample_idx[closest] = best_i
        indices = np.unique(sample_idx)
        indices.sort()

    return [{"threshold": round(float(prep.threshold[i]), 5),
             "net_benefit": round(float(net_benefit[i]), 2)} for i in indices]
