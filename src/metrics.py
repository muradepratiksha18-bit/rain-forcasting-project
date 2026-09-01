"""Evaluation metrics for rainfall forecasts, in millimetres and as rain/no-rain."""
import numpy as np

from .data import RAIN_THRESHOLD_MM


def regression_metrics(true_mm, pred_mm):
    err = pred_mm - true_mm
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true_mm - true_mm.mean()) ** 2))
    return {
        "rmse_mm": float(np.sqrt(np.mean(err ** 2))),
        "mae_mm": float(np.mean(np.abs(err))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "bias_mm": float(err.mean()),
    }


def classification_metrics(true_mm, prob, threshold=0.5):
    """Contingency-table scores for 'will it rain tomorrow'."""
    obs = true_mm > RAIN_THRESHOLD_MM
    pred = prob >= threshold

    tp = int(np.sum(pred & obs))
    fp = int(np.sum(pred & ~obs))
    fn = int(np.sum(~pred & obs))
    tn = int(np.sum(~pred & ~obs))

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    return {
        "accuracy": (tp + tn) / len(obs),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        # Critical Success Index and Heidke Skill Score are the standard
        # meteorological scores; HSS is skill relative to random chance.
        "csi": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "hss": _hss(tp, fp, fn, tn),
        "roc_auc": roc_auc(obs, prob),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def _hss(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    exp = ((tp + fn) * (tp + fp) + (tn + fn) * (tn + fp)) / n
    denom = n - exp
    return (tp + tn - exp) / denom if denom else 0.0


def roc_auc(obs, score):
    """Rank-based AUC (equivalent to the Mann-Whitney U statistic)."""
    obs = np.asarray(obs, dtype=bool)
    n_pos, n_neg = obs.sum(), (~obs).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # Average ranks within ties so identical scores don't bias the statistic.
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return float((ranks[obs].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def best_threshold(true_mm, prob):
    """Pick the decision threshold that maximises F1 on the given set."""
    obs = true_mm > RAIN_THRESHOLD_MM
    grid = np.linspace(0.05, 0.95, 91)
    scores = []
    for t in grid:
        pred = prob >= t
        tp = np.sum(pred & obs); fp = np.sum(pred & ~obs); fn = np.sum(~pred & obs)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(grid[int(np.argmax(scores))]), float(np.max(scores))
