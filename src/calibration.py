"""Post-hoc variance correction for the rainfall head.

A model trained on a squared/Huber loss predicts a conditional mean, which is
always less variable than the observations - so forecasts get compressed toward
the middle and heavy-rain days come out far too low. The standard remedy is to
rescale predictions with a linear fit in the transformed space:

    log1p(mm_corrected) = a * log1p(mm_raw) + b

`a` comes out above 1, which stretches the forecast distribution back out.

The fit uses VALIDATION data only. Fitting it on test would be tuning on the
answer key, and fitting it on train would inherit the training-set overfit.
"""
import numpy as np


def fit(pred_mm_val, true_mm_val):
    """Least-squares fit of the correction on validation predictions."""
    x = np.log1p(np.clip(pred_mm_val, 0, None))
    y = np.log1p(np.clip(true_mm_val, 0, None))
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(a), float(b)


def apply(pred_mm, coef):
    if coef is None:
        return pred_mm
    a, b = coef
    return np.clip(np.expm1(a * np.log1p(np.clip(pred_mm, 0, None)) + b), 0, None)
