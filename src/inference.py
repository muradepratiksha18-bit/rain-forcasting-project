"""Shared inference helpers used by both predict.py and the Streamlit app."""
import numpy as np
import torch

from src import calibration
from src.data import build_features, load_frame
from src.model import RainfallLSTM


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_bundle(model_path):
    """Load weights + the scaler and calibration saved alongside them."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = pick_device()
    model = RainfallLSTM(
        len(ckpt["feature_names"]), cfg["hidden"], cfg["layers"], cfg["dropout"]
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {
        "model": model, "config": cfg, "device": device,
        "features": ckpt["feature_names"], "mu": ckpt["mu"], "sd": ckpt["sd"],
        "calibration": ckpt.get("calibration"), "trained_epoch": ckpt.get("epoch"),
    }


def frame_with_features(csv_path):
    df, _ = build_features(load_frame(csv_path))
    return df


def _run(bundle, windows):
    x = torch.from_numpy(windows.astype(np.float32)).to(bundle["device"])
    with torch.no_grad():
        r, c = bundle["model"](x)
    raw = np.clip(np.expm1(r.cpu().numpy()), 0, None)
    return calibration.apply(raw, bundle["calibration"]), torch.sigmoid(c).cpu().numpy()


def forecast_latest(bundle, df):
    """Forecast the day after the last row of `df`. Returns (mm, prob, target_date)."""
    seq = bundle["config"]["seq_len"]
    if len(df) < seq:
        raise ValueError(f"need at least {seq} days of history, got {len(df)}")
    w = df[bundle["features"]].values[-seq:].astype(np.float64)
    scaled = ((w - bundle["mu"]) / bundle["sd"])[None]
    mm, prob = _run(bundle, scaled)
    target = df["date"].iloc[-1] + np.timedelta64(bundle["config"]["horizon"], "D")
    return float(mm[0]), float(prob[0]), target


def backtest(bundle, df):
    """Predict every day the history supports. Returns a DataFrame of results.

    Note these are in-sample for the training years - use the test-period rows
    (or evaluate.py) for an honest read on accuracy.
    """
    import pandas as pd

    seq, hz = bundle["config"]["seq_len"], bundle["config"]["horizon"]
    vals = ((df[bundle["features"]].values.astype(np.float64) - bundle["mu"])
            / bundle["sd"])
    n = len(vals) - seq - hz + 1
    if n <= 0:
        return pd.DataFrame(columns=["date", "actual_mm", "pred_mm", "prob"])

    windows = np.stack([vals[i:i + seq] for i in range(n)])
    mm, prob = [], []
    for i in range(0, n, 512):                     # batch to bound memory
        a, b = _run(bundle, windows[i:i + 512])
        mm.append(a); prob.append(b)

    idx = np.arange(n) + seq + hz - 1              # row being predicted
    return pd.DataFrame({
        "date": df["date"].values[idx],
        "actual_mm": df["precipitation_sum"].values[idx],
        "pred_mm": np.concatenate(mm),
        "prob": np.concatenate(prob),
    })
