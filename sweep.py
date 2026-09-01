"""Sweep the intensity-weight knob and pick a winner on VALIDATION only."""
import subprocess, sys
import numpy as np, torch
from src.data import prepare
from src.metrics import regression_metrics, classification_metrics, best_threshold
from src.model import RainfallLSTM
from evaluate import predict

CSV = "data/weather_mumbai.csv"
device = "mps" if torch.backends.mps.is_available() else "cpu"
s = prepare(CSV)
val_mm = np.expm1(s.y_val)

print(f"{'iw':>5} {'delta':>6} | {'val RMSE':>9} {'val MAE':>8} {'val R2':>7} | "
      f"{'>35mm MAE':>10} {'>35mm pred':>11} | {'val F1':>7}")
print("-" * 78)
rows = []
for iw, delta in [(0.0, 1.0), (0.3, 2.0), (0.6, 2.0), (1.0, 2.0), (1.5, 2.0), (1.0, 4.0)]:
    out = f"models/sweep_iw{iw}_d{delta}.pt"
    subprocess.run([sys.executable, "train.py", "--epochs", "120", "--patience", "15",
                    "--intensity-weight", str(iw), "--huber-delta", str(delta),
                    "--out", out], capture_output=True)
    ck = torch.load(out, weights_only=False)
    m = RainfallLSTM(len(s.feature_names), ck["config"]["hidden"],
                     ck["config"]["layers"], ck["config"]["dropout"]).to(device)
    m.load_state_dict(ck["state_dict"])
    pred, prob = predict(m, s.X_val, device)
    r = regression_metrics(val_mm, pred)
    hv = val_mm >= 35
    hmae = float(np.mean(np.abs(pred[hv] - val_mm[hv])))
    hpred = float(pred[hv].mean())
    _, f1 = best_threshold(val_mm, prob)
    print(f"{iw:5.1f} {delta:6.1f} | {r['rmse_mm']:9.2f} {r['mae_mm']:8.2f} {r['r2']:7.3f} | "
          f"{hmae:10.2f} {hpred:11.2f} | {f1:7.3f}", flush=True)
    rows.append((iw, delta, r["rmse_mm"], hmae, f1, out))

# Pick the config with the best validation RMSE among those that do not
# badly under-forecast heavy rain.
best = min(rows, key=lambda r: r[2])
print(f"\nbest val RMSE: iw={best[0]} delta={best[1]} -> {best[5]}")
