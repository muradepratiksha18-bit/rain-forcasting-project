"""Evaluate a trained city model on its held-out test years, against baselines.

Usage:
    python evaluate.py                    # mumbai
    python evaluate.py --city chennai
"""
import argparse
import json

import numpy as np
import torch

from src import calibration
from src.cities import data_path, metrics_path, model_path, predictions_path
from src.data import RAIN_THRESHOLD_MM, prepare
from src.metrics import best_threshold, classification_metrics, regression_metrics
from src.model import RainfallLSTM

BANDS = [(0, 1, "dry (<1mm)"), (1, 10, "light 1-10"), (10, 35, "moderate 10-35"),
         (35, 65, "heavy 35-65"), (65, 1e9, "very heavy >65")]


def predict(model, X, device, batch=512):
    """Return (rainfall in mm, probability of rain)."""
    model.eval()
    reg, clf = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            r, c = model(xb)
            reg.append(r.cpu().numpy())
            clf.append(torch.sigmoid(c).cpu().numpy())
    # Undo the log1p transform; clip because negative rainfall is meaningless.
    return np.clip(np.expm1(np.concatenate(reg)), 0, None), np.concatenate(clf)


def baselines(splits):
    """Two references any useful model must beat."""
    # Persistence: "tomorrow looks like today". Deceptively strong for rainfall.
    idx = splits.feature_names.index("precipitation_sum")
    persist = splits.X_test[:, -1, idx] * splits.sd[idx] + splits.mu[idx]
    persist = np.clip(persist, 0, None)

    # Climatology: the long-run training mean.
    clim = np.full(len(splits.y_test), float(np.expm1(splits.y_train).mean()))
    return persist, clim


def evaluate_city(city, model_file=None, csv_file=None):
    """Run the full evaluation and return a results dict (no printing)."""
    ckpt = torch.load(model_file or model_path(city), map_location="cpu",
                      weights_only=False)
    cfg = ckpt["config"]
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")

    s = prepare(csv_file or data_path(city), seq_len=cfg["seq_len"],
                horizon=cfg["horizon"])
    model = RainfallLSTM(len(s.feature_names), cfg["hidden"], cfg["layers"],
                         cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])

    true_mm = np.expm1(s.y_test)
    raw_mm, prob = predict(model, s.X_test, device)
    coef = ckpt.get("calibration")
    pred_mm = calibration.apply(raw_mm, coef)

    # Choose the rain/no-rain cutoff on validation, then apply it to test -
    # tuning it on test would be scoring yourself on your own answer key.
    val_raw, val_prob = predict(model, s.X_val, device)
    thr, val_f1 = best_threshold(np.expm1(s.y_val), val_prob)

    persist, clim = baselines(s)
    bands = []
    for lo, hi, label in BANDS:
        m = (true_mm >= lo) & (true_mm < hi)
        if m.sum() == 0:
            continue
        bands.append({"band": label, "n": int(m.sum()),
                      "mean_observed_mm": float(true_mm[m].mean()),
                      "mean_predicted_mm": float(pred_mm[m].mean()),
                      "mae_mm": float(np.mean(np.abs(pred_mm[m] - true_mm[m])))})

    return {
        "city": city,
        "calibration": coef,
        "threshold": thr,
        "val_f1": val_f1,
        "test_start": str(s.dates_test[0])[:10],
        "test_end": str(s.dates_test[-1])[:10],
        "n_test_days": len(true_mm),
        "rain_rate": float((true_mm > RAIN_THRESHOLD_MM).mean()),
        "mean_annual_mm": float(true_mm.sum() / (len(true_mm) / 365.25)),
        "regression": {
            "lstm": regression_metrics(true_mm, pred_mm),
            "lstm_raw": regression_metrics(true_mm, raw_mm),
            "persistence": regression_metrics(true_mm, persist),
            "climatology": regression_metrics(true_mm, clim),
        },
        "classification": {
            "lstm": classification_metrics(true_mm, prob, threshold=thr),
            "persistence": classification_metrics(
                true_mm, (persist > RAIN_THRESHOLD_MM).astype(float), threshold=0.5),
        },
        "by_intensity": bands,
        "_arrays": {"dates": s.dates_test, "true_mm": true_mm,
                    "pred_mm": pred_mm, "prob": prob},
    }


def report(res):
    print(f"\n[{res['city']}]  test {res['test_start']} -> {res['test_end']}"
          f"  ({res['n_test_days']} days, {res['rain_rate']:.1%} rainy,"
          f" {res['mean_annual_mm']:.0f} mm/yr)")
    if res["calibration"]:
        a, b = res["calibration"]
        print(f"Calibration: log1p(y) = {a:.3f}*log1p(pred) {b:+.3f}  (fit on validation)")
    print(f"Decision threshold {res['threshold']:.2f} "
          f"(chosen on validation, F1={res['val_f1']:.3f})\n")

    r = res["regression"]
    print("RAINFALL AMOUNT (mm)        RMSE     MAE      R2    bias")
    for name, key in [("LSTM (calibrated)", "lstm"), ("LSTM (raw)", "lstm_raw"),
                      ("persistence", "persistence"), ("climatology", "climatology")]:
        m = r[key]
        print(f"  {name:<22} {m['rmse_mm']:7.2f} {m['mae_mm']:7.2f} "
              f"{m['r2']:7.3f} {m['bias_mm']:7.2f}")

    c = res["classification"]
    print("\nRAIN / NO-RAIN (>1mm)       Acc     Prec     Rec      F1     CSI     HSS     AUC")
    for name, key in [("LSTM", "lstm"), ("persistence", "persistence")]:
        m = c[key]
        print(f"  {name:<22} {m['accuracy']:6.3f} {m['precision']:7.3f} "
              f"{m['recall']:7.3f} {m['f1']:7.3f} {m['csi']:7.3f} "
              f"{m['hss']:7.3f} {m['roc_auc']:7.3f}")
    cm = c["lstm"]["confusion"]
    print(f"\n  confusion: TP={cm['tp']}  FP={cm['fp']}  FN={cm['fn']}  TN={cm['tn']}")

    print("\nERROR BY INTENSITY BAND        n     mean obs    mean pred     MAE")
    for b in res["by_intensity"]:
        print(f"  {b['band']:<24} {b['n']:5d} {b['mean_observed_mm']:10.2f} "
              f"{b['mean_predicted_mm']:12.2f} {b['mae_mm']:8.2f}")


def save(res, out=None):
    arrays = res.pop("_arrays")
    out = out or metrics_path(res["city"])
    json.dump(res, open(out, "w"), indent=2)
    np.savez(predictions_path(res["city"]), **arrays)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--city", default="mumbai")
    p.add_argument("--model", help="defaults to this city's checkpoint")
    p.add_argument("--csv", help="defaults to this city's CSV")
    p.add_argument("--out", help="defaults to this city's metrics file")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    res = evaluate_city(args.city, args.model, args.csv)
    if not args.quiet:
        report(res)
    out = save(res, args.out)
    if not args.quiet:
        print(f"\nwrote {out} and {predictions_path(args.city)}")


if __name__ == "__main__":
    main()
