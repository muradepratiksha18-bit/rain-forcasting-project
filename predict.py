"""Forecast tomorrow's rainfall from the most recent observations.

Usage:
    python predict.py                       # uses the tail of the training CSV
    python predict.py --live --city mumbai  # fetch the last 60 days fresh
"""
import argparse
import subprocess
import sys

import numpy as np
import torch

from src import calibration
from src.cities import data_path, model_path
from src.data import build_features, load_frame
from src.model import RainfallLSTM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--city", default="mumbai")
    p.add_argument("--model", help="defaults to this city's checkpoint")
    p.add_argument("--csv", help="defaults to this city's CSV")
    p.add_argument("--live", action="store_true",
                   help="fetch real-time data (up to today) instead of the CSV")
    args = p.parse_args()

    args.model = args.model or model_path(args.city)
    args.csv = args.csv or data_path(args.city)

    if args.live:
        # The archive lags ~5 days, so real-time forecasting uses Open-Meteo's
        # forecast endpoint instead - separate service, separate rate limit.
        from src.data import clean_frame
        from src.live import fetch_live
        live_df = clean_frame(fetch_live(args.city, past_days=92))
        df, _ = build_features(live_df)
    else:
        df, _ = build_features(load_frame(args.csv))

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    cfg, feats = ckpt["config"], ckpt["feature_names"]
    mu, sd = ckpt["mu"], ckpt["sd"]
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")

    seq = cfg["seq_len"]
    if len(df) < seq:
        raise SystemExit(f"need at least {seq} days of history, got {len(df)}")

    window = df[feats].values[-seq:].astype(np.float64)
    x = torch.from_numpy(((window - mu) / sd).astype(np.float32)[None]).to(device)

    model = RainfallLSTM(len(feats), cfg["hidden"], cfg["layers"], cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        r, c = model(x)
    raw = np.clip(np.expm1(r.cpu().numpy()[0]), 0, None)
    mm = float(calibration.apply(np.array([raw]), ckpt.get("calibration"))[0])
    prob = float(torch.sigmoid(c).cpu().numpy()[0])

    last = df["date"].iloc[-1]
    target = last + np.timedelta64(cfg["horizon"], "D")
    label = ("no rain" if prob < .35 else "possible rain" if prob < .65
             else "rain likely" if prob < .85 else "rain very likely")

    source = "live (Open-Meteo forecast API)" if args.live else args.csv
    print(f"\n{args.city.title()}   [{source}]")
    print(f"History through : {last.date()}  ({seq}-day window)")
    print(f"Forecast for    : {target.date()}")
    print(f"  rain probability : {prob:5.1%}   -> {label}")
    print(f"  expected amount  : {mm:5.1f} mm")
    print("\n  Note: the model under-forecasts extreme days; treat any prediction")
    print("  above ~20 mm as 'heavy rain likely', not a precise total.\n")


if __name__ == "__main__":
    main()
