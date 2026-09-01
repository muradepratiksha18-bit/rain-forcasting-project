"""Train the rainfall LSTM.

Usage:
    python train.py
    python train.py --csv data/weather_pune.csv --epochs 120 --seq-len 45
"""
import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src import calibration
from src.cities import data_path, history_path, model_path
from src.data import prepare
from src.model import MultiTaskLoss, RainfallLSTM


def loaders(s, batch_size):
    def make(X, y, c, shuffle):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(c))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return (
        make(s.X_train, s.y_train, s.c_train, True),
        make(s.X_val, s.y_val, s.c_val, False),
    )


def run_epoch(model, loader, lossfn, device, opt=None):
    train = opt is not None
    model.train(train)
    total = n = 0.0
    with torch.set_grad_enabled(train):
        for xb, yb, cb in loader:
            xb, yb, cb = xb.to(device), yb.to(device), cb.to(device)
            pr, pc = model(xb)
            loss, _, _ = lossfn(pr, pc, yb, cb)
            if train:
                opt.zero_grad()
                loss.backward()
                # Recurrent nets on 30-step windows can spike; clip to stay stable.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * len(xb)
            n += len(xb)
    return total / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--city", default="mumbai")
    p.add_argument("--csv", help="defaults to this city's CSV")
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--horizon", type=int, default=1, help="days ahead to predict")
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--clf-weight", type=float, default=0.5)
    p.add_argument("--intensity-weight", type=float, default=0.6,
                   help="how strongly to up-weight heavy-rain days in the loss")
    p.add_argument("--huber-delta", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", help="defaults to this city's checkpoint")
    args = p.parse_args()

    # Each city trains its own model, so paths are derived from --city
    # unless the caller overrides them explicitly.
    args.csv = args.csv or data_path(args.city)
    args.out = args.out or model_path(args.city)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")

    s = prepare(args.csv, seq_len=args.seq_len, horizon=args.horizon)
    print(f"[{args.city}] device={device}  features={len(s.feature_names)}  "
          f"train={len(s.X_train)} val={len(s.X_val)} test={len(s.X_test)}")

    tl, vl = loaders(s, args.batch_size)
    model = RainfallLSTM(len(s.feature_names), args.hidden, args.layers, args.dropout).to(device)

    # Re-weight the positive class by the dry:wet ratio so the classifier
    # can't win by always predicting "dry".
    p_rain = float(s.c_train.mean())
    lossfn = MultiTaskLoss(
        pos_weight=(1 - p_rain) / p_rain, clf_weight=args.clf_weight,
        intensity_weight=args.intensity_weight, delta=args.huber_delta,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)

    best, best_epoch, history = float("inf"), -1, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr = run_epoch(model, tl, lossfn, device, opt)
        va = run_epoch(model, vl, lossfn, device)
        sched.step(va)
        history.append({"epoch": ep, "train_loss": tr, "val_loss": va,
                        "lr": opt.param_groups[0]["lr"]})

        flag = ""
        if va < best - 1e-5:
            best, best_epoch, flag = va, ep, "  *"
            torch.save({
                "state_dict": model.state_dict(),
                "config": vars(args),
                "feature_names": s.feature_names,
                "mu": s.mu, "sd": s.sd,
                "val_loss": va, "epoch": ep,
            }, args.out)
        if ep % 5 == 0 or flag:
            print(f"epoch {ep:3d}  train {tr:.4f}  val {va:.4f}{flag}", flush=True)

        if ep - best_epoch >= args.patience:
            print(f"early stop: no val improvement for {args.patience} epochs")
            break

    # Fit the variance correction on validation using the best checkpoint,
    # then store it alongside the weights so evaluate/predict apply it for free.
    ckpt = torch.load(args.out, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    from evaluate import predict as _predict
    val_pred, _ = _predict(model, s.X_val, device)
    coef = calibration.fit(val_pred, np.expm1(s.y_val))
    ckpt["calibration"] = coef
    ckpt["city"] = args.city          # so downstream code never has to guess
    torch.save(ckpt, args.out)
    print(f"calibration fit on validation: log1p(y) = {coef[0]:.3f}*log1p(pred) + {coef[1]:.3f}")

    json.dump(history, open(history_path(args.city), "w"), indent=2)
    print(f"\nbest val loss {best:.4f} at epoch {best_epoch}  "
          f"({time.time() - t0:.0f}s)  -> {args.out}")


if __name__ == "__main__":
    main()
