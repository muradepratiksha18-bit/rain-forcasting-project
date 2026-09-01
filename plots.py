"""Diagnostic plots for the trained model (run evaluate.py first)."""
import json

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.cities import diagnostics_path, history_path, metrics_path, predictions_path

ap = argparse.ArgumentParser()
ap.add_argument("--city", default="mumbai")
CITY = ap.parse_args().city

d = np.load(predictions_path(CITY))
dates = d["dates"].astype("datetime64[D]")
true_mm, pred_mm, prob = d["true_mm"], d["pred_mm"], d["prob"]
hist = json.load(open(history_path(CITY)))

fig, ax = plt.subplots(3, 2, figsize=(15, 12))

# 1. Learning curves
ep = [h["epoch"] for h in hist]
ax[0, 0].plot(ep, [h["train_loss"] for h in hist], label="train")
ax[0, 0].plot(ep, [h["val_loss"] for h in hist], label="validation")
best = min(hist, key=lambda h: h["val_loss"])
ax[0, 0].axvline(best["epoch"], ls="--", c="grey", lw=1)
ax[0, 0].annotate(f"best (ep {best['epoch']})", (best["epoch"], best["val_loss"]),
                  textcoords="offset points", xytext=(8, 14), fontsize=8, color="grey")
ax[0, 0].set(title="Learning curves", xlabel="epoch", ylabel="multi-task loss")
ax[0, 0].legend()

# 2. Observed vs predicted over the most recent monsoon
yr = dates.astype("datetime64[Y]").astype(int) + 1970
m = yr == yr.max()
ax[0, 1].fill_between(dates[m], true_mm[m], alpha=.45, label="observed", color="#3b6ea5")
ax[0, 1].plot(dates[m], pred_mm[m], lw=1.2, color="#d1495b", label="predicted")
ax[0, 1].set(title=f"Daily rainfall, {yr.max()} - {CITY.title()} (unseen test year)", ylabel="mm")
ax[0, 1].legend()
ax[0, 1].tick_params(axis="x", rotation=30)

# 3. Scatter on a log scale: most days sit at zero, so linear axes hide everything
ax[1, 0].scatter(true_mm + .1, pred_mm + .1, s=7, alpha=.3, color="#3b6ea5")
lim = [.1, max(true_mm.max(), pred_mm.max()) * 1.2]
ax[1, 0].plot(lim, lim, "k--", lw=1, label="perfect")
ax[1, 0].set(xscale="log", yscale="log", xlim=lim, ylim=lim,
             title="Observed vs predicted (log scale)",
             xlabel="observed mm + 0.1", ylabel="predicted mm + 0.1")
ax[1, 0].legend()

# 4. Reliability of the rain probability: does "70% chance" rain 70% of the time?
obs = true_mm > 1.0
bins = np.linspace(0, 1, 11)
idx = np.digitize(prob, bins) - 1
xs, ys, ns = [], [], []
for b in range(10):
    s = idx == b
    if s.sum() >= 10:
        xs.append(prob[s].mean()); ys.append(obs[s].mean()); ns.append(int(s.sum()))
ax[1, 1].plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
ax[1, 1].plot(xs, ys, "o-", color="#d1495b", label="model")
for x, y, n in zip(xs, ys, ns):
    ax[1, 1].annotate(str(n), (x, y), textcoords="offset points", xytext=(4, -11), fontsize=7)
ax[1, 1].set(title="Reliability of rain probability (labels = n days)",
             xlabel="forecast probability", ylabel="observed rain frequency")
ax[1, 1].legend()

# 5. Mean predicted vs observed by intensity band - where the model falls short
mets = json.load(open(metrics_path(CITY)))
bands = mets["by_intensity"]
x = np.arange(len(bands)); w = .38
ax[2, 0].bar(x - w/2, [b["mean_observed_mm"] for b in bands], w, label="observed", color="#3b6ea5")
ax[2, 0].bar(x + w/2, [b["mean_predicted_mm"] for b in bands], w, label="predicted", color="#d1495b")
ax[2, 0].set_xticks(x)
ax[2, 0].set_xticklabels([f"{b['band']}\n(n={b['n']})" for b in bands], fontsize=7)
ax[2, 0].set(title="Mean rainfall by intensity band", ylabel="mm")
ax[2, 0].legend()

# 6. Monthly totals - does the model reproduce the monsoon cycle?
month = dates.astype("datetime64[M]").astype(int) % 12
to = [true_mm[month == k].sum() / 5 for k in range(12)]   # 5 test years
po = [pred_mm[month == k].sum() / 5 for k in range(12)]
ax[2, 1].bar(np.arange(12) - w/2, to, w, label="observed", color="#3b6ea5")
ax[2, 1].bar(np.arange(12) + w/2, po, w, label="predicted", color="#d1495b")
ax[2, 1].set_xticks(range(12))
ax[2, 1].set_xticklabels(list("JFMAMJJASOND"))
ax[2, 1].set(title="Mean monthly rainfall total (test years)", ylabel="mm/month")
ax[2, 1].legend()

fig.suptitle(f"{CITY.title()} - held-out test years", fontsize=13, y=1.002)
fig.tight_layout()
fig.savefig(diagnostics_path(CITY), dpi=130)
print(f"wrote {diagnostics_path(CITY)}")
