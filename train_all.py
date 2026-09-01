"""Train and evaluate one model per city, then print a cross-city comparison.

Usage:
    python train_all.py                     # every city that has data
    python train_all.py --cities mumbai pune
    python train_all.py --force             # retrain even if a model exists
"""
import argparse
import json
import os
import subprocess
import sys
import time

from evaluate import evaluate_city, save
from src.cities import CITY_NAMES, data_path, model_path

SUMMARY = "outputs/summary.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cities", nargs="*", default=CITY_NAMES)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--force", action="store_true",
                   help="retrain cities that already have a checkpoint")
    args = p.parse_args()

    have = [c for c in args.cities if os.path.exists(data_path(c))]
    missing = [c for c in args.cities if c not in have]
    if missing:
        print(f"no data for {', '.join(missing)} - run "
              f"`python download_data.py --all` first\n")
    if not have:
        raise SystemExit("nothing to train")

    results = {}
    for city in have:
        if os.path.exists(model_path(city)) and not args.force:
            print(f"=== {city}: checkpoint exists, skipping training "
                  f"(--force to retrain) ===", flush=True)
        else:
            print(f"=== {city}: training ===", flush=True)
            t0 = time.time()
            r = subprocess.run(
                [sys.executable, "train.py", "--city", city,
                 "--epochs", str(args.epochs), "--patience", str(args.patience)],
                capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-2000:] or r.stderr[-2000:])
                raise SystemExit(f"training failed for {city}")
            print(f"    {r.stdout.strip().splitlines()[-1]}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)

        res = evaluate_city(city)
        results[city] = {k: v for k, v in res.items() if k != "_arrays"}
        save(res)

    json.dump(results, open(SUMMARY, "w"), indent=2)

    # A model that cannot beat persistence has learned nothing useful, so the
    # comparison is shown per city rather than hidden behind an average.
    print(f"\n{'city':<11} {'mm/yr':>6} {'rainy':>6} | {'R2':>6} {'R2 pers':>8} "
          f"| {'F1':>6} {'F1 pers':>8} {'AUC':>6} | beats?")
    print("-" * 76)
    for city, r in results.items():
        rl, rp = r["regression"]["lstm"], r["regression"]["persistence"]
        cl, cp = r["classification"]["lstm"], r["classification"]["persistence"]
        win = "yes" if rl["r2"] > rp["r2"] and cl["f1"] > cp["f1"] else "partial"
        print(f"{city:<11} {r['mean_annual_mm']:6.0f} {r['rain_rate']:6.1%} | "
              f"{rl['r2']:6.3f} {rp['r2']:8.3f} | {cl['f1']:6.3f} "
              f"{cp['f1']:8.3f} {cl['roc_auc']:6.3f} | {win}")
    print(f"\nwrote {SUMMARY}")


if __name__ == "__main__":
    main()
