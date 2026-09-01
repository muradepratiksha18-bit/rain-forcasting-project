"""Download daily historical weather from the Open-Meteo ERA5 archive.

Free, no API key. Produces data/weather_<city>.csv with one row per day.

The public API is rate limited and bills by request *weight*, so a multi-year
hourly request is expensive. This script therefore fetches in small blocks,
caches every block on disk, and backs off politely on HTTP 429. A run that gets
rate limited can simply be re-run: cached blocks are reused and only the missing
ones are re-requested.

Usage:
    python download_data.py                          # Mumbai, 1990-2024
    python download_data.py --city pune --start 1980-01-01
    python download_data.py --lat 12.97 --lon 77.59 --city bengaluru
"""
import argparse
import hashlib
import os
import time

import pandas as pd
import requests

from src.cities import CITIES, data_path
from src.data import DAILY_RAW as DAILY_VARS, HOURLY_RAW as HOURLY_VARS

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
CACHE_DIR = "data/.cache"


def _get(params, max_retries=6, pause=1.5):
    """GET with backoff that actually respects Open-Meteo's rate limiting.

    429 means "slow down", not "give up": we wait and retry with exponential
    backoff, honouring the Retry-After header when the server sends one.
    """
    delay = 20.0
    for attempt in range(1, max_retries + 1):
        r = requests.get(ARCHIVE_URL, params=params, timeout=180)
        if r.status_code == 200:
            time.sleep(pause)       # stay under the per-minute limit
            return r.json()
        if r.status_code == 429 and attempt < max_retries:
            wait = float(r.headers.get("Retry-After", delay))
            print(f"      rate limited; waiting {wait:.0f}s "
                  f"(attempt {attempt}/{max_retries - 1})", flush=True)
            time.sleep(wait)
            delay = min(delay * 2, 300)
            continue
        if r.status_code == 429:
            raise SystemExit(
                "\nOpen-Meteo is still rate limiting after several retries.\n"
                "Progress so far is cached, so just re-run this command later "
                "(an hour is usually plenty) and it will resume where it stopped."
            )
        r.raise_for_status()


def _cache_path(kind, lat, lon, start, end):
    key = f"{kind}_{lat}_{lon}_{start}_{end}"
    digest = hashlib.md5(key.encode()).hexdigest()[:10]
    return os.path.join(CACHE_DIR, f"{kind}_{start}_{end}_{digest}.csv")


def fetch_daily(lat, lon, start, end):
    path = _cache_path("daily", lat, lon, start, end)
    if os.path.exists(path):
        return pd.read_csv(path)
    data = _get({
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "daily": ",".join(DAILY_VARS), "timezone": "auto",
    })["daily"]
    df = pd.DataFrame(data).rename(columns={"time": "date"})
    df.to_csv(path, index=False)
    return df


def fetch_hourly(lat, lon, start, end):
    """Fetch hourly variables and average them to one row per day."""
    path = _cache_path("hourly", lat, lon, start, end)
    if os.path.exists(path):
        return pd.read_csv(path)
    data = _get({
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": ",".join(HOURLY_VARS), "timezone": "auto",
    })["hourly"]
    hdf = pd.DataFrame(data)
    hdf["date"] = pd.to_datetime(hdf["time"]).dt.date.astype(str)
    hdf = hdf.groupby("date")[HOURLY_VARS].mean().reset_index()
    hdf.columns = ["date"] + [f"{c}_mean" for c in HOURLY_VARS]
    hdf.to_csv(path, index=False)
    return hdf


def blocks(start, end, years):
    """Split [start, end] into <=`years`-long inclusive blocks."""
    out, cursor = [], start
    while cursor <= end:
        stop = min(cursor + pd.DateOffset(years=years) - pd.Timedelta(days=1), end)
        out.append((cursor.strftime("%Y-%m-%d"), stop.strftime("%Y-%m-%d")))
        cursor = stop + pd.Timedelta(days=1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--city", default="mumbai")
    p.add_argument("--lat", type=float)
    p.add_argument("--lon", type=float)
    p.add_argument("--start", default="1990-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--out")
    p.add_argument("--force", action="store_true",
                   help="re-download even if the output CSV already exists")
    p.add_argument("--all", action="store_true",
                   help="download every preset city in turn")
    args = p.parse_args()

    if args.all:
        # One city exhausting its retries must not abandon the rest: record it
        # and carry on, then report what still needs a re-run at the end.
        failed = []
        for name in CITIES:
            print(f"\n=== {name} ===", flush=True)
            try:
                fetch_city(name, *CITIES[name], args.start, args.end, args.force)
            except (SystemExit, requests.RequestException) as exc:
                print(f"  !! {name} incomplete: {exc}", flush=True)
                failed.append(name)
        if failed:
            print(f"\nIncomplete: {', '.join(failed)}. Cached blocks are kept, "
                  "so re-running this command later resumes them.")
        else:
            print("\nAll cities downloaded.")
        return

    if args.lat is None or args.lon is None:
        key = args.city.lower()
        if key not in CITIES:
            raise SystemExit(
                f"Unknown city '{args.city}'. Known: {', '.join(CITIES)}. "
                "Or pass --lat/--lon explicitly."
            )
        lat, lon = CITIES[key]
    else:
        lat, lon = args.lat, args.lon

    out = args.out or data_path(args.city)
    fetch_city(args.city, lat, lon, args.start, args.end, args.force, out)


def fetch_city(city, lat, lon, start_s, end_s, force, out=None):
    out = out or data_path(city)
    args_start, args_end = start_s, end_s

    # Don't burn API quota re-fetching a dataset that is already complete.
    if os.path.exists(out) and not force:
        have = pd.read_csv(out)
        if have["date"].min() <= args_start and have["date"].max() >= args_end:
            print(f"{out} already covers {args_start} -> {args_end} "
                  f"({len(have)} rows). Nothing to do; pass --force to re-download.")
            return

    os.makedirs(CACHE_DIR, exist_ok=True)
    start, end = pd.Timestamp(args_start), pd.Timestamp(args_end)
    if end < start:
        raise SystemExit(f"--end ({args_end}) is before --start ({args_start})")

    # Daily data is cheap, so take it in 5-year blocks. Hourly is ~24x heavier
    # per day of data, so keep those requests to 2 years to stay well inside
    # the per-request weight limit that triggers 429s.
    daily_blocks = blocks(start, end, 5)
    hourly_blocks = blocks(start, end, 2)
    total = len(daily_blocks) + len(hourly_blocks)
    done = 0

    dailies = []
    for s, e in daily_blocks:
        done += 1
        cached = os.path.exists(_cache_path("daily", lat, lon, s, e))
        print(f"  [{done}/{total}] daily  {s} -> {e}"
              f"{'  (cached)' if cached else ''}", flush=True)
        dailies.append(fetch_daily(lat, lon, s, e))

    hourlies = []
    for s, e in hourly_blocks:
        done += 1
        cached = os.path.exists(_cache_path("hourly", lat, lon, s, e))
        print(f"  [{done}/{total}] hourly {s} -> {e}"
              f"{'  (cached)' if cached else ''}", flush=True)
        hourlies.append(fetch_hourly(lat, lon, s, e))

    ddf = pd.concat(dailies, ignore_index=True).drop_duplicates("date")
    hdf = pd.concat(hourlies, ignore_index=True).drop_duplicates("date")
    df = ddf.merge(hdf, on="date", how="left").sort_values("date")
    df.to_csv(out, index=False)

    rain = df["precipitation_sum"]
    print(f"\nSaved {out}  ({len(df)} days, {df.date.min()} -> {df.date.max()})")
    print(f"  columns   : {len(df.columns)}")
    print(f"  missing   : {int(df.isna().sum().sum())} cells")
    print(f"  rainy days: {(rain > 1.0).mean():.1%}   mean {rain.mean():.2f} mm   max {rain.max():.1f} mm")


if __name__ == "__main__":
    main()
