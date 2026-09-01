"""Real-time weather for forecasting today -> tomorrow.

The ERA5 *archive* used for training lags real time by about five days, so it
can never answer "will it rain tomorrow". Open-Meteo's forecast endpoint has a
`past_days` parameter that returns the same daily variables right up to today,
which is what the model needs for a live input window. It is a separate service
from the archive with its own, more generous rate limit.

One honest caveat: the models are trained on ERA5 reanalysis, while this
endpoint serves Open-Meteo's operational best-match analysis. The two agree
closely but are not the same product, so live forecasts carry a little extra
error beyond what the held-out test metrics measure.
"""
import pandas as pd
import requests

from src.cities import resolve
from src.data import DAILY_RAW, HOURLY_RAW

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_live(city=None, lat=None, lon=None, past_days=92, include_today=True):
    """Return a raw daily frame ending today (or yesterday), matching the CSV schema.

    `past_days` is capped at 92 by the API. The model only needs `seq_len` days,
    but a longer window makes the 30-day rolling features well-formed.
    """
    if lat is None or lon is None:
        lat, lon = resolve(city)
    past_days = max(1, min(int(past_days), 92))

    common = {
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "past_days": past_days,
        # forecast_days=1 includes today; 0 stops at yesterday.
        "forecast_days": 1 if include_today else 0,
    }

    d = requests.get(FORECAST_URL, params={**common, "daily": ",".join(DAILY_RAW)},
                     timeout=60)
    d.raise_for_status()
    daily = pd.DataFrame(d.json()["daily"]).rename(columns={"time": "date"})

    h = requests.get(FORECAST_URL, params={**common, "hourly": ",".join(HOURLY_RAW)},
                     timeout=60)
    h.raise_for_status()
    hdf = pd.DataFrame(h.json()["hourly"])
    hdf["date"] = pd.to_datetime(hdf["time"]).dt.date.astype(str)
    hdf = hdf.groupby("date")[HOURLY_RAW].mean().reset_index()
    hdf.columns = ["date"] + [f"{c}_mean" for c in HOURLY_RAW]

    df = daily.merge(hdf, on="date", how="left").sort_values("date")
    df["date"] = pd.to_datetime(df["date"])

    # The final row is today, which is still in progress: its daily aggregates
    # are part observation, part same-day model output. Drop rows that are
    # entirely empty, but keep a partially-filled today - it is the most
    # informative row in the window.
    df = df.dropna(subset=["precipitation_sum"]).reset_index(drop=True)
    return df
