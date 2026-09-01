"""Feature engineering and sliding-window construction for the rainfall LSTM."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Fixed split boundaries. These are pinned rather than derived from whatever
# happens to be in the CSV: the app can extend a city's file with fresher data
# at any time, and evaluation must not silently change underneath us. Pinning
# TEST_END also keeps the cross-city comparison like-for-like.
TRAIN_END = "2014-12-31"
VAL_END = "2019-12-31"
TEST_END = "2024-12-31"

# Threshold (mm/day) above which a day counts as "rainy" for the classifier.
# 1 mm is the conventional cutoff used by met agencies (below that is trace/drizzle).
RAIN_THRESHOLD_MM = 1.0

# Raw variables as served by Open-Meteo (daily aggregates, and hourly ones we
# average down ourselves). Shared with download_data.py and src/live.py.
DAILY_RAW = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]

HOURLY_RAW = [
    "relative_humidity_2m",
    "surface_pressure",
    "cloud_cover",
    "dew_point_2m",
]

BASE_FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
    "cloud_cover_mean",
    "dew_point_2m_mean",
]


@dataclass
class Splits:
    """Windowed tensors for one train/val/test split, plus the fitted scaler."""
    X_train: np.ndarray
    y_train: np.ndarray
    c_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    c_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    c_test: np.ndarray
    dates_test: np.ndarray
    feature_names: list = field(default_factory=list)
    mu: np.ndarray = None
    sd: np.ndarray = None


def clean_frame(df):
    """Sort, and fill any gaps. ERA5 has none, but live and user CSVs can."""
    df = df.sort_values("date").reset_index(drop=True)
    df[BASE_FEATURES] = df[BASE_FEATURES].interpolate(limit_direction="both")
    return df


def load_frame(csv_path):
    return clean_frame(pd.read_csv(csv_path, parse_dates=["date"]))


def build_features(df):
    """Add calendar and lag/rolling features. Returns (df, feature_name_list).

    Every engineered column uses only information available up to and including
    that same day, so a window ending at day t leaks nothing about day t+1.
    """
    out = df.copy()
    doy = out["date"].dt.dayofyear

    # Monsoon is the dominant signal; encode the annual cycle as a smooth pair
    # so the model doesn't have to learn that Dec 31 and Jan 1 are neighbours.
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Rain is extremely right-skewed (most days 0, a few days 250 mm).
    # Learning on log1p keeps the loss from being dominated by a handful of days.
    out["precip_log"] = np.log1p(out["precipitation_sum"])

    # Short-memory persistence features. Rainfall is strongly autocorrelated:
    # wet days cluster, and these give the LSTM the running state explicitly.
    for w in (3, 7, 30):
        out[f"precip_roll{w}"] = out["precipitation_sum"].rolling(w, min_periods=1).mean()
    out["wet_streak7"] = (
        (out["precipitation_sum"] > RAIN_THRESHOLD_MM).rolling(7, min_periods=1).sum()
    )
    # Falling pressure and rising humidity precede rain.
    out["pressure_delta"] = out["surface_pressure_mean"].diff().fillna(0.0)
    out["humidity_delta"] = out["relative_humidity_2m_mean"].diff().fillna(0.0)
    # Dew point depression: small gap = air near saturation.
    out["dewpoint_spread"] = out["temperature_2m_mean"] - out["dew_point_2m_mean"]

    features = BASE_FEATURES + [
        "doy_sin", "doy_cos", "precip_log",
        "precip_roll3", "precip_roll7", "precip_roll30", "wet_streak7",
        "pressure_delta", "humidity_delta", "dewpoint_spread",
    ]
    return out, features


def make_windows(values, targets_reg, targets_clf, dates, seq_len, horizon):
    """Slice into (n, seq_len, n_features) windows predicting `horizon` days ahead."""
    X, y, c, d = [], [], [], []
    last = len(values) - seq_len - horizon + 1
    for i in range(last):
        t = i + seq_len + horizon - 1        # index of the day being predicted
        X.append(values[i:i + seq_len])
        y.append(targets_reg[t])
        c.append(targets_clf[t])
        d.append(dates[t])
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(c, dtype=np.float32),
        np.asarray(d),
    )


def prepare(csv_path, seq_len=30, horizon=1, train_end=TRAIN_END,
            val_end=VAL_END, test_end=TEST_END):
    """Full pipeline: load -> engineer -> chronological split -> scale -> window.

    `test_end` caps the evaluation period so results stay reproducible even if
    the CSV later gains more recent days. Pass None to use everything available.
    """
    df = load_frame(csv_path)
    if test_end is not None:
        df = df[df["date"] <= pd.Timestamp(test_end)].reset_index(drop=True)
    df, features = build_features(df)

    # Regression target is log1p(mm); we invert with expm1 at evaluation time.
    y_reg = np.log1p(df["precipitation_sum"].values)
    y_clf = (df["precipitation_sum"].values > RAIN_THRESHOLD_MM).astype(np.float32)
    values = df[features].values.astype(np.float64)
    dates = df["date"].values

    # Split by time, never randomly - a random split would let the model see
    # the future and inflate every metric.
    tr = df["date"] <= pd.Timestamp(train_end)
    va = (df["date"] > pd.Timestamp(train_end)) & (df["date"] <= pd.Timestamp(val_end))
    te = df["date"] > pd.Timestamp(val_end)
    if not te.any():
        raise SystemExit(
            f"{csv_path} has no rows after {val_end} - nothing to test on.")

    # Scaler fit on training rows only.
    mu = values[tr.values].mean(axis=0)
    sd = values[tr.values].std(axis=0)
    sd[sd < 1e-8] = 1.0
    scaled = (values - mu) / sd

    def slice_split(mask):
        idx = np.where(mask.values)[0]
        lo, hi = idx[0], idx[-1] + 1
        # Reach back seq_len days so the first window of val/test is complete.
        lo_ctx = max(0, lo - seq_len - horizon + 1)
        return make_windows(
            scaled[lo_ctx:hi], y_reg[lo_ctx:hi], y_clf[lo_ctx:hi],
            dates[lo_ctx:hi], seq_len, horizon,
        )

    Xtr, ytr, ctr, _ = slice_split(tr)
    Xva, yva, cva, _ = slice_split(va)
    Xte, yte, cte, dte = slice_split(te)

    return Splits(
        Xtr, ytr, ctr, Xva, yva, cva, Xte, yte, cte, dte,
        feature_names=features, mu=mu, sd=sd,
    )
