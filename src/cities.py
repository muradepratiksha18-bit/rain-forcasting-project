"""Registry of supported cities and the file paths derived from them.

Every city gets its own model, scaler and calibration. Indian rainfall regimes
differ far too much to share weights: Mumbai takes ~2,500 mm almost entirely in
the June-September southwest monsoon, Chennai peaks in the *northeast* monsoon
in October-December, Bengaluru is bimodal, and Delhi is semi-arid. A model
trained on one and applied to another is not a forecast.
"""

CITIES = {
    "mumbai": (19.0760, 72.8777),
    "pune": (18.5204, 73.8567),
    "delhi": (28.6139, 77.2090),
    "bengaluru": (12.9716, 77.5946),
    "chennai": (13.0827, 80.2707),
    "kolkata": (22.5726, 88.3639),
}

CITY_NAMES = list(CITIES)


def data_path(city):
    return f"data/weather_{city.lower()}.csv"


def model_path(city):
    return f"models/rainfall_lstm_{city.lower()}.pt"


def metrics_path(city):
    return f"outputs/metrics_{city.lower()}.json"


def history_path(city):
    return f"outputs/history_{city.lower()}.json"


def predictions_path(city):
    return f"outputs/test_predictions_{city.lower()}.npz"


def diagnostics_path(city):
    return f"outputs/diagnostics_{city.lower()}.png"


def resolve(city):
    key = city.lower()
    if key not in CITIES:
        raise SystemExit(
            f"Unknown city '{city}'. Known: {', '.join(CITY_NAMES)}. "
            "Or pass --lat/--lon to download_data.py explicitly."
        )
    return CITIES[key]
