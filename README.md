# Rainfall Prediction with an LSTM

Next-day rainfall forecasting for six Indian cities, each with its own model
trained on 35 years of real daily weather. Every model answers two questions at
once:

1. **Will it rain tomorrow?** (probability, >1 mm threshold)
2. **How much?** (millimetres)

Built with PyTorch. Data comes from the [Open-Meteo ERA5 archive](https://open-meteo.com/) —
free, no API key, no manual download.

---

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python download_data.py --all      # six cities, ~35 years each
python train_all.py                # one model per city, then a comparison table
```

Each city trains in about a minute on an Apple GPU. For a single city:

```bash
python train.py --city chennai && python evaluate.py --city chennai && python plots.py --city chennai
```

**On rate limits.** The Open-Meteo public API bills by request *weight*, and a
multi-year hourly request is expensive, so a full 35-year download can hit
HTTP 429. `download_data.py` handles this rather than falling over:

- every block is cached under `data/.cache/`, so a run that gets rate limited is
  **resumable** — just re-run the same command and it picks up where it stopped
- 429s are retried with exponential backoff, honouring `Retry-After`
- hourly data is fetched in 2-year blocks (daily in 5-year blocks) to keep each
  request well under the weight limit
- if the output CSV already covers the requested range, the script exits without
  making a single API call; pass `--force` to override

Then forecast tomorrow from live data:

```bash
python predict.py --live --city mumbai
```

---

## Real-time forecasting

```bash
python predict.py --city delhi --live
```

```
Delhi   [live (Open-Meteo forecast API)]
History through : 2026-09-01  (30-day window)
Forecast for    : 2026-09-02
  rain probability : 71.9%   -> rain likely
  expected amount  :   2.2 mm
```

**Why a second API.** The ERA5 archive used for training lags real time by about
five days, so it can never answer "will it rain tomorrow". Open-Meteo's forecast
endpoint takes a `past_days` parameter that returns the same daily variables
right up to today, which is exactly the input window the model needs. It is a
separate service from the archive with its own, more generous rate limit — live
forecasts keep working even when the archive is rate limited.

In the app, the Forecast tab has a **Live / Archive CSV** switch, defaulting to
live and refreshing every 30 minutes. If the network is unavailable it falls
back to the archive CSV and says so rather than failing.

Two honest caveats:

- **Today's row is still in progress.** Its daily aggregates blend observation
  with same-day model output, so the most recent input is not a pure
  observation.
- **Slight distribution shift.** The models are trained on ERA5 *reanalysis*,
  while this endpoint serves Open-Meteo's operational analysis. The two agree
  closely but are not the same product, so live forecasts carry a little extra
  error beyond what the held-out test metrics measure.

## Web app

```bash
streamlit run app.py
```

Four tabs:

| Tab | What it shows |
|---|---|
| **Forecast** | Tomorrow's rain probability and amount, plus the 30-day window the model actually read |
| **History explorer** | Predicted vs actual for any year, defaulting to the unseen test years only |
| **Model performance** | Test metrics against both baselines, and the heavy-rain shortfall by intensity band |
| **Compare cities** | All six models side by side, each against its own persistence baseline |
| **How it works** | The loss design, the calibration step, and how leakage is avoided |

The **History explorer** defaults to 2020 onward with a checkbox to include
earlier years, because agreement on the training years is not evidence of skill
and the app shouldn't present it as if it were.

### Deploying to Streamlit Community Cloud

The app needs no secrets and no database — every city's checkpoint (576 KB each)
and history (1.5 MB each) are committed, so the whole deploy is ~13 MB.

1. Push this folder to a GitHub repository.
2. At [share.streamlit.io](https://share.streamlit.io), point a new app at the
   repo with **main file** `app.py`.
3. Deploy. First build takes a few minutes while PyTorch installs.

**The one real gotcha: PyTorch size.** On Linux, the default PyPI `torch` wheel
bundles CUDA and weighs about 2.5 GB, which blows past the deploy limit and is
the usual reason a Torch app fails to build on Streamlit Cloud. `requirements.txt`
therefore pins the CPU-only index:

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.0
```

That pulls a ~200 MB wheel, which is all a 2-layer LSTM needs. The app calls
`torch.load(..., map_location="cpu")` and picks its device at runtime, so the
same code runs on your Mac's GPU locally and on CPU in the cloud with no changes.

**Note on the download buttons.** The sidebar's *Refresh latest data* and
*Download city* buttons write to the container's ephemeral disk. They work fine
on Streamlit Cloud, but anything fetched is lost when the container restarts —
the committed Mumbai CSV is what persists.

---

## Results

Splits are **chronological**, never random, and pinned to the same window for
every city so the comparison is like-for-like:

| Split | Period | Days |
|---|---|---|
| Train | 1990–2014 | 9,101 |
| Validation | 2015–2019 | 1,826 |
| Test | 2020–2024 | 1,827 |

### All cities, held-out test years

Skill is measured against **persistence for that same city** — "tomorrow looks
like today". Comparing a wet city to a dry one on raw error alone would be
meaningless, since RMSE scales with how much rain there is to get wrong.

| City | mm/yr | Rainy days | R² | R² persistence | F1 | F1 persistence | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Mumbai | 2,455 | 37% | **0.462** | 0.201 | **0.931** | 0.922 | 0.988 |
| Pune | 1,141 | 34% | **0.343** | 0.111 | **0.855** | 0.836 | 0.961 |
| Chennai | 1,549 | 41% | **0.211** | -0.245 | **0.775** | 0.737 | 0.886 |
| Bengaluru | 1,154 | 40% | **0.178** | -0.028 | **0.792** | 0.765 | 0.904 |
| Delhi | 819 | 22% | **0.183** | -0.225 | **0.706** | 0.647 | 0.901 |

Every trained city beats its own persistence baseline on both tasks. Kolkata is pending: its archive download is still blocked by the API's daily quota (see **Rate limits** above).

**Skill tracks how monsoon-dominated the city is.** Mumbai takes ~2,500 mm in a
concentrated June–September burst, and that strong seasonal structure is exactly
what a sequence model can exploit — R² 0.462. Delhi and Bengaluru get much of
their rain from scattered convective storms, which are far less predictable from
yesterday's local averages, and skill drops to ~0.18.

Note that persistence has a **negative** R² in Delhi and Bengaluru: repeating
today's rainfall there is worse than just guessing the long-run average. The
LSTM is well clear of both.

## Why one model per city, not one shared model

Indian rainfall regimes differ too much to share weights:

- **Mumbai** — ~2,500 mm, almost all in the southwest monsoon (Jun–Sep)
- **Chennai** — peaks in the *northeast* monsoon (Oct–Dec), the opposite half of the year
- **Bengaluru** — bimodal, with two separate wet seasons
- **Delhi** — semi-arid, ~800 mm, rain on barely a fifth of days

Each city gets its own scaler, its own calibration coefficients, and its own
decision threshold, all fit on that city's data alone. The scaler point matters
most: standardising Delhi's rainfall with Mumbai's mean and standard deviation
would distort every input feature before the model ever sees it.

The app will not silently fall back to another city's weights. If a city has
data but no model, it says so and shows the command to train one.

## Known limitation: heavy rain is under-forecast

This is the model's real weakness, and it is visible in the numbers rather than
hidden behind an aggregate score:

| Intensity | Days | Mean observed | Mean predicted | MAE |
|---|---|---|---|---|
| dry (<1 mm) | 1,145 | 0.05 | 0.32 | 0.29 |
| light (1–10) | 353 | 4.42 | 6.31 | 3.51 |
| moderate (10–35) | 229 | 19.58 | 16.37 | 9.46 |
| heavy (35–65) | 73 | 46.91 | 27.60 | 20.49 |
| very heavy (>65) | 27 | 102.05 | 31.93 | 70.12 |

**Timing is good; magnitude is compressed.** The model reliably knows *when* a
wet spell arrives (see the 2024 panel in `outputs/diagnostics.png` — every
monsoon burst lines up), but it under-shoots the peak of a cloudburst.

Two causes, one of them irreducible here:

- Any model trained on a mean-seeking loss predicts a conditional mean, which is
  less variable than reality. The calibration step (below) corrects most of this.
- A 250 mm cloudburst is driven by mesoscale convection that simply is not
  present in daily-averaged, single-grid-point predictors. No amount of tuning
  recovers information the inputs never contained.

Treat any forecast above ~20 mm as "heavy rain likely" rather than a precise total.

---

## How it works

### Features (24)

Ten raw daily variables (temperature max/min/mean, precipitation sum and hours,
wind speed/gusts/direction, radiation, evapotranspiration) plus four hourly
variables averaged to daily (humidity, surface pressure, cloud cover, dew point),
and ten engineered ones:

- `doy_sin` / `doy_cos` — the annual cycle as a smooth pair, so the model isn't
  asked to learn that Dec 31 and Jan 1 are neighbours. Monsoon is the dominant
  signal in this data, so this matters a lot.
- `precip_roll3/7/30`, `wet_streak7` — running wetness. Rain clusters.
- `pressure_delta`, `humidity_delta` — falling pressure and rising humidity
  precede rain.
- `dewpoint_spread` — temperature minus dew point; a small gap means the air is
  near saturation.

Every engineered column uses only same-day-or-earlier information, so a 30-day
window ending at day *t* leaks nothing about day *t+1*. The scaler is fit on
**training rows only**.

### Model

A 2-layer LSTM (96 hidden units, dropout 0.25) reads a 30-day window and feeds
its final hidden state to two heads: one predicting `log1p(mm)`, one predicting a
rain/no-rain logit.

Why two heads: daily rainfall is **zero-inflated** — about two-thirds of days are
dry. A lone regression head learns to hedge toward zero. Training the classifier
alongside gives the shared encoder a cleaner signal for "is a wet spell starting",
and the two outputs are individually more useful than one number anyway.

### Loss

`intensity-weighted Huber on log1p(mm)  +  0.5 × BCE on rain occurrence`

Three deliberate choices, each fixing a specific failure:

- **`log1p` targets.** Raw rainfall spans 0 to 251 mm. Untransformed, a handful
  of extreme days dominate the gradient.
- **`pos_weight` on the BCE.** Without it the classifier scores 66% by always
  saying "dry".
- **Intensity weighting** (`w = 1 + 0.6·log1p(mm)`). Heavy days are rare and
  Huber saturates on exactly their error range, so their gradient was being
  drowned out by ~1,100 easy dry days. `sweep.py` picked the 0.6 coefficient on
  validation: it lifted validation R² from 0.304 to 0.356.

### Calibration

After training, a linear correction is fit **on validation only**:

```
log1p(mm_corrected) = 1.201 × log1p(mm_raw) − 0.104
```

The slope above 1 stretches the compressed forecast distribution back out. This
is the single largest improvement in the project — test R² 0.327 → 0.462, bias
−3.41 → −1.67 mm — and it is legitimate precisely because the coefficients never
see the test years. The fitted values are stored in the checkpoint, so
`evaluate.py` and `predict.py` apply them automatically.

---

## Files

| File | Purpose |
|---|---|
| `download_data.py` | Fetch daily weather from Open-Meteo (cached, resumable, 429-aware) |
| `src/cities.py` | City registry and all derived file paths |
| `src/data.py` | Feature engineering, pinned chronological splits, sliding windows |
| `src/model.py` | `RainfallLSTM` + `MultiTaskLoss` |
| `src/metrics.py` | RMSE/MAE/R², plus CSI, HSS, ROC-AUC (met-standard scores) |
| `src/calibration.py` | Validation-fit variance correction |
| `train.py` | Train one city |
| `train_all.py` | Train + evaluate every city, write the comparison summary |
| `evaluate.py` | Test metrics vs. persistence and climatology baselines |
| `sweep.py` | Loss-hyperparameter sweep, selected on validation |
| `plots.py` | Six-panel diagnostics → `outputs/diagnostics.png` |
| `predict.py` | Forecast tomorrow; `--live` uses real-time data |
| `app.py` | Streamlit web app |
| `src/inference.py` | Shared load/forecast/backtest helpers |
| `src/live.py` | Real-time weather via Open-Meteo's forecast endpoint |

## Adding another city

The six presets live in `src/cities.py`. Any coordinates work:

```bash
python download_data.py --lat 51.5 --lon -0.12 --city london
```

Then add `"london": (51.5, -0.12)` to `CITIES` in `src/cities.py` and run
`python train.py --city london`.

Other options:

```bash
python train.py --city pune --seq-len 45 --horizon 3
```

`--horizon 3` trains a 3-days-ahead model. Expect accuracy to fall off with
horizon; that decay is worth measuring rather than assuming.

## Possible next steps

- **Neighbouring grid points.** The clearest path to better heavy-rain skill:
  give the model upwind cells so it can see systems approaching, instead of only
  the history directly overhead.
- **Quantile heads.** Predict the 10th/50th/90th percentile instead of a point
  estimate, giving honest uncertainty bands for free.
- **Attention or a Transformer encoder** over the same windows, as a comparison.
