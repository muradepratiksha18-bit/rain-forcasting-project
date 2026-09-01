"""Streamlit front-end for the rainfall LSTM.

    streamlit run app.py
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.cities import (CITY_NAMES, data_path, diagnostics_path, metrics_path,
                        model_path)
from src.data import RAIN_THRESHOLD_MM, build_features, clean_frame
from src.inference import backtest, forecast_latest, frame_with_features, load_bundle

TEST_START = "2020-01-01"      # years no model saw during training
SUMMARY = "outputs/summary.json"
BLUE, RED, GREY = "#3b6ea5", "#d1495b", "#8a8f98"

st.set_page_config(page_title="Rainfall Prediction (LSTM)", page_icon="🌧️",
                   layout="wide", initial_sidebar_state="expanded")


# ----------------------------------------------------------------- data loading
@st.cache_resource(show_spinner=False)
def get_bundle(path, mtime):        # mtime busts the cache when the model changes
    return load_bundle(path)


@st.cache_data(show_spinner=False)
def get_frame(path, mtime):
    return frame_with_features(path)


@st.cache_data(show_spinner=False)
def get_backtest(csv_file, model_file, csv_mtime, model_mtime):
    b = get_bundle(model_file, model_mtime)
    return backtest(b, get_frame(csv_file, csv_mtime))


@st.cache_data(ttl=1800, show_spinner=False)
def get_live_frame(city):
    """Real-time window ending today. Cached for 30 min so a busy app is polite
    to the API; Open-Meteo only refreshes these aggregates a few times a day."""
    from src.live import fetch_live
    df, _ = build_features(clean_frame(fetch_live(city, past_days=92)))
    return df


@st.cache_data(show_spinner=False)
def get_metrics(city, _mtime):
    path = metrics_path(city)
    return json.load(open(path)) if os.path.exists(path) else None


@st.cache_data(show_spinner=False)
def get_summary(_mtime):
    return json.load(open(SUMMARY)) if os.path.exists(SUMMARY) else None


def mtime(p):
    return os.path.getmtime(p) if os.path.exists(p) else 0


def verdict(prob):
    if prob < 0.35:
        return "No rain expected", "🌤️", GREY
    if prob < 0.65:
        return "Rain possible", "🌥️", "#c9a227"
    if prob < 0.85:
        return "Rain likely", "🌧️", BLUE
    return "Rain very likely", "⛈️", RED


def intensity_label(mm):
    if mm < 1:
        return "dry"
    if mm < 10:
        return "light"
    if mm < 35:
        return "moderate"
    if mm < 65:
        return "heavy"
    return "very heavy"


# ---------------------------------------------------------------------- sidebar
st.sidebar.title("🌧️ Rainfall LSTM")

trained = [c for c in CITY_NAMES if os.path.exists(model_path(c))]
if not trained:
    st.error("No trained models found. Run `python download_data.py --all` "
             "then `python train_all.py`.")
    st.stop()

city = st.sidebar.selectbox(
    "City", CITY_NAMES,
    format_func=lambda c: f"{c.title()}" + ("" if c in trained else "  (no model)"))
csv_path, mdl_path = data_path(city), model_path(city)

st.sidebar.caption(
    "Every city has its own model, scaler and calibration — rainfall regimes "
    "differ far too much to share weights across them."
)

if not os.path.exists(csv_path):
    st.sidebar.warning(f"No data for {city}.")
    if st.sidebar.button(f"Download {city} history"):
        with st.spinner(f"Fetching {city} from Open-Meteo (cached & resumable)…"):
            r = subprocess.run(
                [sys.executable, "download_data.py", "--city", city],
                capture_output=True, text=True)
        if r.returncode == 0:
            st.cache_data.clear()
            st.rerun()
        else:
            st.sidebar.error(r.stderr[-800:] or "download failed")
    st.info(f"No data for {city.title()} yet — download it from the sidebar, "
            f"or run `python download_data.py --all`.")
    st.stop()

if not os.path.exists(mdl_path):
    st.warning(
        f"**{city.title()} has data but no trained model yet.**\n\n"
        f"Train it with:\n\n```\npython train.py --city {city}\n"
        f"python evaluate.py --city {city}\n```\n\n"
        "Or train every city at once with `python train_all.py`. "
        "Models are deliberately not shared between cities: a Mumbai model "
        "applied to Chennai would be a demo, not a forecast."
    )
    st.stop()

if st.sidebar.button("⟳ Extend archive CSV", use_container_width=True,
                     help="Pulls newer days from the ERA5 archive into this "
                          "city's training CSV. Not needed for live forecasts, "
                          "and the archive is often rate limited."):
    end = (pd.Timestamp.today() - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    with st.spinner("Fetching the most recent observations…"):
        r = subprocess.run(
            [sys.executable, "download_data.py", "--city", city,
             "--end", end, "--force"],
            capture_output=True, text=True)
    if r.returncode == 0:
        st.cache_data.clear()
        st.rerun()
    else:
        st.sidebar.error((r.stdout or r.stderr)[-600:] or "download failed")

bundle = get_bundle(mdl_path, mtime(mdl_path))
df = get_frame(csv_path, mtime(csv_path))
cfg = bundle["config"]

st.sidebar.divider()
st.sidebar.markdown(
    f"""**Model — {city.title()}**
- 2-layer LSTM, {cfg['hidden']} hidden units
- {cfg['seq_len']}-day input window
- {cfg['horizon']}-day-ahead forecast
- {len(bundle['features'])} input features
- running on `{bundle['device']}`

**Data**
- {len(df):,} days
- {df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d}"""
)

tab_now, tab_hist, tab_perf, tab_cmp, tab_about = st.tabs(
    ["Forecast", "History explorer", "Model performance", "Compare cities",
     "How it works"])


# --------------------------------------------------------------------- forecast
with tab_now:
    source = st.radio(
        "Data source", ["Live (real-time)", "Archive CSV"],
        horizontal=True, label_visibility="collapsed",
        help="Live fetches observations up to today from Open-Meteo's forecast "
             "endpoint. The archive CSV is the training data, which stops "
             "wherever it was last downloaded.")

    fdf, live_ok = df, False
    if source.startswith("Live"):
        try:
            fdf = get_live_frame(city)
            live_ok = True
        except Exception as exc:                       # offline, or API down
            st.warning(f"Live data unavailable ({type(exc).__name__}); "
                       "falling back to the archive CSV.")

    mm, prob, target = forecast_latest(bundle, fdf)
    label, icon, colour = verdict(prob)
    last = fdf["date"].iloc[-1]

    st.subheader(f"{icon} Forecast for {target:%A, %d %B %Y} — {city.title()}")
    if live_ok:
        st.caption(f"Live — based on the {cfg['seq_len']} days up to "
                   f"**{last:%d %b %Y}** (today). Today's row is still in "
                   "progress, so it blends observation with same-day model "
                   "output. Refreshes every 30 minutes.")
    else:
        st.caption(f"Archive — based on the {cfg['seq_len']} days up to "
                   f"{last:%d %b %Y}. The ERA5 archive lags real time by about "
                   "five days, so switch to **Live** for a same-day forecast.")

    c1, c2, c3 = st.columns([1, 1, 1])
    c1.metric("Chance of rain", f"{prob:.0%}", label)
    c2.metric("Expected rainfall", f"{mm:.1f} mm", intensity_label(mm))
    c3.metric("Forecast made from", f"{last:%d %b %Y}",
              f"{cfg['seq_len']}-day window")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": "Probability of rain (>1 mm)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": colour},
            "steps": [
                {"range": [0, 35], "color": "#eef1f4"},
                {"range": [35, 65], "color": "#e2e8ef"},
                {"range": [65, 100], "color": "#d3dde8"},
            ],
        },
    ))
    gauge.update_layout(height=260, margin=dict(t=50, b=10, l=30, r=30))

    left, right = st.columns([1, 2])
    left.plotly_chart(gauge, use_container_width=True)

    # What the model actually looked at.
    w = fdf.tail(cfg["seq_len"])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=w["date"], y=w["precipitation_sum"], name="rainfall",
                         marker_color=BLUE), secondary_y=False)
    fig.add_trace(go.Scatter(x=w["date"], y=w["relative_humidity_2m_mean"],
                             name="humidity", line=dict(color=RED, width=2)),
                  secondary_y=True)
    fig.update_yaxes(title_text="rainfall (mm)", secondary_y=False)
    fig.update_yaxes(title_text="humidity (%)", secondary_y=True)
    fig.update_layout(height=260, margin=dict(t=40, b=10, l=10, r=10),
                      title=f"The {cfg['seq_len']}-day window fed to the model",
                      legend=dict(orientation="h", y=1.15))
    right.plotly_chart(fig, use_container_width=True)

    if mm > 20:
        st.warning(
            "**Heavy rain signalled.** The model systematically under-forecasts "
            "extreme days, so read anything above ~20 mm as *heavy rain likely* "
            "rather than a precise total. See **Model performance** for the "
            "measured shortfall by intensity band."
        )


# ------------------------------------------------------------- history explorer
with tab_hist:
    st.subheader("Predicted vs actual, day by day")
    bt = get_backtest(csv_path, mdl_path, mtime(csv_path), mtime(mdl_path))
    bt["date"] = pd.to_datetime(bt["date"])

    test_only = st.checkbox(
        f"Show only unseen test years ({TEST_START[:4]} onward)", value=True,
        help="Before this date the model was trained on these days, so agreement "
             "there is not evidence of skill.")
    view = bt[bt["date"] >= TEST_START] if test_only else bt

    years = sorted(view["date"].dt.year.unique())
    if not years:
        st.info("No rows in this range.")
    else:
        year = st.select_slider("Year", options=years, value=years[-1])
        y = view[view["date"].dt.year == year]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y["date"], y=y["actual_mm"], name="actual",
                                 fill="tozeroy", line=dict(color=BLUE, width=0.5)))
        fig.add_trace(go.Scatter(x=y["date"], y=y["pred_mm"], name="predicted",
                                 line=dict(color=RED, width=1.6)))
        fig.update_layout(height=380, margin=dict(t=30, b=10, l=10, r=10),
                          yaxis_title="rainfall (mm)",
                          legend=dict(orientation="h", y=1.1),
                          hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        obs, pred = y["actual_mm"] > RAIN_THRESHOLD_MM, y["prob"] >= 0.5
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"{year} total, actual", f"{y['actual_mm'].sum():,.0f} mm")
        k2.metric(f"{year} total, predicted", f"{y['pred_mm'].sum():,.0f} mm",
                  f"{y['pred_mm'].sum() - y['actual_mm'].sum():+,.0f} mm")
        k3.metric("Rain days called right", f"{(obs == pred).mean():.0%}")
        k4.metric("Wettest day", f"{y['actual_mm'].max():.0f} mm",
                  f"predicted {y.loc[y['actual_mm'].idxmax(), 'pred_mm']:.0f} mm")

        with st.expander("Browse the underlying days"):
            show = y.copy()
            show["date"] = show["date"].dt.strftime("%Y-%m-%d")
            show["prob"] = (show["prob"] * 100).round(1)
            st.dataframe(
                show.rename(columns={"actual_mm": "actual (mm)",
                                     "pred_mm": "predicted (mm)",
                                     "prob": "rain probability (%)"})
                    .round(2).sort_values("date", ascending=False),
                use_container_width=True, hide_index=True, height=320)


# ---------------------------------------------------------------- model metrics
with tab_perf:
    m = get_metrics(city, mtime(metrics_path(city)))
    if not m:
        st.info(f"Run `python evaluate.py --city {city}` to generate metrics.")
    else:
        st.subheader(f"Held-out test period: {m['test_start']} → {m['test_end']}")
        st.caption(f"{m['n_test_days']:,} days the model never saw in training. "
                   "Splits are chronological, never random.")

        r, c = m["regression"], m["classification"]
        # st.table renders real HTML: small comparison tables stay readable,
        # unlike st.dataframe's canvas grid which collapses narrow columns.
        st.markdown("**Will it rain tomorrow?**")
        keys = ["accuracy", "precision", "recall", "f1", "csi", "hss", "roc_auc"]
        st.table(pd.DataFrame(
            [[f"{c['lstm'][k]:.3f}" for k in keys],
             [f"{c['persistence'][k]:.3f}" for k in keys]],
            index=["LSTM", "persistence (same as today)"],
            columns=["Accuracy", "Precision", "Recall", "F1", "CSI", "HSS", "ROC-AUC"],
        ))

        st.markdown("**How much rain?**")
        rows = [("LSTM (calibrated)", r["lstm"]), ("LSTM (raw)", r["lstm_raw"]),
                ("persistence", r["persistence"]), ("climatology", r["climatology"])]
        st.table(pd.DataFrame(
            [[f"{v['rmse_mm']:.2f}", f"{v['mae_mm']:.2f}",
              f"{v['r2']:.3f}", f"{v['bias_mm']:+.2f}"] for _, v in rows],
            index=[n for n, _ in rows],
            columns=["RMSE (mm)", "MAE (mm)", "R²", "bias (mm)"],
        ))
        st.caption("Persistence is the honest bar: rainfall is strongly "
                   "autocorrelated, so a model that cannot beat *tomorrow looks "
                   "like today* has learned nothing useful.")

        st.markdown("**Known limitation — heavy rain is under-forecast**")
        bands = pd.DataFrame(m["by_intensity"])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bands["band"], y=bands["mean_observed_mm"],
                             name="observed", marker_color=BLUE))
        fig.add_trace(go.Bar(x=bands["band"], y=bands["mean_predicted_mm"],
                             name="predicted", marker_color=RED))
        fig.update_layout(barmode="group", height=320, yaxis_title="mean mm",
                          margin=dict(t=30, b=10, l=10, r=10),
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "Timing is good; magnitude is compressed. Part of this is "
            "irreducible here — a cloudburst is driven by mesoscale convection "
            "that simply is not present in daily-averaged, single-grid-point "
            "inputs."
        )
        st.table(pd.DataFrame(
            [[f"{r['n']:,}", f"{r['mean_observed_mm']:.2f}",
              f"{r['mean_predicted_mm']:.2f}", f"{r['mae_mm']:.2f}"]
             for _, r in bands.iterrows()],
            index=bands["band"].tolist(),
            columns=["Days", "Mean observed (mm)", "Mean predicted (mm)", "MAE (mm)"],
        ))

        if os.path.exists(diagnostics_path(city)):
            with st.expander("Full diagnostic panel"):
                st.image(diagnostics_path(city), use_container_width=True)


# ----------------------------------------------------------------- comparison
with tab_cmp:
    summary = get_summary(mtime(SUMMARY))
    if not summary:
        st.info("Run `python train_all.py` to train every city and build "
                "`outputs/summary.json`.")
    else:
        st.subheader("All cities, held-out test years")
        st.caption("Each city has its own model. Skill is reported against "
                   "persistence for that same city, because a wet city and a "
                   "dry city are not comparable on raw error alone.")

        rows = []
        for c, r in summary.items():
            rl, rp = r["regression"]["lstm"], r["regression"]["persistence"]
            cl, cp = r["classification"]["lstm"], r["classification"]["persistence"]
            rows.append({
                "city": c.title(),
                "mm/yr": f"{r['mean_annual_mm']:,.0f}",
                "rainy days": f"{r['rain_rate']:.0%}",
                "R²": f"{rl['r2']:.3f}",
                "R² persist": f"{rp['r2']:.3f}",
                "RMSE (mm)": f"{rl['rmse_mm']:.2f}",
                "F1": f"{cl['f1']:.3f}",
                "F1 persist": f"{cp['f1']:.3f}",
                "ROC-AUC": f"{cl['roc_auc']:.3f}",
            })
        table = pd.DataFrame(rows).set_index("city")
        st.table(table)

        cities = [c.title() for c in summary]
        c1, c2 = st.columns(2)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=cities, name="LSTM", marker_color=BLUE,
                             y=[summary[c]["regression"]["lstm"]["r2"] for c in summary]))
        fig.add_trace(go.Bar(x=cities, name="persistence", marker_color=GREY,
                             y=[summary[c]["regression"]["persistence"]["r2"] for c in summary]))
        fig.update_layout(barmode="group", height=330, title="Rainfall amount — R²",
                          yaxis_title="R²", margin=dict(t=50, b=10, l=10, r=10),
                          legend=dict(orientation="h", y=1.15))
        c1.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=cities, name="LSTM", marker_color=BLUE,
                              y=[summary[c]["classification"]["lstm"]["f1"] for c in summary]))
        fig2.add_trace(go.Bar(x=cities, name="persistence", marker_color=GREY,
                              y=[summary[c]["classification"]["persistence"]["f1"] for c in summary]))
        fig2.update_layout(barmode="group", height=330, title="Rain / no-rain — F1",
                           yaxis_title="F1", margin=dict(t=50, b=10, l=10, r=10),
                           legend=dict(orientation="h", y=1.15))
        c2.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Annual rainfall — why one shared model would not work**")
        fig3 = go.Figure(go.Bar(
            x=cities, y=[summary[c]["mean_annual_mm"] for c in summary],
            marker_color=BLUE,
            text=[f"{summary[c]['mean_annual_mm']:,.0f} mm" for c in summary],
            textposition="outside"))
        fig3.update_layout(height=300, yaxis_title="mm per year",
                           margin=dict(t=20, b=10, l=10, r=10))
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Wet and dry cities differ by several times in annual total, "
                   "and their monsoon timing differs too — Chennai peaks in the "
                   "northeast monsoon (Oct–Dec), Bengaluru is bimodal. Sharing "
                   "one scaler across them would distort every input feature.")


# ------------------------------------------------------------------------ about
with tab_about:
    st.markdown(f"""
### The model

A 2-layer LSTM ({cfg['hidden']} hidden units, dropout {cfg['dropout']}) reads a
{cfg['seq_len']}-day window of {len(bundle['features'])} weather features and
feeds its final hidden state to **two heads**: one predicts `log1p(mm)`, the
other a rain / no-rain logit.

**Why two heads.** Daily rainfall is zero-inflated — about two-thirds of days are
dry. A lone regression head learns to hedge toward zero. Training the classifier
alongside gives the shared encoder a cleaner signal for *is a wet spell starting*,
and the two outputs are individually more useful than one number anyway.

### The loss

`intensity-weighted Huber on log1p(mm)  +  {cfg.get('clf_weight', 0.5)} × BCE on rain occurrence`

- **`log1p` targets** — raw rainfall spans 0 to 251 mm; untransformed, a handful
  of extreme days dominates the gradient.
- **`pos_weight` on the BCE** — without it the classifier scores 66% by always
  saying "dry".
- **Intensity weighting** — heavy days are rare and Huber saturates on exactly
  their error range, so their gradient was drowned out by ~1,100 easy dry days.

### Calibration

After training, a linear correction is fit **on validation data only**:

`log1p(mm_corrected) = {bundle['calibration'][0]:.3f} × log1p(mm_raw) {bundle['calibration'][1]:+.3f}`

A slope above 1 stretches the compressed forecast distribution back out. This was
the single largest improvement in the project — test R² 0.327 → 0.462 — and it is
legitimate precisely because the coefficients never see the test years.

### Avoiding leakage

Every engineered feature uses only same-day-or-earlier information, the scaler is
fit on training rows only, and the train/validation/test split is chronological.
A random split would let the model peek at the future and inflate every metric.

### Data

[Open-Meteo ERA5 archive](https://open-meteo.com/) — free, no API key.
Ten raw daily variables, four hourly variables averaged to daily, and ten
engineered features (annual-cycle sin/cos, rolling rain totals, wet streaks,
pressure and humidity deltas, dew-point spread).
""")
