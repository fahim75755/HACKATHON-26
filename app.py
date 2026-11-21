"""Task 4 - dashboard for public health officers.

Run:  streamlit run app.py
Officer picks a place (city, ward) and a date; the app forecasts dengue cases
2, 3 or 4 weeks later, the risk tier, the recommended action and the main drivers.
"""
import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dengue import engine
from dengue.config import (
    FEATURES_PATH, HORIZONS, HUMID_COL, METRICS_PATH, PREDICTIONS_PATH, RAIN_COL,
    TARGET_COL, TEMP_COL,
)
from dengue.model import load_artifacts
from dengue.tiers import DEFAULT_THRESHOLDS, TIER_META, TIERS

st.set_page_config(page_title="Dengue Early-Warning", page_icon="🦟", layout="wide")


# ---------------------------------------------------------------- loading (cached)
@st.cache_resource(show_spinner="Loading models ...")
def get_models():
    return load_artifacts()


@st.cache_data(show_spinner="Loading forecasts ...")
def get_tables():
    feats = pd.read_parquet(FEATURES_PATH)
    preds = pd.read_parquet(PREDICTIONS_PATH)
    metrics = json.loads(METRICS_PATH.read_text())
    return feats, preds, metrics


missing = [p for p in (FEATURES_PATH, PREDICTIONS_PATH, METRICS_PATH) if not p.exists()]
if missing:
    st.error("Model files not found. Run `python train.py` once, then reload this page.")
    st.stop()

t_load = time.perf_counter()
models = get_models()
features, preds, metrics = get_tables()
load_ms = (time.perf_counter() - t_load) * 1000

data_start, data_end = features["date"].min(), features["date"].max()

# ---------------------------------------------------------------- sidebar inputs
st.sidebar.header("📍 Place and date")
cities = sorted(features["city"].unique())
city = st.sidebar.selectbox("City", cities, index=cities.index("Dhaka North"))
wards = sorted(features.loc[features["city"] == city, "ward_id"].unique())
ward = st.sidebar.selectbox("Ward", wards)

chosen_date = st.sidebar.date_input(
    "Forecast from week of",
    value=pd.Timestamp("2025-07-14").date(),
    min_value=data_start.date(),
    max_value=pd.Timestamp("2027-12-31").date(),
    help="Data runs from %s to %s. For later dates you enter recent weather yourself."
    % (data_start.date(), data_end.date()),
)
horizon = st.sidebar.segmented_control(
    "Weeks ahead", HORIZONS, default=3, required=True,
    format_func=lambda h: f"{h} weeks",
)

issue_week = engine.snap_to_week(features, ward, pd.Timestamp(chosen_date))
manual_mode = issue_week is None
traits_base = engine.ward_traits(features, ward)

st.sidebar.divider()
st.sidebar.header("⚡ What-if")
st.sidebar.caption("Change recent weather or ward conditions and see the forecast update.")
rain_pct = st.sidebar.slider("Rain change, last 4 weeks (%)", -80, 200, 0, step=10)
humid_delta = st.sidebar.slider("Humidity change (points)", -20, 20, 0, step=2)
temp_delta = st.sidebar.slider("Temperature change (°C)", -4.0, 4.0, 0.0, step=0.5)
wl = st.sidebar.slider("Water-logging risk", 0.0, 1.0, round(traits_base["water_logging_risk_index"], 2),
                       step=0.01, key=f"wl_{ward}")
san = st.sidebar.slider("Sanitation score", 0.0, 1.0, round(traits_base["sanitation_score"], 2),
                        step=0.01, key=f"san_{ward}")
traits = {**traits_base, "water_logging_risk_index": wl, "sanitation_score": san}
what_if = (
    rain_pct != 0 or humid_delta != 0 or temp_delta != 0
    or abs(wl - traits_base["water_logging_risk_index"]) > 0.005
    or abs(san - traits_base["sanitation_score"]) > 0.005
)

# ---------------------------------------------------------------- header
st.title("🦟 Urban Dengue Early-Warning")
st.caption(
    f"{city} · {ward} · LightGBM forecasts from weather of the last 4 weeks "
    f"(rain needs 2–4 weeks to become mosquitoes and then cases)."
)

# ---------------------------------------------------------------- build forecast
artifact = models[horizon]
t_pred = time.perf_counter()
if manual_mode:
    issue_week = pd.Timestamp(chosen_date)
    st.info(
        f"No recorded weather for {issue_week.date()} (data ends {data_end.date()}). "
        "Enter the last 4 weeks below - pre-filled with this ward's usual values for this time of year."
    )
    normals = engine.seasonal_normals(features, ward, issue_week)
    edited = st.data_editor(
        normals.rename(columns={
            "date": "Week of", RAIN_COL: "Rain (mm)", HUMID_COL: "Humidity (%)",
            TEMP_COL: "Avg temp (°C)", TARGET_COL: "Reported cases",
        }),
        hide_index=True, disabled=["Week of"], key=f"manual_{ward}_{issue_week.date()}",
        column_config={"Week of": st.column_config.DateColumn(format="YYYY-MM-DD")},
    )
    weeks = edited.rename(columns={
        "Rain (mm)": RAIN_COL, "Humidity (%)": HUMID_COL,
        "Avg temp (°C)": TEMP_COL, "Reported cases": TARGET_COL,
    })
    weeks = engine.apply_what_if(weeks, rain_pct, humid_delta, temp_delta)
    X = engine.build_row(features, ward, issue_week, weeks, traits)
    fc = engine.predict_live(artifact, X, issue_week)
else:
    weeks = engine.recent_weeks(features, ward, issue_week)
    frow = features[(features["ward_id"] == ward) & (features["date"] == issue_week)]
    if what_if:
        weeks = engine.apply_what_if(weeks, rain_pct, humid_delta, temp_delta)
        X = engine.build_row(features, ward, issue_week, weeks, traits)
        fc = engine.predict_live(artifact, X, issue_week, frow[f"target_h{horizon}"].iloc[0])
    else:
        X = frow
        fc = engine.lookup(preds, features, ward, issue_week, horizon)
pred_ms = (time.perf_counter() - t_pred) * 1000

if what_if:
    st.warning(
        f"What-if active: rain {rain_pct:+d}%, humidity {humid_delta:+d}, temperature {temp_delta:+.1f}°C, "
        f"water-logging {wl:.2f}, sanitation {san:.2f}"
    )

tab_ward, tab_map, tab_valid, tab_about = st.tabs(
    ["📍 Ward forecast", "🗺️ All wards", "📊 Model validation", "ℹ️ How it works"]
)

# ---------------------------------------------------------------- tab 1: ward forecast
with tab_ward:
    st.subheader(
        f"Forecast for week of {fc.target_date.date()}  "
        f"({horizon} weeks after {fc.issue_date.date()})"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected cases", f"{fc.cases:.1f}", border=True)
    c2.metric("Likely range (80%)", f"{fc.low:.0f} – {fc.high:.0f}", border=True)
    c3.metric("Risk tier", fc.tier, border=True)
    if fc.actual is not None and not what_if:
        c4.metric("What actually happened", f"{fc.actual:.0f} cases", border=True,
                  help="Recorded cases in that week (shown for past dates, to check the model).")
    else:
        c4.metric("Weeks of warning", f"{horizon}", border=True)

    st.markdown(
        f"""<div style="padding:12px 16px;border-radius:8px;border-left:6px solid {fc.color};
        background:rgba(128,128,128,0.10);margin:4px 0 14px 0;">
        <b style="color:{fc.color}">{fc.tier.upper()} RISK</b> &nbsp;·&nbsp; {ward}, {city}<br/>
        <b>Recommended action:</b> {fc.action}</div>""",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1.15])
    with left:
        st.markdown("**Why this forecast** - what pushes cases up or down")
        drv, base = engine.drivers(artifact, X)
        drv = drv.iloc[::-1]
        colors = ["#C62828" if v > 0 else "#2E7D32" for v in drv["effect_cases"]]
        span = max(drv["effect_cases"].abs().max(), 0.5)
        fig = go.Figure(go.Bar(
            x=drv["effect_cases"], y=drv["driver"], orientation="h", marker_color=colors,
            text=[f"{v:+.1f}" for v in drv["effect_cases"]], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:+.2f} cases<extra></extra>",
        ))
        fig.update_layout(
            height=360, margin=dict(l=10, r=40, t=10, b=10),
            xaxis=dict(title=f"Cases added (+) or removed (−) vs a baseline of {base:.1f}",
                       range=[-span * 1.25 if (drv["effect_cases"] < 0).any() else 0, span * 1.25]),
        )
        st.plotly_chart(fig)
    with right:
        st.markdown("**Rain and cases** - shaded weeks are the inputs to this forecast")
        hist = features[(features["ward_id"] == ward) & (features["date"] <= fc.issue_date)].tail(20)
        fig = go.Figure()
        fig.add_bar(x=hist["date"], y=hist[RAIN_COL], name="Rain (mm)",
                    marker_color="rgba(100,181,246,0.40)")
        if manual_mode or what_if:
            fig.add_bar(x=weeks["date"] if "date" in weeks else weeks["Week of"], y=weeks[RAIN_COL],
                        name="Rain used (what-if / entered)",
                        marker_color="rgba(239,108,0,0.55)")
        fig.add_scatter(x=hist["date"], y=hist[TARGET_COL], name="Cases", mode="lines+markers",
                        line=dict(color="#F06292", width=3), yaxis="y2")
        future = features[(features["ward_id"] == ward) & (features["date"] > fc.issue_date)
                          & (features["date"] <= fc.target_date)]
        if len(future):
            fig.add_scatter(x=future["date"], y=future[TARGET_COL], name="Cases (actual, later)",
                            mode="lines+markers", line=dict(color="#B0BEC5", dash="dot"), yaxis="y2")
        fig.add_scatter(
            x=[fc.target_date], y=[fc.cases], mode="markers", name=f"Forecast (+{horizon} wk)", yaxis="y2",
            marker=dict(color=fc.color, size=14, symbol="diamond"),
            error_y=dict(type="data", symmetric=False, array=[fc.high - fc.cases],
                         arrayminus=[fc.cases - fc.low], color=fc.color),
        )
        fig.add_vrect(x0=fc.issue_date - pd.Timedelta(days=24), x1=fc.issue_date + pd.Timedelta(days=3),
                      fillcolor="rgba(30,136,229,0.10)", line_width=0)
        fig.update_layout(
            height=360, margin=dict(l=10, r=10, t=10, b=10), barmode="overlay",
            yaxis=dict(title="Rain (mm)", side="right", showgrid=False, rangemode="tozero"),
            yaxis2=dict(title="Weekly cases", overlaying="y", side="left", rangemode="tozero"),
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig)

    with st.expander("Weather used for this forecast (last 4 weeks)"):
        show = weeks.copy()
        show_cols = [c for c in ["date", "Week of", RAIN_COL, HUMID_COL, TEMP_COL, TARGET_COL] if c in show]
        st.dataframe(show[show_cols], hide_index=True)

# ---------------------------------------------------------------- tab 2: all wards
with tab_map:
    if manual_mode:
        st.info("The all-ward view uses recorded data. Pick a date up to %s." % data_end.date())
    else:
        st.subheader(f"Which wards to target first - forecast for {fc.target_date.date()}")
        week = preds[preds["date"] == fc.issue_date].copy()
        week["Expected cases"] = week[f"pred_h{horizon}"].round(1)
        week["Range"] = week[f"low_h{horizon}"].round(0).astype(int).astype(str) + "–" + \
            week[f"high_h{horizon}"].round(0).astype(int).astype(str)
        week["Risk tier"] = [engine.assign_risk_tier(v) for v in week[f"pred_h{horizon}"]]
        k = st.columns(4)
        for col, tier in zip(k, reversed(TIERS)):
            col.metric(f"{tier} wards", int((week["Risk tier"] == tier).sum()), border=True)

        st.markdown(f"**Ward heatmap - {city}**, forecasts issued over the last 16 weeks")
        recent = preds[(preds["city"] == city) & (preds["date"] <= fc.issue_date)]
        recent = recent[recent["date"] > fc.issue_date - pd.Timedelta(weeks=16)]
        grid = recent.pivot(index="ward_id", columns="date", values=f"pred_h{horizon}")
        tier_idx = np.vectorize(lambda v: TIERS.index(engine.assign_risk_tier(v)))(grid.to_numpy())
        colors = [TIER_META[t]["color"] for t in TIERS]
        scale = []
        for i, c in enumerate(colors):
            scale += [[i / len(colors), c], [(i + 1) / len(colors), c]]
        fig = go.Figure(go.Heatmap(
            z=tier_idx, x=[f"{(d + pd.Timedelta(weeks=horizon)).date()}" for d in grid.columns],
            y=grid.index, zmin=-0.5, zmax=len(TIERS) - 0.5, colorscale=scale,
            text=np.round(grid.to_numpy(), 1), texttemplate="%{text}",
            hovertemplate="%{y}<br>week of %{x}<br>expected %{text} cases<extra></extra>",
            colorbar=dict(tickvals=list(range(len(TIERS))), ticktext=TIERS, title="Tier"),
        ))
        fig.update_layout(height=max(300, 26 * len(grid) + 80), margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Forecast week")
        st.plotly_chart(fig)

        st.markdown("**All 60 wards, highest risk first**")
        st.dataframe(
            week.sort_values("Expected cases", ascending=False)[
                ["city", "ward_id", "Risk tier", "Expected cases", "Range"]
            ].rename(columns={"city": "City", "ward_id": "Ward"}),
            hide_index=True, height=320,
        )

# ---------------------------------------------------------------- tab 3: validation
with tab_valid:
    st.subheader(f"How accurate is it? Tested on {metrics['test_year']} (never used in training)")
    st.caption("Train: target weeks up to 2023 · Tune / early stopping: 2024 · Test: 2025. "
               "Split by date, never shuffled, so the model never sees the future.")
    names = {
        "lightgbm": "LightGBM (ours)",
        "negative_binomial_glm": "Negative binomial GLM",
        "same_week_last_year": "Same week last year",
        "persistence_cases_now": "This week's cases",
    }
    rows = []
    for hk, m in metrics["horizons"].items():
        for key, label in names.items():
            r = m[key]
            rows.append({"Weeks ahead": int(hk[1:]), "Model": label, "MAE": r["MAE"], "RMSE": r["RMSE"],
                         "Tier accuracy": r["tier_accuracy"],
                         "High+Critical caught": r["high_critical_recall"],
                         "80% range coverage": r.get("range_80_coverage")})
    table = pd.DataFrame(rows)
    fig = go.Figure()
    for label in names.values():
        sub = table[table["Model"] == label]
        fig.add_bar(x=[f"{h} weeks" for h in sub["Weeks ahead"]], y=sub["MAE"], name=label)
    fig.update_layout(height=320, barmode="group", yaxis_title="Average error (cases/week, lower is better)",
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig)
    st.dataframe(table, hide_index=True, column_config={
        "Tier accuracy": st.column_config.NumberColumn(format="percent"),
        "High+Critical caught": st.column_config.NumberColumn(format="percent"),
        "80% range coverage": st.column_config.NumberColumn(format="percent"),
    })

    st.markdown(f"**{ward}: forecast vs actual through {metrics['test_year']}** ({horizon} weeks ahead)")
    w_pred = preds[preds["ward_id"] == ward][["date", f"pred_h{horizon}", f"low_h{horizon}", f"high_h{horizon}"]].copy()
    w_pred["target_date"] = w_pred["date"] + pd.Timedelta(weeks=horizon)
    w_act = features[features["ward_id"] == ward][["date", TARGET_COL]]
    w = w_pred.merge(w_act, left_on="target_date", right_on="date", suffixes=("", "_t"))
    w = w[w["target_date"].dt.year == metrics["test_year"]]
    fig = go.Figure()
    fig.add_scatter(x=w["target_date"], y=w[f"high_h{horizon}"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=w["target_date"], y=w[f"low_h{horizon}"], line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(239,108,0,0.18)", name="80% range")
    fig.add_scatter(x=w["target_date"], y=w[TARGET_COL], name="Actual", line=dict(color="#F06292", width=2))
    fig.add_scatter(x=w["target_date"], y=w[f"pred_h{horizon}"], name="Forecast", line=dict(color="#EF6C00", dash="dash"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Weekly cases",
                      legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig)

# ---------------------------------------------------------------- tab 4: about
with tab_about:
    t = DEFAULT_THRESHOLDS
    st.markdown(f"""
**What goes in:** city, ward, a date, and how many weeks ahead (2, 3 or 4).
The app takes that ward's rain, humidity, temperature and reported cases for the 4 weeks
up to the date, plus ward traits (population density, water-logging risk, sanitation).

**Why weeks of delay:** rain fills drains and containers; Aedes eggs need 7–14 days to become
mosquitoes, then the virus incubates. In this data, cases follow rain most strongly about 3 weeks later.

**The model:** one LightGBM model per horizon (Poisson objective for case counts) plus two
quantile models for the 80% range. Drivers come from LightGBM's built-in TreeSHAP.

**Risk tiers (weekly cases per ward):** Low < {t['medium']:.0f} ≤ Medium < {t['high']:.0f} ≤ High
< {t['critical']:.0f} ≤ Critical.

**Cost:** models load once per server; normal forecasts are looked up from a precomputed table;
the model only runs live for what-if changes or new weather. No paid APIs and no LLM calls.
""")
    rt = metrics["runtime"]
    st.caption(
        f"This page: data + models ready in {load_ms:.0f} ms (cached), forecast in {pred_ms:.1f} ms "
        f"({fc.source}). Training all models: {rt['total_train_seconds']} s."
    )
