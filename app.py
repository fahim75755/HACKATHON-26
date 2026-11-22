"""Task 4 - dashboard for public health officers.

Run:  streamlit run app.py
The officer picks a place (city, ward) and a date. The app shows the dengue forecast
for 2, 3 and 4 weeks later, the risk tier for each, an early-warning headline, the
recommended action, the main drivers, and which wards to act on first.
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
from dengue.tiers import DEFAULT_THRESHOLDS, TIER_META, TIERS, assign_risk_tier

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


def tier_badge(tier: str) -> str:
    return (f'<span style="background:{TIER_META[tier]["color"]};color:#fff;padding:2px 10px;'
            f'border-radius:12px;font-weight:700;font-size:0.85rem">{tier.upper()}</span>')


# ---------------------------------------------------------------- sidebar: inputs only
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
    help="Recorded data runs from %s to %s. For later dates you enter the recent weather yourself."
    % (data_start.date(), data_end.date()),
)

st.sidebar.divider()
st.sidebar.header("🌧️ What-if")
extra_rain = st.sidebar.slider(
    "Extra rain in the last 1–2 days (mm)", 0, 200, 0, step=10,
    help="Rain that fell in the last day or two and is not in the weekly record yet. "
         "It is added to this week's rain. Guide (Bangladesh Meteorological Dept., per day): "
         "heavy 44–88 mm, very heavy 89 mm or more.",
)
what_if = extra_rain > 0

issue_week = engine.snap_to_week(features, ward, pd.Timestamp(chosen_date))
manual_mode = issue_week is None
traits = engine.ward_traits(features, ward)

# ---------------------------------------------------------------- header
st.title("🦟 Urban Dengue Early-Warning")

# ---------------------------------------------------------------- build the 2/3/4-week outlook
t_pred = time.perf_counter()
if manual_mode:
    issue_week = pd.Timestamp(chosen_date)
    st.caption(f"{city} · {ward} · dengue risk for the 2–4 weeks after {issue_week:%d %b %Y}")
    st.info(
        f"No recorded weather for {issue_week:%d %b %Y} (data ends {data_end:%d %b %Y}). "
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
        "Week of": "date", "Rain (mm)": RAIN_COL, "Humidity (%)": HUMID_COL,
        "Avg temp (°C)": TEMP_COL, "Reported cases": TARGET_COL,
    })
    weeks = engine.add_recent_rain(weeks, extra_rain)
    X = engine.build_row(features, ward, issue_week, weeks, traits)
    fcs = engine.outlook(models, preds, features, ward, issue_week, X)
else:
    st.caption(f"{city} · {ward} · dengue risk for the 2–4 weeks after {issue_week:%d %b %Y}")
    weeks = engine.recent_weeks(features, ward, issue_week)
    if what_if:
        weeks = engine.add_recent_rain(weeks, extra_rain)
        X = engine.build_row(features, ward, issue_week, weeks, traits)
        fcs = engine.outlook(models, preds, features, ward, issue_week, X)
    else:
        X = features[(features["ward_id"] == ward) & (features["date"] == issue_week)]
        fcs = engine.outlook(models, preds, features, ward, issue_week)
pred_ms = (time.perf_counter() - t_pred) * 1000

if what_if:
    st.warning(f"What-if active: {extra_rain} mm of extra rain in the last 1–2 days, added to this week's "
               f"{float(weeks[RAIN_COL].iloc[-1]) - extra_rain:.0f} mm. Set the slider back to 0 to see the "
               "forecast from recorded weather.")

tab_ward, tab_map, tab_valid, tab_about = st.tabs(
    ["📍 Ward forecast", "🗺️ Wards to act on", "📊 Accuracy (for reviewers)", "ℹ️ How it works"]
)

# ---------------------------------------------------------------- tab 1: ward forecast
with tab_ward:
    now_cases = float(weeks[TARGET_COL].iloc[-1])
    now_tier = assign_risk_tier(now_cases)
    peak = max(fcs, key=lambda f: f.cases)
    worst = peak.tier
    watch_flags = [f.watch for f in fcs if f.watch and TIERS.index(f.watch) > TIERS.index(worst)]
    watch = max(watch_flags, key=TIERS.index) if watch_flags else None
    last = fcs[-1].cases
    if last >= now_cases * 1.25 and last - now_cases >= 1:
        trend = "rising ↑"
    elif last <= now_cases * 0.8 and now_cases - last >= 1:
        trend = "falling ↓"
    else:
        trend = "steady →"

    when = f"week of {peak.target_date:%d %b} ({peak.horizon} weeks from now)"
    if worst in ("High", "Critical"):
        title = f"⚠ {worst.upper()} RISK in the next 2–4 weeks"
        detail = f"Peak of about {peak.cases:.0f} cases in the {when}."
        color, action = TIER_META[worst]["color"], TIER_META[worst]["action"]
    elif watch:
        title = f"⚠ WATCH - could reach {watch.upper()} in the next 2–4 weeks"
        detail = (f"Expected {worst} (up to {peak.cases:.0f} cases), but the likely range reaches {watch}. "
                  "Keep teams ready and check again next week.")
        color = TIER_META[watch]["color"]
        action = f"{TIER_META[worst]['action']} <b>If it rises:</b> {TIER_META[watch]['action']}"
    elif worst == "Medium":
        title = "MEDIUM RISK in the next 2–4 weeks"
        detail = f"Peak of about {peak.cases:.0f} cases in the {when}."
        color, action = TIER_META[worst]["color"], TIER_META[worst]["action"]
    else:
        title = "✓ LOW RISK - no outbreak expected in the next 4 weeks"
        detail = ("Under 1 case a week expected." if peak.cases < 1
                  else f"At most about {peak.cases:.0f} cases a week.")
        color, action = TIER_META[worst]["color"], TIER_META[worst]["action"]

    st.markdown(
        f"""<div style="padding:14px 18px;border-radius:10px;border-left:8px solid {color};
        background:rgba(128,128,128,0.10);margin:4px 0 16px 0;">
        <div style="font-size:1.35rem;font-weight:800;color:{color}">{title}</div>
        <div style="margin-top:4px">{detail}</div>
        <div style="margin-top:4px;opacity:0.85">Now: <b>{now_cases:.0f} cases</b> this week
        ({now_tier}) &nbsp;·&nbsp; trend over the next 4 weeks: <b>{trend}</b></div>
        <div style="margin-top:8px"><b>Recommended action:</b> {action}</div></div>""",
        unsafe_allow_html=True,
    )

    cols = st.columns(len(fcs))
    for col, fc in zip(cols, fcs):
        with col.container(border=True):
            st.markdown(f"**In {fc.horizon} weeks** · week of {fc.target_date:%d %b %Y}")
            st.markdown(
                f'<div style="font-size:2.2rem;font-weight:800;line-height:1.1">{fc.cases:.1f}'
                f'<span style="font-size:1rem;font-weight:400"> cases</span></div>'
                f'<div style="margin:6px 0">{tier_badge(fc.tier)}</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Likely range (80%): {fc.low:.0f} – {fc.high:.0f} cases")
            if fc.watch:
                st.markdown(f":orange[**⚠ Could reach {fc.watch}**]")
            if fc.actual is not None and not what_if:
                st.caption(f"Actually recorded: **{fc.actual:.0f} cases** ({assign_risk_tier(fc.actual)})")

    left, right = st.columns([1, 1.15])
    with left:
        st.markdown(f"**Why** - what drives the {peak.horizon}-week forecast (the highest of the three)")
        rain_sum = float(X["rain_sum_4w"].iloc[0])
        rain_normal = rain_sum - float(X["rain_anom_4w"].iloc[0])
        if rain_normal > 1:
            ratio = rain_sum / rain_normal
            word = "wetter than usual" if ratio > 1.15 else "drier than usual" if ratio < 0.85 else "about usual"
            st.caption(f"🌧️ Rain over the last 4 weeks: **{rain_sum:.0f} mm**, {ratio:.1f}× this ward's "
                       f"usual for this time of year ({word}).")
        drv, base = engine.drivers(models[peak.horizon], X)
        icons = {"Rain": "🌧️", "Unusually": "🌧️", "Humidity": "💧", "Temperature": "🌡️", "Season": "📅",
                 "Cases": "🦟", "Population": "🏘️", "Water": "🏘️", "Sanitation": "🏘️"}
        drv["label"] = [next((i for k, i in icons.items() if d.startswith(k)), "") + " " + d for d in drv["driver"]]
        drv = drv.iloc[::-1]
        colors = ["#C62828" if v > 0 else "#2E7D32" for v in drv["effect_cases"]]
        span = max(drv["effect_cases"].abs().max(), 0.5)
        fig = go.Figure(go.Bar(
            x=drv["effect_cases"], y=drv["label"], orientation="h", marker_color=colors,
            text=[f"{v:+.1f}" for v in drv["effect_cases"]], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:+.2f} cases<extra></extra>",
        ))
        fig.update_layout(
            height=360, margin=dict(l=10, r=40, t=10, b=10),
            xaxis=dict(title=f"Cases added (red) / removed (green) vs a typical week ({base:.1f})",
                       range=[-span * 1.25 if (drv["effect_cases"] < 0).any() else 0, span * 1.25]),
        )
        st.plotly_chart(fig)
        st.caption("🌧️💧🌡️ weather · 🏘️ ward conditions · 📅 time of year · 🦟 recent cases")
    with right:
        st.markdown("**Rain and cases** - shaded weeks are the inputs; diamonds are the 2, 3 and 4-week forecasts")
        hist = features[(features["ward_id"] == ward) & (features["date"] <= issue_week)].tail(20)
        fig = go.Figure()
        fig.add_bar(x=hist["date"], y=hist[RAIN_COL], name="Rain (mm)", marker_color="rgba(100,181,246,0.40)")
        if manual_mode or what_if:
            fig.add_bar(x=weeks["date"], y=weeks[RAIN_COL], name="Rain used (what-if / entered)",
                        marker_color="rgba(239,108,0,0.55)")
        fig.add_scatter(x=hist["date"], y=hist[TARGET_COL], name="Cases", mode="lines+markers",
                        line=dict(color="#F06292", width=3), yaxis="y2")
        future = features[(features["ward_id"] == ward) & (features["date"] > issue_week)
                          & (features["date"] <= fcs[-1].target_date)]
        if len(future) and not what_if:
            fig.add_scatter(x=future["date"], y=future[TARGET_COL], name="Cases (actual, later)",
                            mode="lines+markers", line=dict(color="#B0BEC5", dash="dot"), yaxis="y2")
        fig.add_scatter(
            x=[f.target_date for f in fcs], y=[f.cases for f in fcs], mode="markers", name="Forecast",
            yaxis="y2", marker=dict(color=[f.color for f in fcs], size=14, symbol="diamond"),
            error_y=dict(type="data", symmetric=False, array=[f.high - f.cases for f in fcs],
                         arrayminus=[f.cases - f.low for f in fcs], color="#9E9E9E"),
        )
        fig.add_vrect(x0=issue_week - pd.Timedelta(days=24), x1=issue_week + pd.Timedelta(days=3),
                      fillcolor="rgba(30,136,229,0.10)", line_width=0)
        fig.update_layout(
            height=360, margin=dict(l=10, r=10, t=10, b=10), barmode="overlay",
            yaxis=dict(title="Rain (mm)", side="right", showgrid=False, rangemode="tozero"),
            yaxis2=dict(title="Weekly cases", overlaying="y", side="left", rangemode="tozero"),
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig)

# ---------------------------------------------------------------- tab 2: wards to act on
with tab_map:
    if manual_mode:
        st.info(f"This view uses recorded data. Pick a date up to {data_end:%d %b %Y}.")
    else:
        st.subheader(f"Wards to act on - forecasts made {issue_week:%d %b %Y} for the next 2–4 weeks")
        if what_if:
            st.caption("Uses recorded weather for every ward (the what-if only changes the selected ward).")
        allw = engine.all_wards_outlook(preds, issue_week)
        k = st.columns(5)
        for col, tier in zip(k, reversed(TIERS)):
            col.metric(f"{tier} wards", int((allw["worst_tier"] == tier).sum()), border=True,
                       help=f"Wards whose highest forecast in the next 2–4 weeks is {tier}.")
        k[4].metric("Watch", int(allw["watch"].notna().sum()), border=True,
                    help="Expected lower, but the likely range reaches High or Critical.")

        st.markdown("**Ranked list - act on the top rows first** (all 60 wards)")
        table = pd.DataFrame({
            "City": allw["city"], "Ward": allw["ward_id"],
            "Highest risk": allw["worst_tier"],
            "Peak cases": allw["peak_cases"].round(1),
            "Peak week": allw["peak_week"].dt.strftime("%d %b %Y"),
            **{f"In {h} wk": allw[f"pred_h{h}"].round(1) for h in HORIZONS},
            "Watch": allw["watch"].map(lambda w: f"⚠ could reach {w}" if isinstance(w, str) else ""),
        })
        styled = table.style.map(
            lambda t: f"background-color:{TIER_META[t]['color']};color:white;font-weight:700",
            subset=["Highest risk"],
        ).format({c: "{:.1f}" for c in ["Peak cases"] + [f"In {h} wk" for h in HORIZONS]})
        st.dataframe(styled, hide_index=True, height=380)

        st.markdown(f"**Ward heatmap - {city}**: risk tier for each of the next 2–4 weeks")
        cw = allw[allw["city"] == city].sort_values("peak_cases")
        z = np.array([[TIERS.index(t) for t in cw[f"tier_h{h}"]] for h in HORIZONS]).T
        txt = [[f"{v:.1f}" for v in r] for r in cw[[f"pred_h{h}" for h in HORIZONS]].to_numpy(dtype=float)]
        colors = [TIER_META[t]["color"] for t in TIERS]
        scale = []
        for i, c in enumerate(colors):
            scale += [[i / len(colors), c], [(i + 1) / len(colors), c]]
        fig = go.Figure(go.Heatmap(
            z=z, x=[f"In {h} weeks<br>{issue_week + pd.Timedelta(weeks=h):%d %b}" for h in HORIZONS],
            y=cw["ward_id"], zmin=-0.5, zmax=len(TIERS) - 0.5, colorscale=scale,
            text=txt, texttemplate="%{text}", xgap=2, ygap=2,
            hovertemplate="%{y}<br>%{x}<br>expected %{text} cases<extra></extra>",
            colorbar=dict(tickvals=list(range(len(TIERS))), ticktext=TIERS, title="Tier"),
        ))
        fig.update_layout(height=max(300, 26 * len(cw) + 90), margin=dict(l=10, r=10, t=10, b=10),
                          xaxis=dict(side="top"))
        st.plotly_chart(fig)

# ---------------------------------------------------------------- tab 3: accuracy (reviewers)
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

    vh = 3
    st.markdown(f"**{ward}: {vh}-week forecast vs actual through {metrics['test_year']}**")
    w_pred = preds[preds["ward_id"] == ward][["date", f"pred_h{vh}", f"low_h{vh}", f"high_h{vh}"]].copy()
    w_pred["target_date"] = w_pred["date"] + pd.Timedelta(weeks=vh)
    w_act = features[features["ward_id"] == ward][["date", TARGET_COL]]
    w = w_pred.merge(w_act, left_on="target_date", right_on="date", suffixes=("", "_t"))
    w = w[w["target_date"].dt.year == metrics["test_year"]]
    fig = go.Figure()
    fig.add_scatter(x=w["target_date"], y=w[f"high_h{vh}"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=w["target_date"], y=w[f"low_h{vh}"], line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(239,108,0,0.18)", name="80% range")
    fig.add_scatter(x=w["target_date"], y=w[TARGET_COL], name="Actual", line=dict(color="#F06292", width=2))
    fig.add_scatter(x=w["target_date"], y=w[f"pred_h{vh}"], name="Forecast", line=dict(color="#EF6C00", dash="dash"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Weekly cases",
                      legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig)

# ---------------------------------------------------------------- tab 4: about
with tab_about:
    t = DEFAULT_THRESHOLDS
    st.markdown(f"""
**What goes in:** city, ward and a date. The app takes that ward's rain, humidity, temperature
and reported cases for the 4 weeks up to the date, plus ward traits (population density,
water-logging risk, sanitation).

**What comes out:** expected cases and risk tier for **2, 3 and 4 weeks later**, all made on the same
day from the same data, plus a headline, the recommended action and the main drivers.

**Why weeks of delay:** rain fills drains and containers; Aedes eggs need 7–14 days to become
mosquitoes, then the virus incubates. In this data, cases follow rain most strongly about 3 weeks later.

**The model:** one LightGBM model per horizon (Poisson objective for case counts) plus two
quantile models for the 80% range. Drivers come from LightGBM's built-in TreeSHAP.

**Risk tiers (weekly cases per ward):** Low < {t['medium']:.0f} ≤ Medium < {t['high']:.0f} ≤ High
< {t['critical']:.0f} ≤ Critical.

**Watch flag:** shown when the expected cases are below High but the top of the 80% range reaches
High or Critical. On 2025 data this catches 91–95% of High/Critical weeks, compared with 51–63%
from the expected value alone, at the cost of more false alarms.

**Cost:** models load once per server; normal forecasts are looked up from a precomputed table;
the model only runs live for the extra-rain what-if or newly entered weather. No paid APIs and no LLM calls.

**What-if, extra rain in the last 1–2 days:** rain that fell after the weekly record was made is added to this
week's rain, and all three forecasts are recomputed live. Its biggest effect is on the 3 and 4-week forecasts,
because new breeding water needs that long to turn into cases.
""")
    rt = metrics["runtime"]
    st.caption(
        f"This page: data + models ready in {load_ms:.0f} ms (cached), 3 forecasts in {pred_ms:.1f} ms "
        f"({fcs[0].source}). Training all models: {rt['total_train_seconds']} s."
    )
