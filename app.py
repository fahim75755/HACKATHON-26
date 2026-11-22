"""Task 4 - dashboard for public health officers.

Run:  streamlit run app.py
The officer picks a place (a city or all cities, a ward or all wards) and a date. The page has one
tab each for 2, 3 and 4 weeks ahead:
- All wards: every ward in scope for that week - ranked list, ward risk map, counts, and what to do
  with that much warning time.
- One ward: only that ward for that week - expected cases, range, risk tier, what to do, how
  reliable that lead time is, the main drivers, and rain vs cases.
"""
import json

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

ALL_CITIES, ALL_WARDS = "All cities", "All wards"


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

models = get_models()
features, preds, metrics = get_tables()

data_start, data_end = features["date"].min(), features["date"].max()


def tier_badge(tier: str) -> str:
    return (f'<span style="background:{TIER_META[tier]["color"]};color:#fff;padding:2px 10px;'
            f'border-radius:12px;font-weight:700;font-size:0.85rem">{tier.upper()}</span>')


def n_cases(x: float) -> str:
    v = int(round(float(x)))
    return f"{v} case" if v == 1 else f"{v} cases"


def headline_box(color: str, title: str, lines: list[str], action: str) -> None:
    body = "".join(f'<div style="margin-top:4px">{x}</div>' for x in lines)
    st.markdown(
        f"""<div style="padding:14px 18px;border-radius:10px;border-left:8px solid {color};
        background:rgba(128,128,128,0.10);margin:4px 0 16px 0;">
        <div style="font-size:1.35rem;font-weight:800;color:{color}">{title}</div>{body}
        <div style="margin-top:8px"><b>Recommended action:</b> {action}</div></div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- sidebar: inputs only
st.sidebar.header("📍 Place and date")
cities = sorted(features["city"].unique())
city = st.sidebar.selectbox("City", [ALL_CITIES] + cities)
pool = features if city == ALL_CITIES else features[features["city"] == city]
wards = sorted(pool["ward_id"].unique())
ward = st.sidebar.selectbox(
    "Ward", [ALL_WARDS] + wards,
    help="Pick one ward for its detailed forecast, or All wards to see every ward ranked by risk.",
)
chosen_date = st.sidebar.date_input(
    "Forecast from week of",
    value=pd.Timestamp("2025-07-14").date(),
    min_value=data_start.date(),
    max_value=pd.Timestamp("2027-12-31").date(),
    help="Recorded data runs from %s to %s. For later dates, pick one ward and enter its recent weather."
    % (data_start.date(), data_end.date()),
)

st.sidebar.divider()
st.sidebar.header("🌧️ What-if")
extra_rain = st.sidebar.slider(
    "Extra rain in the last 1–2 days (mm)", 0, 200, 0, step=10,
    help="Rain that fell in the last day or two and is not in the weekly record yet. "
         "It is added to this week's rain in every ward shown. Guide (Bangladesh Meteorological Dept., "
         "per day): heavy 44–88 mm, very heavy 89 mm or more.",
)
what_if = extra_rain > 0

overview = ward == ALL_WARDS
issue_week = engine.snap_to_week(features, wards[0] if overview else ward, pd.Timestamp(chosen_date))
manual_mode = issue_week is None
if manual_mode:
    issue_week = pd.Timestamp(chosen_date)

# ---------------------------------------------------------------- header
st.title("🦟 Urban Dengue Early-Warning")
place = (f"{city} · {len(wards)} wards" if overview else f"{engine.ward_traits(features, ward)['city']} · {ward}")
st.caption(f"{place} · dengue risk for the 2–4 weeks after {issue_week:%d %b %Y}")
if what_if:
    st.warning(f"What-if active: {extra_rain} mm of extra rain in the last 1–2 days, added to this week's rain"
               f"{' in every ward' if overview else ''}. Set the slider back to 0 to see the forecast from "
               "recorded weather.")

# ---------------------------------------------------------------- forecasts for every ward in scope
# What officers can still do with each amount of warning time.
LEAD = {
    2: ("Respond now",
        "Send fogging and larvicide teams to the High and Critical wards below, alert their clinics and "
        "hospitals, and check beds, saline and test-kit stock."),
    3: ("Stop the breeding",
        "Drain water-logged spots, clear containers and blocked drains, and run clean-up drives in the wards "
        "below. Done now, this breaks the breeding cycle before these cases appear."),
    4: ("Plan ahead",
        "Order saline, test kits and larvicide for the wards below, schedule field teams, and brief hospitals "
        "on the expected load."),
}

week_tbl = None
if not manual_mode:
    frows = features[(features["date"] == issue_week) & features["ward_id"].isin(wards)]
    if what_if:
        week_tbl = engine.predict_table(models, frows, extra_rain)
    else:
        week_tbl = preds[(preds["date"] == issue_week) & preds["ward_id"].isin(wards)]
    week_tbl = week_tbl.merge(
        frows[["ward_id", TARGET_COL]].rename(columns={TARGET_COL: "cases_now_rec"}), on="ward_id")


@st.cache_data
def reliability() -> dict:
    """Per horizon, on the test year: MAE, tier accuracy, High/Critical weeks caught (alone and with watch)."""
    out = {}
    for h in HORIZONS:
        m = metrics["horizons"][f"h{h}"]["lightgbm"]
        d = preds[["ward_id", "date", f"high_h{h}"]].merge(
            features[["ward_id", "date", f"target_h{h}", f"target_date_h{h}"]], on=["ward_id", "date"])
        d = d[d[f"target_date_h{h}"].dt.year == metrics["test_year"]]
        actual_high = d[f"target_h{h}"] >= DEFAULT_THRESHOLDS["high"]
        flagged = d[f"high_h{h}"] >= DEFAULT_THRESHOLDS["high"]  # expected High, or the watch flag
        out[h] = {"mae": m["MAE"], "tier_acc": m["tier_accuracy"], "recall": m["high_critical_recall"],
                  "recall_watch": float((flagged & actual_high).sum() / max(int(actual_high.sum()), 1))}
    return out


def render_horizon(h: int) -> None:
    """All wards in scope for one lead time: which will be High/Critical in h weeks, and what to do now."""
    target = issue_week + pd.Timedelta(weeks=h)
    t = week_tbl.copy()
    t["cases"] = t[f"pred_h{h}"].astype(float)
    t["low"] = t[f"low_h{h}"].astype(float)
    t["high"] = t[f"high_h{h}"].astype(float)
    t["tier"] = [assign_risk_tier(v) for v in t["cases"]]
    t["watch"] = [engine.watch_tier(c, hi) or "" for c, hi in zip(t["cases"], t["high"])]
    t["rank"] = t["tier"].map(TIERS.index)
    t = t.sort_values(["rank", "cases"], ascending=False).reset_index(drop=True)

    n = len(t)
    counts = t["tier"].value_counts()
    n_crit, n_high, n_med, n_low = (int(counts.get(x, 0)) for x in ("Critical", "High", "Medium", "Low"))
    n_watch = int((t["watch"] != "").sum())
    top = t.iloc[0]
    when = f"the week of {target:%d %b %Y}"
    lines = []
    if n_crit or n_high:
        title = f"⚠ {n_crit + n_high} of {n} wards HIGH or CRITICAL in {when}"
        lines.append(f"Highest: <b>{top['ward_id']}</b> ({top['city']}), about {n_cases(top['cases'])}.")
        if n_watch:
            lines.append(f"{n_watch} more wards could reach High (watch).")
        color, action = TIER_META[top["tier"]]["color"], f"<b>{LEAD[h][0]}:</b> {LEAD[h][1]}"
    elif n_watch:
        title = f"⚠ WATCH - {n_watch} of {n} wards could reach HIGH in {when}"
        lines.append("Expected cases stay below High, but the likely range reaches it.")
        color, action = TIER_META["High"]["color"], f"<b>{LEAD[h][0]}:</b> {LEAD[h][1]}"
    elif n_med:
        title = f"MEDIUM RISK in {n_med} of {n} wards in {when}"
        color, action = TIER_META["Medium"]["color"], TIER_META["Medium"]["action"]
    else:
        title = f"✓ LOW RISK in all {n} wards in {when}"
        color, action = TIER_META["Low"]["color"], TIER_META["Low"]["action"]
    lines.append(f"<span style='opacity:0.85'>Expected: <b>{n_cases(t['cases'].sum())}</b> across these wards "
                 f"that week &nbsp;·&nbsp; now: <b>{n_cases(t['cases_now_rec'].sum())}</b></span>")
    headline_box(color, title, lines, action)

    r = reliability()[h]
    st.caption(f"ℹ️ How reliable is the {h}-week forecast? In {metrics['test_year']} it was off by "
               f"{r['mae']:.1f} cases a week on average, got the risk tier right {r['tier_acc']:.0%} of the time, "
               f"and caught {r['recall']:.0%} of High/Critical weeks ({r['recall_watch']:.0%} counting watch flags).")

    k = st.columns(5)
    boxes = [("Critical wards", n_crit), ("High wards", n_high), ("Watch", n_watch),
             ("Medium wards", n_med), ("Low wards", n_low)]
    for col, (label, val) in zip(k, boxes):
        col.metric(label, val, border=True,
                   help="Expected lower, but the likely range reaches High or Critical." if label == "Watch" else None)

    if city == ALL_CITIES:
        st.markdown("**By city**")
        by_city = t.groupby("city").agg(
            Wards=("ward_id", "size"),
            Critical=("tier", lambda s: int((s == "Critical").sum())),
            High=("tier", lambda s: int((s == "High").sum())),
            Watch=("watch", lambda s: int((s != "").sum())),
            Medium=("tier", lambda s: int((s == "Medium").sum())),
            Low=("tier", lambda s: int((s == "Low").sum())),
            **{"Cases now": ("cases_now_rec", "sum"), "Expected cases": ("cases", "sum")},
        ).reset_index().rename(columns={"city": "City"})
        by_city = by_city.sort_values(["Critical", "High", "Watch", "Expected cases"], ascending=False)
        st.dataframe(by_city.style.format({"Cases now": "{:.0f}", "Expected cases": "{:.0f}"}),
                     hide_index=True, key=f"city_{h}")

    st.markdown(f"**Ward risk map - {when}** (each box is a ward; the number is expected cases)")
    t["ward_no"] = t["ward_id"].str.split("_").str[-1]
    rows = sorted(t["city"].unique())[::-1]
    cols = sorted(t["ward_no"].unique())
    grid = {v: t.pivot(index="city", columns="ward_no", values=v).reindex(index=rows, columns=cols)
            for v in ("rank", "cases", "ward_id")}
    txt = [["" if pd.isna(x) else f"{x:.1f}" for x in row] for row in grid["cases"].to_numpy(dtype=float)]
    scale = []
    for i, c in enumerate(TIER_META[x]["color"] for x in TIERS):
        scale += [[i / len(TIERS), c], [(i + 1) / len(TIERS), c]]
    fig = go.Figure(go.Heatmap(
        z=grid["rank"].to_numpy(dtype=float), x=cols, y=rows, zmin=-0.5, zmax=len(TIERS) - 0.5,
        colorscale=scale, text=txt, texttemplate="%{text}", xgap=3, ygap=3, hoverongaps=False,
        customdata=grid["ward_id"].fillna("").to_numpy(dtype=object),
        hovertemplate="%{customdata}<br>expected %{text} cases<extra></extra>",
        colorbar=dict(tickvals=list(range(len(TIERS))), ticktext=TIERS, title="Tier"),
    ))
    fig.update_layout(height=110 + 50 * len(rows), margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(side="top", title="Ward number", type="category"))
    st.plotly_chart(fig, key=f"map_{h}")

    st.markdown(f"**Ranked list - act on the top rows first** ({n} wards)")
    table = pd.DataFrame({
        "City": t["city"], "Ward": t["ward_id"], "Risk": t["tier"],
        "Expected cases": t["cases"].round(1),
        "Likely range": t["low"].round().astype(int).astype(str) + "–" + t["high"].round().astype(int).astype(str),
        "Cases now": t["cases_now_rec"].round().astype(int),
        "Watch": t["watch"].map(lambda w: f"⚠ could reach {w}" if w else ""),
    })

    styled = table.style.map(
        lambda x: f"background-color:{TIER_META[x]['color']};color:white;font-weight:700", subset=["Risk"],
    ).format({"Expected cases": "{:.1f}"})
    st.dataframe(styled, hide_index=True, height=380, key=f"rank_{h}")
    st.download_button(
        "⬇️ Download this list (CSV)", table.to_csv(index=False).encode("utf-8"),
        file_name=f"dengue_wards_{h}wk_{target:%Y-%m-%d}.csv", mime="text/csv", key=f"dl_{h}",
    )


DOT = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}
# The same lead-time actions as LEAD, written for a single ward.
LEAD_WARD = {
    2: "Send fogging and larvicide teams to this ward, alert its clinics and hospitals, and check beds, "
       "saline and test-kit stock.",
    3: "Drain water-logged spots, clear containers and blocked drains, and run a clean-up drive in this ward. "
       "Done now, this breaks the breeding cycle before these cases appear.",
    4: "Order saline, test kits and larvicide for this ward, schedule field teams, and brief local hospitals.",
}


def ward_forecasts() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The selected ward's last 4 weeks, model input and 2/3/4-week forecasts.

    Uses recorded weather, recorded weather plus the extra-rain what-if, or (after the data ends)
    weather the officer enters in a table shown above the tabs.
    """
    traits = engine.ward_traits(features, ward)
    if manual_mode:
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
        weeks = engine.recent_weeks(features, ward, issue_week)
        if what_if:
            weeks = engine.add_recent_rain(weeks, extra_rain)
            X = engine.build_row(features, ward, issue_week, weeks, traits)
            fcs = engine.outlook(models, preds, features, ward, issue_week, X)
        else:
            X = features[(features["ward_id"] == ward) & (features["date"] == issue_week)]
            fcs = engine.outlook(models, preds, features, ward, issue_week)
    return weeks, X, {f.horizon: f for f in fcs}


def render_ward_week(h: int, weeks: pd.DataFrame, X: pd.DataFrame, fc) -> None:
    """One ward, one lead time: headline, what to do, how reliable, why, and the rain/cases picture."""
    now_cases = float(weeks[TARGET_COL].iloc[-1])
    now_tier = assign_risk_tier(now_cases)
    if fc.cases >= now_cases * 1.25 and fc.cases - now_cases >= 1:
        trend = "rising ↑"
    elif fc.cases <= now_cases * 0.8 and now_cases - fc.cases >= 1:
        trend = "falling ↓"
    else:
        trend = "about the same →"

    when = f"the week of {fc.target_date:%d %b %Y}"
    if fc.tier in ("High", "Critical"):
        title = f"⚠ {fc.tier.upper()} RISK in {when}"
        color, action = fc.color, f"<b>{LEAD[h][0]}:</b> {LEAD_WARD[h]}"
    elif fc.watch:
        title = f"⚠ WATCH - could reach {fc.watch.upper()} in {when}"
        color = TIER_META[fc.watch]["color"]
        action = f"{fc.action} <b>If it rises ({LEAD[h][0].lower()}):</b> {LEAD_WARD[h]}"
    elif fc.tier == "Medium":
        title = f"MEDIUM RISK in {when}"
        color, action = fc.color, fc.action
    else:
        title = f"✓ LOW RISK in {when}"
        color, action = fc.color, fc.action
    lines = [
        f"Expected about <b>{fc.cases:.1f} cases</b> (likely {fc.low:.0f}–{fc.high:.0f}), "
        f"{h} weeks after {fc.issue_date:%d %b}.",
        f"<span style='opacity:0.85'>Now: <b>{n_cases(now_cases)}</b> this week ({now_tier}) "
        f"&nbsp;·&nbsp; compared with now: <b>{trend}</b></span>",
    ]
    if fc.actual is not None and not what_if:
        lines.append(f"✔ Actually recorded that week: <b>{n_cases(fc.actual)}</b> ({assign_risk_tier(fc.actual)})")
    headline_box(color, title, lines, action)

    r = reliability()[h]
    st.caption(f"ℹ️ How reliable is the {h}-week forecast? In {metrics['test_year']} it was off by "
               f"{r['mae']:.1f} cases a week on average, got the risk tier right {r['tier_acc']:.0%} of the time, "
               f"and caught {r['recall']:.0%} of High/Critical weeks ({r['recall_watch']:.0%} counting watch flags).")
    if week_tbl is not None:
        order = week_tbl[f"pred_h{h}"].astype(float).rank(ascending=False, method="min")
        pos = int(order[week_tbl["ward_id"] == ward].iloc[0])
        scope_name = "all cities" if city == ALL_CITIES else city
        st.caption(f"📊 This ward ranks **#{pos} of {len(week_tbl)}** wards in {scope_name} for that week. "
                   "Pick *All wards* to see the full list.")

    left, right = st.columns([1, 1.15])
    with left:
        st.markdown(f"**Why** - what drives the {h}-week forecast")
        rain_sum = float(X["rain_sum_4w"].iloc[0])
        rain_normal = rain_sum - float(X["rain_anom_4w"].iloc[0])
        if rain_normal > 1:
            ratio = rain_sum / rain_normal
            word = "wetter than usual" if ratio > 1.15 else "drier than usual" if ratio < 0.85 else "about usual"
            st.caption(f"🌧️ Rain over the last 4 weeks: **{rain_sum:.0f} mm**, {ratio:.1f}× this ward's "
                       f"usual for this time of year ({word}).")
        drv, base = engine.drivers(models[h], X)
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
        st.plotly_chart(fig, key=f"drivers_{h}")
        st.caption("🌧️💧🌡️ weather · 🏘️ ward conditions · 📅 time of year · 🦟 recent cases")
    with right:
        st.markdown(f"**Rain and cases** - shaded weeks are the inputs; the diamond is the {h}-week forecast")
        hist = features[(features["ward_id"] == ward) & (features["date"] <= issue_week)].tail(20)
        fig = go.Figure()
        fig.add_bar(x=hist["date"], y=hist[RAIN_COL], name="Rain (mm)", marker_color="rgba(100,181,246,0.40)")
        if manual_mode or what_if:
            fig.add_bar(x=weeks["date"], y=weeks[RAIN_COL], name="Rain used (what-if / entered)",
                        marker_color="rgba(239,108,0,0.55)")
        fig.add_scatter(x=hist["date"], y=hist[TARGET_COL], name="Cases", mode="lines+markers",
                        line=dict(color="#F06292", width=3), yaxis="y2")
        future = features[(features["ward_id"] == ward) & (features["date"] > issue_week)
                          & (features["date"] <= fc.target_date)]
        if len(future) and not what_if:
            fig.add_scatter(x=future["date"], y=future[TARGET_COL], name="Cases (actual, later)",
                            mode="lines+markers", line=dict(color="#B0BEC5", dash="dot"), yaxis="y2")
        fig.add_scatter(
            x=[fc.target_date], y=[fc.cases], mode="markers", name=f"Forecast (+{h} wk)", yaxis="y2",
            marker=dict(color=fc.color, size=15, symbol="diamond"),
            error_y=dict(type="data", symmetric=False, array=[fc.high - fc.cases],
                         arrayminus=[fc.cases - fc.low], color="#9E9E9E"),
        )
        fig.add_vrect(x0=issue_week - pd.Timedelta(days=24), x1=issue_week + pd.Timedelta(days=3),
                      fillcolor="rgba(30,136,229,0.10)", line_width=0)
        fig.update_layout(
            height=360, margin=dict(l=10, r=10, t=10, b=10), barmode="overlay",
            yaxis=dict(title="Rain (mm)", side="right", showgrid=False, rangemode="tozero"),
            yaxis2=dict(title="Weekly cases", overlaying="y", side="left", rangemode="tozero"),
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig, key=f"rain_{h}")


def render_accuracy() -> None:
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
    scope_wards = wards if overview else [ward]
    label = (f"{city}, all {len(scope_wards)} wards added up" if overview else ward)
    st.markdown(f"**{label}: {vh}-week forecast vs actual through {metrics['test_year']}**")
    w_pred = preds[preds["ward_id"].isin(scope_wards)][["ward_id", "date", f"pred_h{vh}", f"low_h{vh}", f"high_h{vh}"]].copy()
    w_pred["target_date"] = w_pred["date"] + pd.Timedelta(weeks=vh)
    w_act = features[features["ward_id"].isin(scope_wards)][["ward_id", "date", TARGET_COL]]
    w = w_pred.merge(w_act, left_on=["ward_id", "target_date"], right_on=["ward_id", "date"], suffixes=("", "_t"))
    w = w[w["target_date"].dt.year == metrics["test_year"]]
    w = w.groupby("target_date", as_index=False)[[f"pred_h{vh}", f"low_h{vh}", f"high_h{vh}", TARGET_COL]].sum()
    fig = go.Figure()
    if not overview:  # quantile ranges do not add up across wards, so the band is shown for one ward only
        fig.add_scatter(x=w["target_date"], y=w[f"high_h{vh}"], line=dict(width=0), showlegend=False,
                        hoverinfo="skip")
        fig.add_scatter(x=w["target_date"], y=w[f"low_h{vh}"], line=dict(width=0), fill="tonexty",
                        fillcolor="rgba(239,108,0,0.18)", name="80% range")
    fig.add_scatter(x=w["target_date"], y=w[TARGET_COL], name="Actual", line=dict(color="#F06292", width=2))
    fig.add_scatter(x=w["target_date"], y=w[f"pred_h{vh}"], name="Forecast", line=dict(color="#EF6C00", dash="dash"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Weekly cases",
                      legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig)


# ---------------------------------------------------------------- layout
# Tabs are always 2 / 3 / 4 weeks ahead. The Ward input decides what is inside them:
# All wards -> every ward for that week; one ward -> only that ward for that week.
if not overview:
    w_weeks, w_X, w_fcs = ward_forecasts()  # for later dates this shows the weather-entry table first


def tab_label(h: int) -> str:
    label = f"⏱️ {h} weeks ahead · {issue_week + pd.Timedelta(weeks=h):%d %b}"
    if not overview:
        fc = w_fcs[h]
        label += f" · {DOT[fc.tier]} {fc.tier}" + (" ⚠" if fc.watch else "")
    return label


tabs = st.tabs([tab_label(h) for h in HORIZONS] + ["📊 Accuracy (for reviewers)"])
for h, tab in zip(HORIZONS, tabs):
    with tab:
        if not overview:
            render_ward_week(h, w_weeks, w_X, w_fcs[h])
        elif manual_mode:
            st.info(f"The all-wards view uses recorded data (up to {data_end:%d %b %Y}). "
                    "For later dates, pick one ward and enter its recent weather.")
        else:
            render_horizon(h)

with tabs[-1]:
    render_accuracy()
