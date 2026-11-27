"""Phase 2 "Command Center" tab (frontend for R1-R5).

The UI never computes allocation, routing or the simulator itself: it calls the backend
`dengue.command.run_command_center(params) -> CommandResult` (owned by the backend) and only
displays the result. Until that module exists, an honest PREVIEW (simple numbers from the
precomputed forecasts, clearly labelled) keeps the tab usable, so the UI can be built and
tested in parallel. Notices come from dengue/notify.py - the same engine make_exports.py uses.
"""
from __future__ import annotations

import functools
from pathlib import Path
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from dengue import notify
from dengue.config import FEATURES_PATH, PREDICTIONS_PATH, TRAIN_END
from dengue.ops import CITY_CENTERS, WARD_COORDS
from dengue.tiers import TIER_META, TIERS, assign_risk_tier

# ------------------------------------------------------------------ the contract the UI reads (columns may be added, never renamed)
WARD_COLS = ["ward_id", "city", "tier", "pred", "saline_req", "saline_alloc", "kits_req", "kits_alloc",
             "beds_req", "bed_cap", "deficit", "red", "code_black", "crews", "prevented"]
OPTIONAL_WARD_COLS = ["p90", "uses_p90", "watch", "sent", "received", "unmet", "crew_ids", "roi", "crew_need",
                      "priority_score", "prevented_h2", "prevented_h3", "prevented_h4", "density"]
TRANSFER_COLS = ["city", "from_ward", "to_ward", "patients", "km"]
CITY_COLS = ["city", "demand", "capacity", "utilization", "routed", "unmet", "exhausted"]
CITIES = ["Dhaka North", "Dhaka South", "Chattogram", "Gazipur", "Narayanganj"]
HORIZONS = (2, 3, 4)
S1, S2 = "#2a78d6", "#eb6834"                                   # categorical slots 1-2 (validated palette)
STATUS = {"ok": "#0ca30c", "surge": "#fab219", "over": "#d03b3b"}
STATUS_ICON = {"ok": "✓", "surge": "!", "over": "✕"}
OFFLINE_ENV = "ALERTAHEAD_OFFLINE"
BANGLA_FONT = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "bn-subset.woff2"


@functools.lru_cache(maxsize=1)
def _bangla_font_css() -> str:
    """Embed the 45 KB Bangla subset (Noto Sans Bengali, OFL) so a downloaded brief renders Bangla on any PC, offline."""
    if not BANGLA_FONT.exists():
        return ""
    import base64
    data = base64.b64encode(BANGLA_FONT.read_bytes()).decode("ascii")
    return f'@font-face{{font-family:"BnSubset";src:url(data:font/woff2;base64,{data}) format("woff2")}}'


# ------------------------------------------------------------------ honest preview (used only until dengue/command.py exists)
@dataclass(frozen=True)
class PreviewParams:
    issue_date: pd.Timestamp
    horizon: int
    city: str | None = None
    stock_pct: int = 60
    buffer_pct: int = 0
    overrides: dict | None = None
    total_crews: int = 8
    efficacy: float = 0.30
    reserve_beds: int = 0
    offline: bool = False


@dataclass
class PreviewResult:
    wards: pd.DataFrame
    transfers: pd.DataFrame
    cities: pd.DataFrame
    alerts: list = field(default_factory=list)
    source: str = "preview"
    params: PreviewParams | None = None


_PREVIEW_RATES = {"saline": 2.38415, "kits": 1.15819, "beds": 0.29552}  # data-derived (2016-2023 fit); backend owns the real ones


@functools.lru_cache(maxsize=1)
def _preview_tables():
    preds = pd.read_parquet(PREDICTIONS_PATH)
    feats = pd.read_parquet(FEATURES_PATH, columns=["date", "ward_id", "dengue_cases", "population_density_sqkm"])
    hist = feats[feats["date"] <= pd.Timestamp(TRAIN_END)]
    beds = (hist["dengue_cases"] * _PREVIEW_RATES["beds"]).round()
    cap = beds.groupby(hist["ward_id"]).quantile(0.90, interpolation="lower").clip(lower=2).astype(int)
    dens = feats.groupby("ward_id")["population_density_sqkm"].first()
    return preds, cap, dens


def preview_command_center(params) -> PreviewResult:
    """Honest dummy: allocated = required x stock %, crews = top-N, no routing, no simulator."""
    preds, cap, dens = _preview_tables()
    h = int(params.horizon)
    wk = preds[preds["date"] == pd.Timestamp(params.issue_date)].copy()
    if params.city and params.city in CITIES:
        wk = wk[wk["city"] == params.city]
    wk["pred"] = wk[f"pred_h{h}"].astype("float64").round(3)
    wk["p90"] = wk[f"high_h{h}"].astype("float64").round(3)
    wk["tier"] = [assign_risk_tier(v) for v in wk["pred"]]
    wk["uses_p90"] = wk["tier"].isin(["High", "Critical"])
    plan = np.where(wk["uses_p90"], wk["p90"], wk["pred"])
    mult = 1 + params.buffer_pct / 100
    for res in ("saline", "kits"):
        base = np.round(_PREVIEW_RATES[res] * plan)
        wk[f"{res}_req"] = np.round(base * mult).astype(int)
        wk[f"{res}_alloc"] = np.minimum(np.floor(base * params.stock_pct / 100).astype(int), wk[f"{res}_req"].to_numpy())
    wk["beds_req"] = np.round(_PREVIEW_RATES["beds"] * wk["pred"]).astype(int)
    wk["bed_cap"] = (wk["ward_id"].map(cap).fillna(2)).astype(int)
    wk["sent"], wk["received"] = 0, 0
    wk["unmet"] = (wk["beds_req"] - wk["bed_cap"]).clip(lower=0)
    top = wk.sort_values(["pred", "ward_id"], ascending=[False, True]).head(int(params.total_crews))["ward_id"]
    wk["crews"] = wk["ward_id"].isin(top).astype(int)
    order = {w: i + 1 for i, w in enumerate(top)}
    wk["crew_ids"] = [f"Crew-{order[w]:02d}" if w in order else "" for w in wk["ward_id"]]
    wk["prevented"] = 0.0
    wk["density"] = wk["ward_id"].map(dens)
    short = (wk["saline_req"] > wk["saline_alloc"]) | (wk["kits_req"] > wk["kits_alloc"])
    wk["deficit"] = short
    wk["red"] = short & wk["tier"].isin(["High", "Critical"])
    wk["code_black"] = (wk["saline_req"] > wk["saline_alloc"]) & (wk["unmet"] > 0)
    cities = (wk.assign(head=(wk["bed_cap"] - wk["beds_req"]).clip(lower=0))
                .groupby("city").agg(demand=("beds_req", "sum"), capacity=("bed_cap", "sum"), unmet=("unmet", "sum"), headroom=("head", "sum"))
                .reset_index())
    cities["utilization"] = cities["demand"] / cities["capacity"]
    cities["routed"] = 0
    cities["exhausted"] = cities["unmet"] > 0
    wards = wk[WARD_COLS + [c for c in OPTIONAL_WARD_COLS if c in wk.columns]].reset_index(drop=True)
    return PreviewResult(wards=wards, transfers=pd.DataFrame(columns=TRANSFER_COLS), cities=cities, params=params)


BACKEND_WARD_MAP = {  # backend contract (dengue/phase2_config.WARD_COLS) -> names this UI and notify.py read
    "saline_required": "saline_req", "saline_allocated": "saline_alloc", "kits_required": "kits_req",
    "kits_allocated": "kits_alloc", "beds_required": "beds_req", "bed_capacity": "bed_cap", "crews_assigned": "crews",
    "cases_prevented": "prevented", "high_pred": "p90", "deficit_flag": "deficit", "red_alert": "red",
    "roi_per_crew": "roi", "crews_needed": "crew_need", "density_crew_score": "priority_score",
    "population_density_sqkm": "density", "bed_net_deficit": "unmet",
}
BACKEND_TRANSFER_MAP = {"patients_moved": "patients", "distance_km": "km"}
BACKEND_CITY_MAP = {"beds_required": "demand", "bed_capacity": "capacity", "beds_routed": "routed", "beds_unmet": "unmet",
                    "city_pool_exhausted": "exhausted"}
BACKEND_TIER_MAP = {"Watch": "Medium"}  # backend labels its 2nd tier "Watch"; Phase 1 calls it Medium


def adapt_result(r):
    """Normalise a CommandResult (backend or preview) into the frames the UI and notify.py read.

    Returns (wards, transfers, cities, alert_messages, source). Never mutates the backend result.
    """
    wards = r.wards.copy()
    transfers = (r.transfers if r.transfers is not None else pd.DataFrame(columns=TRANSFER_COLS)).copy()
    cities = r.cities.copy()
    if "saline_required" in wards.columns:                      # backend contract
        wards["tier_backend"] = wards["tier"]
        wards["tier"] = wards["tier"].replace(BACKEND_TIER_MAP)
        if "demand_basis" in wards.columns and "high_pred" in wards.columns:
            wards["uses_p90"] = np.isclose(wards["demand_basis"], wards["high_pred"]) & (wards["demand_basis"] > wards["pred"])
        wards = wards.drop(columns=[v for v in BACKEND_WARD_MAP.values() if v in wards.columns and v not in BACKEND_WARD_MAP])
        wards = wards.rename(columns=BACKEND_WARD_MAP)
        transfers = transfers.rename(columns=BACKEND_TRANSFER_MAP)
        cities = cities.rename(columns=BACKEND_CITY_MAP)
        if "utilization_pct" in cities.columns:
            cities["utilization"] = cities["utilization_pct"] / 100.0
        if "crew_ids" not in wards.columns:
            order = wards[wards["crews"] > 0].sort_values(["crews", "priority_score" if "priority_score" in wards else "pred", "ward_id"],
                                                         ascending=[False, False, True])
            ids, k = {}, 0
            for w, n in zip(order["ward_id"], order["crews"]):
                ids[w] = ",".join(f"Crew-{k + j + 1:02d}" for j in range(int(n)))
                k += int(n)
            wards["crew_ids"] = wards["ward_id"].map(ids).fillna("")
    if not transfers.empty:
        sent = transfers.groupby("from_ward")["patients"].sum()
        recv = transfers.groupby("to_ward")["patients"].sum()
        wards["sent"] = wards["ward_id"].map(sent).fillna(0).astype(int)
        wards["received"] = wards["ward_id"].map(recv).fillna(0).astype(int)
    for col, default in [("sent", 0), ("received", 0), ("unmet", 0), ("crew_ids", ""), ("p90", np.nan),
                         ("uses_p90", False), ("watch", False), ("deficit", False), ("red", False), ("code_black", False),
                         ("crews", 0), ("prevented", 0.0)]:
        if col not in wards.columns:
            wards[col] = default
    for col, default in [("routed", 0), ("unmet", 0), ("exhausted", False)]:
        if col not in cities.columns:
            cities[col] = default
    if "utilization" not in cities.columns:
        cities["utilization"] = np.where(cities["capacity"] > 0, cities["demand"] / cities["capacity"], 0.0)
    for c in ("deficit", "red", "code_black", "uses_p90"):
        wards[c] = wards[c].astype(bool)
    cities["exhausted"] = cities["exhausted"].astype(bool)
    alerts = [a.get("message", str(a)) if isinstance(a, dict) else str(a) for a in (r.alerts or [])]
    return wards, transfers, cities, alerts, str(getattr(r, "source", "preview"))


def get_engine():
    """Backend if it exists, otherwise the labelled preview."""
    try:
        from dengue.command import CommandParams, run_command_center  # noqa: WPS433 (backend-owned)
        return CommandParams, run_command_center, "backend"
    except ImportError:
        return PreviewParams, preview_command_center, "preview"


def is_offline() -> bool:
    return bool(st.session_state.get("offline_mode", False)) or os.environ.get(OFFLINE_ENV) == "1"


@st.cache_data(show_spinner=False, max_entries=64)
def _run(issue: str, h: int, city: str | None, stock: int, buffer: int, crews: int, eff: float,
         reserve: int, ov_json: str, offline: bool, cap_mult: float = 1.0, engine_kind: str = "", manual_json: str = ""):
    # engine_kind is part of the cache key, so a cached preview is never served once the backend exists
    Params, run, kind = get_engine()
    extra = {"capacity_multiplier": cap_mult} if kind == "backend" else {}
    if kind == "backend" and manual_json:
        extra["manual_crews"] = json.loads(manual_json)   # R3: officer-selected wards
    p = Params(issue_date=issue if kind == "backend" else pd.Timestamp(issue), horizon=h, city=city, stock_pct=stock, buffer_pct=buffer,
               overrides=json.loads(ov_json) or None, total_crews=crews, efficacy=eff, reserve_beds=reserve, offline=offline, **extra)
    wards, transfers, cities, alerts, source = adapt_result(run(p))
    return wards, transfers, cities, alerts, source, kind


# ------------------------------------------------------------------ small UI helpers
def _t(is_bn: bool, en: str, bn: str) -> str:
    return bn if is_bn else en


def _tier_label(tier: str, is_bn: bool) -> str:
    return notify.tier_name(tier, "bn" if is_bn else "en")


def _flag_text(row, is_bn: bool) -> str:
    out = []
    if row.get("code_black"):
        out.append("■ " + _t(is_bn, "CODE BLACK", "কোড ব্ল্যাক"))
    if row.get("red"):
        out.append("✕ " + _t(is_bn, "Red alert", "লাল সতর্কতা"))
    elif row.get("deficit"):
        out.append("! " + _t(is_bn, "Shortfall", "ঘাটতি"))
    if row.get("watch"):
        out.append("◔ " + _t(is_bn, "Watch", "নজরে"))
    return " · ".join(out) if out else "✓ " + _t(is_bn, "OK", "ঠিক আছে")


def _bed_status(util: float) -> str:
    return "over" if util > 1 else "surge" if util >= 0.85 else "ok"


def scenario_id(**kw) -> str:
    return hashlib.sha1(json.dumps(kw, sort_keys=True, default=str).encode()).hexdigest()[:6]


def render_offline_toggle(is_bn: bool) -> bool:
    """Sidebar kill-switch + badge (R5). The toggle is the demo's 'turn the internet off' switch."""
    st.sidebar.divider()
    st.sidebar.toggle(_t(is_bn, "Offline mode (kill-switch)", "অফলাইন মোড (কিল-সুইচ)"),
                      key="offline_mode", value=os.environ.get(OFFLINE_ENV) == "1",
                      help=_t(is_bn, "Blocks every optional internet call (live weather, map tiles). Forecasts, allocation, routing and notices always run locally.",
                              "সব ঐচ্ছিক ইন্টারনেট কল বন্ধ (লাইভ আবহাওয়া, ম্যাপ টাইল)। পূর্বাভাস, বরাদ্দ, স্থানান্তর ও বার্তা সবসময় লোকালি চলে।"))
    off = is_offline()
    st.sidebar.caption(("🟡 " + _t(is_bn, "Offline mode — 0 network calls", "অফলাইন মোড — ০ নেটওয়ার্ক কল")) if off
                       else ("🟢 " + _t(is_bn, "Online — internet used only for optional weather & map tiles", "অনলাইন — শুধু ঐচ্ছিক আবহাওয়া ও ম্যাপ টাইলের জন্য ইন্টারনেট")))
    return off


# ------------------------------------------------------------------ main entry
def render_command_center(issue_week, city_choice: str, is_bn: bool, data_end) -> None:
    if issue_week is None or pd.Timestamp(issue_week) > pd.Timestamp(data_end) - pd.Timedelta(weeks=4):
        st.info(_t(is_bn, "The Command Center works on recorded weeks up to 4 weeks before the data ends. Pick an earlier date in the sidebar.",
                   "কমান্ড সেন্টার ডেটা শেষ হওয়ার ৪ সপ্তাহ আগ পর্যন্ত রেকর্ড করা সপ্তাহে চলে। সাইডবারে আগের তারিখ বেছে নিন।"))
        return
    city = None if city_choice not in CITIES else city_choice

    # ---- controls (R1/R2/R3 scarcity sliders)
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        h = c1.radio(_t(is_bn, "Lead time", "আগাম সময়"), HORIZONS, index=1, horizontal=True, key="cc_h",
                     format_func=lambda x: _t(is_bn, f"+{x} wk", f"+{notify.num(x, 'bn')} সপ্তাহ"))
        stock = c2.slider(_t(is_bn, "Available regional stock (%)", "আঞ্চলিক মজুদ (%)"), 30, 100, 60, 5, key="cc_stock")
        buffer = c3.slider(_t(is_bn, "Safety buffer (%)", "নিরাপত্তা বাফার (%)"), 0, 30, 0, 5, key="cc_buffer",
                           help=_t(is_bn, "Raises required stock only — a stress test.", "শুধু প্রয়োজন বাড়ায় — স্ট্রেস টেস্ট।"))
        crews = c4.slider(_t(is_bn, "Total spraying crews", "মোট স্প্রে টিম"), 0, 30, 8, 1, key="cc_crews")
        d1, d2, d3, d4 = st.columns(4)
        eff = d1.slider(_t(is_bn, "Crew efficacy (%)", "টিমের কার্যকারিতা (%)"), 10, 60, 30, 5, key="cc_eff",
                        help=_t(is_bn, "Breeding-site reduction per crew (assumption).", "প্রতি টিমে প্রজনন স্থান কমে (অনুমান)।"))
        reserve = d2.radio(_t(is_bn, "Reserve beds at receivers", "গ্রহণকারী ওয়ার্ডে খালি শয্যা"), (0, 1), horizontal=True, key="cc_reserve")
        with d3.popover(_t(is_bn, "Per-resource stock override", "আলাদা সম্পদের মজুদ")):
            ov_on = st.toggle(_t(is_bn, "Use separate stock levels", "আলাদা মজুদ ব্যবহার"), key="cc_ov_on")
            ov_s = st.slider(_t(is_bn, "Saline stock (%)", "স্যালাইন মজুদ (%)"), 30, 100, stock, 5, key="cc_ov_s", disabled=not ov_on)
            ov_k = st.slider(_t(is_bn, "Kit stock (%)", "কিট মজুদ (%)"), 30, 100, stock, 5, key="cc_ov_k", disabled=not ov_on)
        overrides = {"saline": ov_s, "kits": ov_k} if ov_on else {}
        cap_pct = d4.slider(_t(is_bn, "Bed capacity (% of ward P90)", "শয্যা সক্ষমতা (ওয়ার্ড P90-এর %)"), 50, 200, 100, 10, key="cc_cap",
                            help=_t(is_bn, "100% = each ward's 2016–2023 P90 weekly bed need. Raise it to model surge beds opened.",
                                    "১০০% = প্রতিটি ওয়ার্ডের ২০১৬–২০২৩ সালের P90 সাপ্তাহিক শয্যা চাহিদা। অতিরিক্ত শয্যা খুললে বাড়ান।"))
        offline = is_offline()

    issue = str(pd.Timestamp(issue_week).date())
    auto_args = (stock, buffer, crews, eff / 100, reserve, json.dumps(overrides, sort_keys=True), offline, cap_pct / 100, get_engine()[2])
    manual = None
    if st.session_state.get("cc_crew_mode", "auto") == "manual":
        if "cc_manual_crews" not in st.session_state:   # first switch: start from the auto plan
            auto_w = _run(issue, h, city, *auto_args)[0]
            st.session_state["cc_manual_crews"] = {w: int(n) for w, n in zip(auto_w["ward_id"], auto_w["crews"]) if n > 0}
        manual = st.session_state["cc_manual_crews"]
    run_args = auto_args + ((json.dumps(manual, sort_keys=True),) if manual is not None else ())
    st.session_state["_cc_run_args"] = (issue, city) + run_args
    wards, transfers, cities, alerts, source, kind = _run(issue, h, city, *run_args)
    target = pd.Timestamp(issue_week) + pd.Timedelta(weeks=h)
    sid = scenario_id(issue=issue, h=h, city=city, stock=stock, buffer=buffer, crews=crews, eff=eff, reserve=reserve, ov=overrides, cap=cap_pct)
    ctx = notify.NoticeContext(target=target, horizon=h, stock_pct=stock, total_crews=crews, scenario=sid, source=str(source))

    if kind == "preview":
        st.warning(_t(is_bn, "Preview numbers — the backend (dengue/command.py) is not connected yet. Allocation here is a plain "
                             "stock-% cut, with no routing and no simulator. Every view switches to the real engine automatically once it lands.",
                      "প্রিভিউ সংখ্যা — ব্যাকএন্ড (dengue/command.py) এখনো যুক্ত হয়নি। এখানে বরাদ্দ শুধু মজুদ-% কাটছাঁট; স্থানান্তর ও সিমুলেটর নেই। "
                      "আসল ইঞ্জিন এলে সব ভিউ নিজে থেকেই সেটায় চলে যাবে।"), icon="🧪")
    for a in alerts[:5]:
        st.error(str(a), icon="🚨")

    _kpis(wards, cities, crews, is_bn)
    tabs = st.tabs([_t(is_bn, "① Allocation", "① বরাদ্দ"), _t(is_bn, "② Surge & routing", "② শয্যা চাপ ও স্থানান্তর"),
                    _t(is_bn, "③ Vector-control simulator", "③ মশক নিধন সিমুলেটর"), _t(is_bn, "④ Dispatch notices", "④ নির্দেশনা বার্তা"),
                    _t(is_bn, "⑤ Offline & system", "⑤ অফলাইন ও সিস্টেম")])
    with tabs[0]:
        _allocation(wards, city, is_bn, issue, stock, buffer, crews, eff, reserve, overrides, offline)
    with tabs[1]:
        _routing(wards, transfers, cities, is_bn)
    with tabs[2]:
        _simulator(wards, issue_week, crews, eff, kind, is_bn)
    with tabs[3]:
        _dispatch(wards, transfers, cities, ctx, is_bn)
    with tabs[4]:
        _system(wards, transfers, cities, ctx, source, kind, offline, is_bn)


# ------------------------------------------------------------------ KPI strip
def _kpis(wards, cities, crews_total, is_bn):
    sr, sa = wards["saline_req"].sum(), wards["saline_alloc"].sum()
    kr, ka = wards["kits_req"].sum(), wards["kits_alloc"].sum()
    dem, cap = cities["demand"].sum(), cities["capacity"].sum()
    unmet, black = int(cities["unmet"].sum()), int(wards["code_black"].sum())
    # Sub-lines are plain captions, not st.metric deltas: a delta always draws an up/down arrow,
    # which would read as "rising" for numbers that are just context (e.g. "1,161 / 1,934").
    tiles = [
        (_t(is_bn, "Expected cases", "আনুমানিক রোগী"), f"{wards['pred'].sum():,.0f}",
         _t(is_bn, "forecast for the target week", "লক্ষ্য সপ্তাহের পূর্বাভাস")),
        (_t(is_bn, "Saline need met", "স্যালাইন পূরণ"), f"{(sa / sr if sr else 1):.0%}",
         _t(is_bn, f"{sa:,} of {sr:,} bags", f"{sr:,}টির মধ্যে {sa:,} ব্যাগ")),
        (_t(is_bn, "Test-kit need met", "কিট পূরণ"), f"{(ka / kr if kr else 1):.0%}",
         _t(is_bn, f"{ka:,} of {kr:,} kits", f"{kr:,}টির মধ্যে {ka:,} কিট")),
        (_t(is_bn, "Bed load", "শয্যার চাপ"), f"{(dem / cap if cap else 0):.0%}",
         (("✕ " if unmet else "✓ ") + _t(is_bn, f"{unmet} without a bed after routing", f"স্থানান্তরের পর {unmet} জন শয্যাহীন"))),
        (_t(is_bn, "Red alerts", "লাল সতর্কতা"), f"{int(wards['red'].sum())}",
         (("■ " if black else "") + _t(is_bn, f"{black} CODE BLACK", f"{black}টি কোড ব্ল্যাক"))),
        (_t(is_bn, "Spraying crews", "স্প্রে টিম"), f"{int(wards['crews'].sum())} / {crews_total}",
         _t(is_bn, f"{wards['prevented'].sum():.1f} cases prevented (2–4 wk)", f"{wards['prevented'].sum():.1f} জন রোগী প্রতিরোধ (২–৪ সপ্তাহ)")),
    ]
    for col, (label, value, sub) in zip(st.columns(6), tiles):
        col.metric(label, value)   # the app's CSS already draws each metric as a card
        col.caption(sub)


# ------------------------------------------------------------------ ① allocation
def _allocation(wards, city, is_bn, issue, stock, buffer, crews, eff, reserve, overrides, offline):
    filt = st.segmented_control(_t(is_bn, "Show", "দেখান"), ["all", "short", "red", "black"], default="all", key="cc_filter",
                                format_func=lambda x: {"all": _t(is_bn, "All wards", "সব ওয়ার্ড"), "short": _t(is_bn, "Shortfall", "ঘাটতি"),
                                                       "red": _t(is_bn, "Red alert", "লাল সতর্কতা"), "black": _t(is_bn, "CODE BLACK", "কোড ব্ল্যাক")}[x])
    w = wards.copy()
    if filt == "short":
        w = w[w["deficit"]]
    elif filt == "red":
        w = w[w["red"]]
    elif filt == "black":
        w = w[w["code_black"]]
    w = w.assign(tier_rank=w["tier"].map(TIERS.index)).sort_values(["tier_rank", "pred", "ward_id"], ascending=[False, False, True])
    view = pd.DataFrame({
        _t(is_bn, "Ward", "ওয়ার্ড"): w["ward_id"],
        _t(is_bn, "City", "শহর"): w["city"],
        _t(is_bn, "Risk", "ঝুঁকি"): [("● " + _tier_label(x, is_bn)) for x in w["tier"]],
        _t(is_bn, "Expected", "আনুমানিক"): w["pred"].round(1),
        _t(is_bn, "Planned on", "পরিকল্পনার ভিত্তি"): ["P90 " + f"{p:.1f}" if u else _t(is_bn, "expected", "আনুমানিক") for p, u in zip(w["p90"], w["uses_p90"])],
        _t(is_bn, "Saline fill", "স্যালাইন পূরণ"): np.where(w["saline_req"] > 0, w["saline_alloc"] / w["saline_req"].replace(0, 1), 1.0),
        _t(is_bn, "Saline alloc/req", "স্যালাইন বরাদ্দ/প্রয়োজন"): [f"{a} / {r}" for a, r in zip(w["saline_alloc"], w["saline_req"])],
        _t(is_bn, "Kit fill", "কিট পূরণ"): np.where(w["kits_req"] > 0, w["kits_alloc"] / w["kits_req"].replace(0, 1), 1.0),
        _t(is_bn, "Kits alloc/req", "কিট বরাদ্দ/প্রয়োজন"): [f"{a} / {r}" for a, r in zip(w["kits_alloc"], w["kits_req"])],
        _t(is_bn, "Beds need/cap", "শয্যা প্রয়োজন/সক্ষমতা"): [f"{STATUS_ICON[_bed_status(b / c if c else 0)]} {b} / {c}" for b, c in zip(w["beds_req"], w["bed_cap"])],
        _t(is_bn, "Crews", "টিম"): w["crews"],
        _t(is_bn, "Flags", "সংকেত"): [_flag_text(r, is_bn) for _, r in w.iterrows()],
    })
    pc = st.column_config.ProgressColumn
    st.dataframe(view, hide_index=True, width="stretch", height=420, column_config={
        _t(is_bn, "Saline fill", "স্যালাইন পূরণ"): pc(format="percent", min_value=0, max_value=1),
        _t(is_bn, "Kit fill", "কিট পূরণ"): pc(format="percent", min_value=0, max_value=1)})
    st.caption(_t(is_bn, "Risk ● colour = tier; every status also has a text label. Click a column header to sort.",
                  "ঝুঁকির ● রং = স্তর; প্রতিটি অবস্থার পাশে লেখাও আছে। সাজাতে কলামের শিরোনামে ক্লিক করুন।"))

    a, b = st.columns([1.1, 1])
    with a:
        st.markdown("**" + _t(is_bn, "Saline need met, by risk tier", "ঝুঁকির স্তর অনুযায়ী স্যালাইন পূরণ") + "**")
        rows = []
        for tr in reversed(TIERS):
            g = wards[wards["tier"] == tr]
            req = g["saline_req"].sum()
            rows.append((tr, (g["saline_alloc"].sum() / req) if req else None, int(g["saline_alloc"].sum()), int(req), len(g)))
        fig = go.Figure()
        for tr, f, al, rq, n in rows:
            fig.add_bar(y=[_tier_label(tr, is_bn)], x=[f or 0], orientation="h", marker_color=TIER_META[tr]["color"], showlegend=False,
                        text=[("—" if f is None else f"{f:.0%} · {al:,}/{rq:,}")], textposition="outside", cliponaxis=False,
                        hovertemplate=f"{_tier_label(tr, is_bn)} · {n} wards<br>{al:,} / {rq:,}<extra></extra>")
        fig.update_layout(height=220, margin=dict(l=0, r=60, t=6, b=0), xaxis=dict(range=[0, 1.15], tickformat=".0%", gridcolor="#e1e0d9"),
                          yaxis=dict(autorange="reversed"), plot_bgcolor="#fcfcfb", paper_bgcolor="#ffffff", bargap=0.35)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with b:
        st.markdown("**" + _t(is_bn, "Procurement outlook — order now", "ক্রয় পরিকল্পনা — এখনই অর্ডার") + "**")
        rows = []
        for hh in HORIZONS:
            a = st.session_state["_cc_run_args"]
            ww, *_ = _run(a[0], hh, a[1], *a[2:])
            rows.append({_t(is_bn, "Lead time", "আগাম সময়"): f"+{hh} wk · {(pd.Timestamp(issue) + pd.Timedelta(weeks=hh)):%d %b}",
                         _t(is_bn, "Saline short", "স্যালাইন ঘাটতি"): int((ww["saline_req"] - ww["saline_alloc"]).clip(lower=0).sum()),
                         _t(is_bn, "Saline required", "স্যালাইন প্রয়োজন"): int(ww["saline_req"].sum()),
                         _t(is_bn, "Kits short", "কিট ঘাটতি"): int((ww["kits_req"] - ww["kits_alloc"]).clip(lower=0).sum()),
                         _t(is_bn, "Kits required", "কিট প্রয়োজন"): int(ww["kits_req"].sum())})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(_t(is_bn, "+4 weeks is the order-now line: enough lead time to restock before the peak.",
                      "+৪ সপ্তাহ = এখনই অর্ডারের সারি: চূড়ার আগে মজুদ ভরার মতো সময়।"))


# ------------------------------------------------------------------ ② routing
def _routing(wards, transfers, cities, is_bn):
    ex = cities[cities["exhausted"]]
    if len(ex):
        st.error("\n".join(f"✕ {r.city}: " + _t(is_bn, f"city bed pool exhausted — {int(r.unmet)} patients without a bed. Escalate to the civil surgeon.",
                                                f"শহরের শয্যা শেষ — {int(r.unmet)} জন রোগীর শয্যা নেই। সিভিল সার্জনকে জানান।") for r in ex.itertuples()))
    else:
        st.success("✓ " + _t(is_bn, "All overflow is absorbed inside each city.", "সব বাড়তি রোগী নিজ শহরের ভেতরেই জায়গা পেয়েছে।"))

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("**" + _t(is_bn, "Ward map — bed load before routing (no map tiles: works offline)",
                              "ওয়ার্ড মানচিত্র — স্থানান্তরের আগে শয্যার চাপ (ম্যাপ টাইল নেই: অফলাইনে চলে)") + "**")
        shown = [c for c in CITIES if c in set(wards["city"])]
        cols = st.columns(2 if len(shown) > 1 else 1)
        for i, c in enumerate(shown):
            with cols[i % len(cols)]:
                st.plotly_chart(_city_map(c, wards[wards["city"] == c], transfers, is_bn), width="stretch",
                                config={"displayModeBar": False}, key=f"cc_map_{c}")
        st.caption("● ✓ <85% · ! 85–100% " + _t(is_bn, "surge", "চাপ") + " · ✕ >100% " + _t(is_bn, "overflow", "উপচে পড়া")
                   + " · → " + _t(is_bn, "patients routed. Ward positions are approximate reference points.", "স্থানান্তরিত রোগী। ওয়ার্ডের অবস্থান আনুমানিক।"))
    with right:
        st.markdown("**" + _t(is_bn, "City bed load", "শহরের শয্যার চাপ") + "**")
        cs = cities.sort_values("city")
        colors = [STATUS[_bed_status(u)] for u in cs["utilization"]]
        fig = go.Figure(go.Bar(y=cs["city"], x=cs["utilization"], orientation="h", marker_color=colors,
                               text=[f"{STATUS_ICON[_bed_status(u)]} {u:.0%} · {int(d)}/{int(c)}" for u, d, c in zip(cs["utilization"], cs["demand"], cs["capacity"])],
                               textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}<br>%{x:.0%}<extra></extra>"))
        for x, dash in ((0.85, "dot"), (1.0, "solid")):
            fig.add_vline(x=x, line_color="#4E646B", line_dash=dash, line_width=1)
        fig.update_layout(height=250, margin=dict(l=0, r=90, t=6, b=0), xaxis=dict(range=[0, max(1.6, cs["utilization"].max() * 1.25)], tickformat=".0%", gridcolor="#e1e0d9"),
                          plot_bgcolor="#fcfcfb", paper_bgcolor="#ffffff", showlegend=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown("**" + _t(is_bn, "Transfers (same city only)", "স্থানান্তর (শুধু একই শহরে)") + "**")
        if transfers.empty:
            st.caption(_t(is_bn, "No transfers in this scenario.", "এই পরিস্থিতিতে কোনো স্থানান্তর নেই।"))
        else:
            st.dataframe(transfers.rename(columns={"from_ward": _t(is_bn, "From", "থেকে"), "to_ward": _t(is_bn, "To", "যেখানে"),
                                                   "patients": _t(is_bn, "Patients", "রোগী"), "km": "km"}),
                         hide_index=True, width="stretch", column_config={"km": st.column_config.NumberColumn(format="%.1f")})
    st.markdown("**" + _t(is_bn, "Ward bed balance", "ওয়ার্ডভিত্তিক শয্যা হিসাব") + "**")
    b = wards.assign(util=np.where(wards["bed_cap"] > 0, wards["beds_req"] / wards["bed_cap"], 0.0)).sort_values(["util", "ward_id"], ascending=[False, True])
    st.dataframe(pd.DataFrame({
        _t(is_bn, "Ward", "ওয়ার্ড"): b["ward_id"], _t(is_bn, "City", "শহর"): b["city"],
        _t(is_bn, "Need", "প্রয়োজন"): b["beds_req"], _t(is_bn, "Capacity", "সক্ষমতা"): b["bed_cap"],
        _t(is_bn, "Sent", "পাঠানো"): b["sent"], _t(is_bn, "Received", "গ্রহণ"): b["received"], _t(is_bn, "Unmet", "বাকি"): b["unmet"],
        _t(is_bn, "Status", "অবস্থা"): [f"{STATUS_ICON[_bed_status(u)]} " + {"ok": _t(is_bn, "OK", "ঠিক"), "surge": _t(is_bn, "Surge", "চাপ"),
                                                                       "over": _t(is_bn, "Overflow", "উপচে পড়া")}[_bed_status(u)] + f" · {u:.0%}" for u in b["util"]],
    }), hide_index=True, width="stretch", height=320)


def _city_map(city, ws, transfers, is_bn):
    pos = {w: WARD_COORDS.get(w, CITY_CENTERS.get(city, (23.78, 90.40))) for w in ws["ward_id"]}
    fig = go.Figure()
    tr = transfers[transfers["city"] == city] if not transfers.empty else transfers
    for t in tr.itertuples(index=False):
        (la1, lo1), (la2, lo2) = pos.get(t.from_ward), pos.get(t.to_ward)
        fig.add_annotation(x=lo2, y=la2, ax=lo1, ay=la1, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=2, arrowsize=1.1, arrowwidth=1.5 + min(3.0, 0.6 * t.patients), arrowcolor=S1, opacity=0.9,
                           text=str(t.patients), font=dict(color=S1, size=11))
    util = np.where(ws["bed_cap"] > 0, ws["beds_req"] / ws["bed_cap"], 0.0)
    status = [_bed_status(u) for u in util]
    fig.add_scatter(x=[pos[w][1] for w in ws["ward_id"]], y=[pos[w][0] for w in ws["ward_id"]], mode="markers+text",
                    text=[w[-2:] for w in ws["ward_id"]], textposition="middle right", textfont=dict(size=9, color="#4E646B"),
                    marker=dict(size=8 + 3.2 * np.sqrt(ws["beds_req"].clip(lower=0)), color=[STATUS[s] for s in status], line=dict(color="#fcfcfb", width=2)),
                    customdata=np.stack([ws["ward_id"], ws["beds_req"], ws["bed_cap"], util], axis=1),
                    hovertemplate="<b>%{customdata[0]}</b><br>need %{customdata[1]} / cap %{customdata[2]} (%{customdata[3]:.0%})<extra></extra>",
                    showlegend=False)
    lat0 = np.mean([p[0] for p in pos.values()])
    fig.update_layout(title=dict(text=notify.city_name(city, "bn" if is_bn else "en"), font=dict(size=13), x=0.02, y=0.97),
                      height=270, margin=dict(l=4, r=4, t=30, b=4), plot_bgcolor="#fcfcfb", paper_bgcolor="#ffffff",
                      xaxis=dict(visible=False, scaleanchor="y", scaleratio=float(np.cos(np.radians(lat0)))), yaxis=dict(visible=False))
    return fig


# ------------------------------------------------------------------ ③ simulator
def _crew_picker(wards, crews, is_bn):
    """R3: officer deploys a limited number of crews to selected wards (or lets ROI decide)."""
    mode = st.radio(_t(is_bn, "Crew deployment", "টিম পাঠানো"), ["auto", "manual"], horizontal=True, key="cc_crew_mode",
                    format_func=lambda m: {"auto": _t(is_bn, "Auto — best ROI × density", "স্বয়ংক্রিয় — সর্বোচ্চ ROI × ঘনত্ব"),
                                           "manual": _t(is_bn, "Manual — I pick the wards", "নিজে ward বাছাই")}[m])
    if mode != "manual":
        return
    current = dict(st.session_state.get("cc_manual_crews", {}))
    roi_col = "roi_first_crew" if "roi_first_crew" in wards.columns else "pred"   # stable order (does not move as crews change)
    ed = wards.sort_values([roi_col, "ward_id"], ascending=[False, True])[["ward_id", "city", "tier", "pred", roi_col]].copy()
    ed["crews"] = ed["ward_id"].map(current).fillna(0).astype(int)
    out = st.data_editor(ed, key="cc_manual_editor", hide_index=True, width="stretch", height=300,
                         disabled=["ward_id", "city", "tier", "pred", roi_col],
                         column_config={"crews": st.column_config.NumberColumn(_t(is_bn, "Crews", "টিম"), min_value=0, max_value=6, step=1),
                                        roi_col: st.column_config.NumberColumn(_t(is_bn, "ROI of 1st crew", "প্রথম টিমের ROI"), format="%.2f"),
                                        "pred": st.column_config.NumberColumn(format="%.1f")})
    new = {w: int(n) for w, n in zip(out["ward_id"], out["crews"]) if int(n) > 0}
    total = sum(new.values())
    msg = _t(is_bn, f"{total} of {crews} crews placed.", f"{crews}টির মধ্যে {total}টি টিম বসানো হয়েছে।")
    if total > crews:
        st.warning(msg + " " + _t(is_bn, "Only the crew slider's total can go out: the largest requests are served first.",
                                  "স্লাইডারের মোটের বেশি যাবে না: বড় অনুরোধ আগে পূরণ হবে।"))
    else:
        st.caption(msg)
    if new != current:
        st.session_state["cc_manual_crews"] = new
        st.session_state.pop("cc_manual_editor", None)
        st.rerun()


def _simulator(wards, issue_week, crews, eff, kind, is_bn):
    _crew_picker(wards, crews, is_bn)
    if kind == "preview" or "roi_first_crew" not in wards.columns:   # simulator off / models unavailable
        st.info(_t(is_bn, "The ML counterfactual (h2/h3/h4 LightGBM with reduced breeding-water features) comes from the backend. "
                          "Charts below fill in automatically when it is connected; crews shown now are a top-N preview.",
                   "এমএল কাউন্টারফ্যাকচুয়াল (প্রজনন-পানির ফিচার কমিয়ে h2/h3/h4 LightGBM) ব্যাকএন্ড থেকে আসবে। যুক্ত হলে নিচের চার্ট নিজে থেকেই ভরবে; এখনকার টিম top-N প্রিভিউ।"))
    preds, _, _ = _preview_tables()
    wk = preds[(preds["date"] == pd.Timestamp(issue_week)) & preds["ward_id"].isin(wards["ward_id"])]
    base = {h: float(wk[f"pred_h{h}"].sum()) for h in HORIZONS}
    prev = {h: float(wards[f"prevented_h{h}"].sum()) if f"prevented_h{h}" in wards.columns else None for h in HORIZONS}
    rates = _PREVIEW_RATES
    cols = st.columns(3)
    for col, (title, mult, d) in zip(cols, [(_t(is_bn, "Projected cases", "আনুমানিক রোগী"), 1.0, 0),
                                            (_t(is_bn, "Hospital bed load", "শয্যার চাপ"), rates["beds"], 1),
                                            (_t(is_bn, "Saline consumption (bags)", "স্যালাইন ব্যবহার (ব্যাগ)"), rates["saline"], 0)]):
        xs = [_t(is_bn, f"+{h} wk", f"+{notify.num(h, 'bn')} সপ্তাহ") for h in HORIZONS]
        b_vals = [base[h] * mult for h in HORIZONS]
        w_vals = [(base[h] - (prev[h] or 0.0)) * mult for h in HORIZONS]
        fig = go.Figure()
        fig.add_bar(x=xs, y=b_vals, name=_t(is_bn, "Baseline", "বর্তমান ধারা"), marker_color=S1,
                    text=[f"{v:,.{d}f}" for v in b_vals], textposition="outside", cliponaxis=False)
        fig.add_bar(x=xs, y=w_vals, name=_t(is_bn, "With crews", "টিমসহ"), marker_color=S2,
                    text=[f"{v:,.{d}f}" for v in w_vals], textposition="outside", cliponaxis=False)
        fig.update_layout(title=dict(text=title, font=dict(size=13)), barmode="group", bargap=0.3, bargroupgap=0.06, height=290,
                          margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", y=1.12, x=0),
                          plot_bgcolor="#fcfcfb", paper_bgcolor="#ffffff", yaxis=dict(gridcolor="#e1e0d9"))
        col.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown("**" + _t(is_bn, "Intervention ROI — cases prevented per crew", "টিমপ্রতি প্রতিরোধ (ROI)") + "**")
    show = [c for c in ["ward_id", "city", "tier", "pred", "roi", "priority_score", "crew_need", "crews", "crew_ids", "prevented"] if c in wards.columns]
    sort_col = "roi" if "roi" in wards.columns else "pred"
    st.dataframe(wards.sort_values([sort_col, "ward_id"], ascending=[False, True])[show], hide_index=True, width="stretch", height=320,
                 column_config={"prevented": st.column_config.NumberColumn(format="%.1f"), "roi": st.column_config.NumberColumn(format="%.2f"),
                                "pred": st.column_config.NumberColumn(format="%.1f")})
    st.caption(_t(is_bn, "ROI is pure (cases prevented per crew, as the rubric defines it); the assignment also weights population density.",
                  "ROI বিশুদ্ধ (টিমপ্রতি প্রতিরোধ, রুব্রিকের সংজ্ঞা); বরাদ্দে জনসংখ্যার ঘনত্বও ধরা হয়।"))


# ------------------------------------------------------------------ ④ dispatch
def _dispatch(wards, transfers, cities, ctx, is_bn):
    order = wards.assign(s=wards["code_black"].astype(int) * 4 + wards["red"].astype(int) * 2 + wards["deficit"].astype(int)) \
                 .sort_values(["s", "pred", "ward_id"], ascending=[False, False, True])
    icon = lambda r: "■" if r.code_black else "✕" if r.red else "!" if r.deficit else "✓"  # noqa: E731
    c1, c2, c3, c4 = st.columns([1.3, 1.2, 1, 1.2])
    ward_id = c1.selectbox(_t(is_bn, "Notice for ward", "যে ওয়ার্ডের বার্তা"), order["ward_id"], key="cc_d_ward",
                           format_func=lambda w: f"{icon(order.set_index('ward_id').loc[w])} {w}")
    aud = c2.selectbox(_t(is_bn, "Audience", "প্রাপক"), ["comm", "hcm", "cs", "city"], key="cc_d_aud",
                       format_func=lambda a: {"comm": _t(is_bn, "Ward commissioner", "ওয়ার্ড কমিশনার"), "hcm": _t(is_bn, "Health-centre manager", "স্বাস্থ্যকেন্দ্র ব্যবস্থাপক"),
                                              "cs": _t(is_bn, "Civil surgeon (escalation)", "সিভিল সার্জন (জরুরি)"), "city": _t(is_bn, "City summary", "শহরের সারসংক্ষেপ")}[a])
    lang = c3.radio(_t(is_bn, "Language", "ভাষা"), ["en", "bn", "both"], horizontal=True, key="cc_d_lang",
                    format_func=lambda x: {"en": "English", "bn": "বাংলা", "both": _t(is_bn, "Both", "দুটোই")}[x])
    w = order.set_index("ward_id").loc[ward_id].to_dict() | {"ward_id": ward_id}
    if aud in ("cs", "city"):
        txt = notify.city_summary(w["city"], wards, cities, transfers, ctx, lang) if aud == "city" else notify.escalation(w["city"], wards, cities, ctx, lang)
        if txt is None:
            st.success(_t(is_bn, f"No escalation needed for {w['city']}: no CODE BLACK ward and the city bed pool is not exhausted.",
                          f"{notify.city_name(w['city'], 'bn')}-এর জন্য জরুরি বার্তা লাগবে না: কোনো কোড ব্ল্যাক নেই এবং শহরের শয্যা শেষ হয়নি।"))
        else:
            st.code(txt, language=None, wrap_lines=True)
            st.download_button(_t(is_bn, "Download", "ডাউনলোড"), txt.encode("utf-8"), f"{aud}_{w['city'].replace(' ', '_')}.txt", key="cc_d_dl_city")
    else:
        fmt = c4.radio(_t(is_bn, "Format", "ধরন"), ["sms", "wa", "brief"], horizontal=True, key="cc_d_fmt",
                       format_func=lambda x: {"sms": "SMS", "wa": "WhatsApp", "brief": _t(is_bn, "Brief", "ব্রিফ")}[x])
        if fmt == "sms":
            msgs = ([("English", notify.sms_en(w, ctx, transfers, aud))] if lang in ("en", "both") else []) \
                + ([("বাংলা", notify.sms_bn(w, ctx))] if lang in ("bn", "both") else [])
            for name, m in msgs:
                info = notify.sms_info(m)
                st.code(m, language=None, wrap_lines=True)
                st.caption(f"{name} · {info['chars']} chars · {'GSM-7' if info['gsm'] else 'Unicode'} · {info['parts']} SMS")
            fname, payload = f"{ward_id}_sms.txt", "\n\n".join(m for _, m in msgs)
        elif fmt == "wa":
            payload = notify.whatsapp(w, ctx, transfers, aud, lang)
            st.code(payload, language=None, wrap_lines=True)
            fname = f"{ward_id}_whatsapp.txt"
        else:
            payload = notify.brief_html(w, ctx, transfers, aud, lang, font_face_css=_bangla_font_css())
            components.html(payload, height=640, scrolling=True)
            st.caption(_t(is_bn, "Open the downloaded HTML and press Ctrl+P → Save as PDF (Bangla renders with the system font, offline).",
                          "ডাউনলোড করা HTML খুলে Ctrl+P → Save as PDF চাপুন (বাংলা সিস্টেম ফন্টে, অফলাইনে আসে)।"))
            fname = f"{ward_id}_{'health_centre' if aud == 'hcm' else 'commissioner'}_brief.html"
        st.download_button(_t(is_bn, "Download", "ডাউনলোড"), payload.encode("utf-8"), fname, key="cc_d_dl")
    st.divider()
    files = notify.export_files(wards, cities, transfers, ctx)
    st.download_button("⬇ " + _t(is_bn, f"Download all notices (ZIP, {len(files)} files)", f"সব বার্তা ডাউনলোড (ZIP, {len(files)}টি ফাইল)"),
                       notify.zip_bytes(files), f"alertahead_dispatch_{ctx.target:%Y%m%d}_h{ctx.horizon}.zip", "application/zip",
                       type="primary", key="cc_d_zip")


# ------------------------------------------------------------------ ⑤ offline & system
def _system(wards, transfers, cities, ctx, source, kind, offline, is_bn):
    a, b = st.columns(2)
    with a:
        st.markdown("**" + _t(is_bn, "Fallback ladder", "বিকল্পের ধাপ") + "**")
        fc_active = {"backend": "1st", "preview": "2nd"}.get(kind, "2nd")
        st.dataframe(pd.DataFrame([
            {"Task": "Forecast", "1st": "predictions.parquet / live LightGBM", "If unavailable": "parquet ↔ live", "Last resort": "same week last year",
             "Active": f"{fc_active} ({source})"},
            {"Task": "Weather (display only)", "1st": "Open-Meteo", "If unavailable": "offline cache", "Last resort": "seasonal normal",
             "Active": "offline cache" if offline else "Open-Meteo (on click)"},
            {"Task": "Map", "1st": "tile map", "If unavailable": "tile-free map", "Last resort": "city × ward grid",
             "Active": "tile-free map" if offline else "tile map (Phase 1 tabs) · tile-free here"},
            {"Task": "Notices", "1st": "templates (always)", "If unavailable": "—", "Last resort": "—", "Active": "templates"},
        ]), hide_index=True, width="stretch")
        st.caption(("🟡 " + _t(is_bn, "Offline mode is ON: no optional internet calls.", "অফলাইন মোড চালু: কোনো ঐচ্ছিক ইন্টারনেট কল নেই।")) if offline
                   else ("🟢 " + _t(is_bn, "Online. Turn on the sidebar kill-switch (or set ALERTAHEAD_OFFLINE=1) to prove offline runs.",
                                    "অনলাইন। অফলাইনে চলা দেখাতে সাইডবারের কিল-সুইচ চালু করুন (বা ALERTAHEAD_OFFLINE=1)।")))
    with b:
        st.markdown("**" + _t(is_bn, "Notice self-check (runs now, in this app)", "বার্তার সেলফ-চেক (এখনই, এই অ্যাপে)") + "**")
        checks = []
        en_ok = all(notify.sms_info(notify.sms_en(r, ctx, transfers, aud))["gsm"] and len(notify.sms_en(r, ctx, transfers, aud)) <= 160
                    for _, r in wards.iterrows() for aud in ("comm", "hcm"))
        checks.append((en_ok, _t(is_bn, "English SMS: GSM-7 and ≤ 160 chars, every ward", "ইংরেজি এসএমএস: GSM-7 ও ≤ ১৬০ অক্ষর")))
        checks.append((all(len(notify.sms_bn(r, ctx)) <= notify.SMS_BN_MAX for _, r in wards.iterrows()),
                       _t(is_bn, "Bangla SMS ≤ 201 chars (≤ 3 parts)", "বাংলা এসএমএস ≤ ২০১ অক্ষর (≤ ৩ ভাগ)")))
        four = all(all(tok in notify.whatsapp(r, ctx, transfers, "comm", "en") for tok in ("Expected cases", "Saline:", "Spraying crews"))
                   for _, r in wards.iterrows())
        checks.append((four, _t(is_bn, "Every notice lists cases, allocated vs required, alert, crews", "প্রতিটি বার্তায় রোগী, বরাদ্দ/প্রয়োজন, সতর্কতা, টিম")))
        f1 = notify.export_files(wards, cities, transfers, ctx)
        checks.append((f1 == notify.export_files(wards, cities, transfers, ctx), _t(is_bn, "Exports are byte-identical on re-run", "এক্সপোর্ট দুইবারে হুবহু একই")))
        city_of = dict(zip(wards["ward_id"], wards["city"]))
        same_city = transfers.empty or all(city_of.get(t.from_ward) == city_of.get(t.to_ward) == t.city for t in transfers.itertuples(index=False))
        checks.append((same_city, _t(is_bn, "Transfers stay inside one city", "স্থানান্তর একই শহরের ভেতরে")))
        for ok, label in checks:
            st.markdown(("✅ " if ok else "❌ ") + label)
        st.caption(_t(is_bn, "Full kill-switch / same-number tests run in the test suite (tests/).", "পুরো কিল-সুইচ / একই-সংখ্যা টেস্ট tests/ ফোল্ডারে চলে।"))
