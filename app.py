"""AlertAhead - dengue early-warning dashboard for public health officers (English / Bangla).

Run:  streamlit run app.py
The officer picks a place (a city or all cities, a ward or all wards) and a date. The page has one
tab each for 2, 3 and 4 weeks ahead: all wards ranked for that week (risk map, list, CSV), or one
ward's forecast for that week (range, tier, what to do, how reliable, drivers, rain vs cases).
UI by Omar Faruk Naim on the forecasting engine in dengue/.
"""
import base64
import json
from pathlib import Path

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
from dengue.ui_command import render_command_center, render_offline_toggle
from dengue.ui_extras import render_historical_replay_and_ethics, render_map_and_dispatch

st.set_page_config(
    page_title="AlertAhead | Dengue Early-Warning",
    page_icon="🛡️",
    layout="wide",
)

ALL_CITIES, ALL_WARDS = "All cities", "All wards"


# ---------------------------------------------------------------- local hero image (Base64)
def load_hero_image_base64() -> tuple[str | None, str]:
    """Load real mosquito-on-skin hero photo from assets/ so it never breaks."""
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    assets_dir = Path(__file__).resolve().parent / "assets"  # works from any working directory
    preferred = [assets_dir / f"hero_mosquito{ext}" for ext in (".jpg", ".jpeg", ".png", ".webp")]
    for path in preferred:
        if path.exists():
            return base64.b64encode(path.read_bytes()).decode("utf-8"), mime_map[path.suffix.lower()]

    if assets_dir.exists():
        for path in sorted(assets_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in mime_map:
                return base64.b64encode(path.read_bytes()).decode("utf-8"), mime_map[path.suffix.lower()]
    return None, "image/jpeg"


hero_b64, hero_mime = load_hero_image_base64()
if hero_b64:
    hero_bg_css = (
        f"linear-gradient(95deg, rgba(11, 37, 46, 0.90) 0%, rgba(15, 76, 92, 0.66) 46%, rgba(15, 76, 92, 0.10) 100%), "
        f"url('data:{hero_mime};base64,{hero_b64}')"
    )
else:
    hero_bg_css = "linear-gradient(120deg, #0B252E 0%, #0F4C5C 55%, #246A73 100%)"


# ---------------------------------------------------------------- editorial light CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #F5F7F6;
        color: #16323B;
        font-family: system-ui, -apple-system, 'Segoe UI', 'Nirmala UI', 'Noto Sans Bengali', 'Hind Siliguri', sans-serif;
    }}

    /* No top bar: hide Streamlit's toolbar (menu, running/Stop indicator) and make the bar
       see-through and click-through, so it takes no space. The sidebar's re-open arrow keeps working. */
    header[data-testid="stHeader"] {{
        background: transparent !important;
        border-bottom: none !important;
        box-shadow: none !important;
        pointer-events: none;
    }}
    header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"],
    header[data-testid="stHeader"] [data-testid="stSidebarCollapsedControl"] {{
        pointer-events: auto;
    }}
    /* Hide only the menu, running/Stop indicator and Deploy - the toolbar itself stays, because
       the sidebar's re-open arrow (stExpandSidebarButton) lives inside it. */
    [data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stStatusWidget"],
    [data-testid="stAppDeployButton"], [data-testid="stDecoration"] {{
        display: none !important;
    }}

    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1260px;
    }}

    h1, h2, h3, h4, .stMarkdown, label, .stCaption, p {{
        font-family: system-ui, -apple-system, 'Segoe UI', 'Nirmala UI', 'Noto Sans Bengali', 'Hind Siliguri', sans-serif !important;
    }}
    h1, h2, h3, h4 {{
        color: #0E2B35 !important;
    }}

    /* Soft ocean-teal sidebar matching the hero palette */
    section[data-testid="stSidebar"] {{
        background-color: #EDF4F4 !important;
        border-right: 1px solid #D2E1E3 !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: #16323B;
    }}

    /* Soft casual inputs & dropdowns */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-testid="stDateInput"] > div > div {{
        background-color: #FFFFFF !important;
        border: 1px solid #C5D7DA !important;
        border-radius: 10px !important;
        color: #16323B !important;
    }}
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover {{
        border-color: #246A73 !important;
    }}
    div[data-baseweb="select"] *,
    div[data-baseweb="input"] input {{
        color: #16323B !important;
        fill: #246A73 !important;
    }}
    div[data-baseweb="popover"] ul,
    ul[data-baseweb="menu"] {{
        background-color: #FFFFFF !important;
        border: 1px solid #D2E1E3 !important;
        border-radius: 10px !important;
    }}
    ul[data-baseweb="menu"] li {{
        color: #16323B !important;
    }}
    ul[data-baseweb="menu"] li:hover {{
        background-color: #EBF4F5 !important;
    }}

    /* Top Brand Bar */
    .top-brand {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 2px 0 6px 0;
    }}
    .brand-title {{
        font-size: 2.1rem;
        font-weight: 800;
        color: #0B252E;
        letter-spacing: -0.6px;
        line-height: 1.1;
    }}
    .brand-sub {{
        font-size: 1.05rem;
        color: #4D6670;
        font-weight: 500;
        margin-top: 2px;
    }}
    @media (max-width: 768px) {{
        .brand-title {{ font-size: 1.6rem; }}
        .brand-sub {{ font-size: 0.92rem; }}
    }}

    /* Healthline-style Trust Strip */
    .trust-strip {{
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        align-items: center;
        justify-content: space-between;
        background-color: #FFFFFF;
        border: 1px solid #DCE5E3;
        border-radius: 12px;
        padding: 10px 18px;
        margin: 8px 0 16px 0;
        font-size: 0.84rem;
        color: #2B4852;
        font-weight: 600;
    }}
    .trust-item {{
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .trust-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #246A73;
        display: inline-block;
    }}

    /* Real Photo Full-Bleed Hero Banner */
    .editorial-hero {{
        background-image: {hero_bg_css};
        background-size: cover;
        background-position: right center;
        border-radius: 20px;
        padding: 30px 36px;
        min-height: 210px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 18px 36px -16px rgba(11, 37, 46, 0.30);
        margin-bottom: 18px;
        position: relative;
        overflow: hidden;
    }}
    .hero-eyebrow {{
        display: inline-block;
        width: fit-content;
        background: rgba(255, 255, 255, 0.16);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.28);
        color: #EAF7F6 !important;
        padding: 5px 13px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }}
    .hero-headline {{
        color: #FFFFFF !important;
        font-size: 2.35rem !important;
        font-weight: 800 !important;
        line-height: 1.18 !important;
        max-width: 560px;
        margin: 0 0 10px 0 !important;
        text-shadow: 0 2px 12px rgba(0, 0, 0, 0.26);
    }}
    .hero-desc {{
        color: rgba(255, 255, 255, 0.93) !important;
        font-size: 1rem;
        line-height: 1.55;
        max-width: 530px;
        margin: 0;
    }}
    .hero-scope-pill {{
        display: inline-block;
        width: fit-content;
        margin-top: 16px;
        background: rgba(11, 37, 46, 0.52);
        border: 1px solid rgba(255, 255, 255, 0.22);
        color: #FFFFFF !important;
        padding: 6px 14px;
        border-radius: 10px;
        font-size: 0.85rem;
        font-weight: 600;
    }}

    /* Clean Light Metric Cards */
    div[data-testid="stMetric"] {{
        background-color: #FFFFFF;
        border: 1px solid #DCE5E3 !important;
        padding: 16px 18px;
        border-radius: 14px;
        box-shadow: 0 4px 12px rgba(15, 76, 92, 0.04);
    }}
    div[data-testid="stMetricLabel"] p {{
        color: #4D6670 !important;
        font-weight: 600 !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: #0B252E !important;
        font-weight: 800 !important;
    }}

    /* Tabs */
    button[data-baseweb="tab"] {{
        color: #4D6670 !important;
        font-weight: 700 !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: #0F4C5C !important;
        border-bottom-color: #0F4C5C !important;
    }}

    /* Download / Primary buttons */
    .stDownloadButton > button,
    .stButton > button {{
        background-color: #0F4C5C !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 0.5rem 1.1rem !important;
    }}
    .stDownloadButton > button:hover,
    .stButton > button:hover {{
        background-color: #196477 !important;
        color: #FFFFFF !important;
    }}

    @media (max-width: 768px) {{
        .editorial-hero {{
            padding: 28px 22px;
            min-height: 240px;
        }}
        .hero-headline {{
            font-size: 1.75rem !important;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


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

PLOTLY_LIGHT = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(color="#16323B", family="system-ui, Segoe UI, Nirmala UI, Noto Sans Bengali, sans-serif"),
)

TIER_BN = {
    "Low": "কম",
    "Medium": "মাঝারি",
    "High": "উচ্চ",
    "Critical": "চরম",
}
TIER_ACTION_BN = {
    "Low": "নিয়মিত পরিচ্ছন্নতা বজায় রাখুন এবং বাড়ির আশপাশে জমে থাকা পানি সপ্তাহে একবার পরিষ্কার করুন।",
    "Medium": "বাড়ির টব, ছাদ ও ড্রেনে জমে থাকা পানি সরান, মশারি ব্যবহার করুন এবং লার্ভা পরীক্ষা চালু রাখুন।",
    "High": "দ্রুত ফগিং ও লার্ভিসাইড স্প্রে শুরু করুন এবং এলাকার হাসপাতাল ও ক্লিনিকগুলোকে সতর্ক করুন।",
    "Critical": "জরুরি ভিত্তিতে মশক নিধন অভিযান চালান এবং হাসপাতালে শয্যা, স্যালাইন ও টেস্ট কিট প্রস্তুত রাখুন।",
}


# ---------------------------------------------------------------- top header & language toggle
nav_left, nav_right = st.columns([5, 1.25])
with nav_right:
    lang = st.selectbox("Language / ভাষা", ["English", "বাংলা"], label_visibility="collapsed")

is_bn = lang == "বাংলা"


def t_tier(tier: str) -> str:
    return TIER_BN.get(tier, tier) if is_bn else tier


def n_cases(x: float) -> str:
    v = int(round(float(x)))
    if is_bn:
        return f"{v} জন রোগী"
    return f"{v} case" if v == 1 else f"{v} cases"


def headline_box(color: str, title: str, lines: list[str], action: str) -> None:
    ink = next((m["ink"] for m in TIER_META.values() if m["color"] == color), color)
    body = "".join(f'<div style="margin-top:5px;color:#233D47;font-size:0.96rem;">{x}</div>' for x in lines)
    action_label = "সুপারিশকৃত পদক্ষেপ:" if is_bn else "Recommended action:"
    st.markdown(
        f"""<div style="padding:18px 22px;border-radius:14px;border:1px solid #DCE5E3;
        border-left:7px solid {color};background:#FFFFFF;box-shadow:0 6px 18px rgba(15,76,92,0.06);
        margin:6px 0 16px 0;">
        <div style="font-size:1.3rem;font-weight:800;color:{ink};line-height:1.25;">{title}</div>
        {body}
        <div style="margin-top:10px;padding-top:10px;border-top:1px solid #ECF2F0;color:#16323B;font-size:0.96rem;">
            <b>{action_label}</b> {action}
        </div></div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- sidebar: inputs (synced slider + number box)
st.sidebar.header("📍 " + ("এলাকা ও তারিখ" if is_bn else "Place and date"))
cities = sorted(features["city"].unique())
city = st.sidebar.selectbox(
    "শহর" if is_bn else "City",
    [ALL_CITIES] + cities,
    format_func=lambda x: ("সব শহর" if is_bn else "All cities") if x == ALL_CITIES else x,
)
pool = features if city == ALL_CITIES else features[features["city"] == city]
wards = sorted(pool["ward_id"].unique())
ward = st.sidebar.selectbox(
    "ওয়ার্ড" if is_bn else "Ward",
    [ALL_WARDS] + wards,
    format_func=lambda x: ("সব ওয়ার্ড" if is_bn else "All wards") if x == ALL_WARDS else x,
    help=(
        "একটি ওয়ার্ডের বিস্তারিত পূর্বাভাস দেখতে সেটি বেছে নিন, অথবা সব ওয়ার্ডের ঝুঁকি তালিকা দেখতে 'সব ওয়ার্ড' রাখুন।"
        if is_bn
        else "Pick one ward for its detailed forecast, or All wards to see every ward ranked by risk."
    ),
)
chosen_date = st.sidebar.date_input(
    "কোন সপ্তাহ থেকে পূর্বাভাস" if is_bn else "Forecast from week of",
    value=pd.Timestamp("2025-07-14").date(),
    min_value=data_start.date(),
    max_value=pd.Timestamp("2027-12-31").date(),
    help=(
        f"সংরক্ষিত ডেটা আছে {data_start.date()} থেকে {data_end.date()} পর্যন্ত। পরবর্তী তারিখের জন্য যেকোনো একটি ওয়ার্ড বেছে সাম্প্রতিক আবহাওয়া লিখুন।"
        if is_bn
        else f"Recorded data runs from {data_start.date()} to {data_end.date()}. For later dates, pick one ward and enter its recent weather."
    ),
)

st.sidebar.divider()
st.sidebar.header("🌧️ " + ("আবহাওয়া সিমুলেটর (What-if)" if is_bn else "What-if Simulator"))

extra_rain = st.sidebar.slider(
    "গত ১–২ দিনের বাড়তি বৃষ্টি (মি.মি.)" if is_bn else "Extra rain in the last 1–2 days (mm)",
    min_value=0,
    max_value=200,
    value=0,
    step=5,
    key="extra_rain_slider",
    help=(
        "গত ১–২ দিনে হওয়া বাড়তি বৃষ্টি যা এখনো সাপ্তাহিক রেকর্ডে যোগ হয়নি। বাংলাদেশ আবহাওয়া অধিদপ্তরের মানদণ্ড (দৈনিক): ভারী ৪৪–৮৮ মি.মি., অতি ভারী ৮৯+ মি.মি.।"
        if is_bn
        else "Rain that fell in the last day or two and is not in the weekly record yet. "
             "It is added to this week's rain in every ward shown. Guide (Bangladesh Meteorological Dept., "
             "per day): heavy 44–88 mm, very heavy 89 mm or more."
    ),
)
what_if = extra_rain > 0
offline_mode = render_offline_toggle(is_bn)

overview = ward == ALL_WARDS
issue_week = engine.snap_to_week(features, wards[0] if overview else ward, pd.Timestamp(chosen_date))
manual_mode = issue_week is None
if manual_mode:
    issue_week = pd.Timestamp(chosen_date)


# ---------------------------------------------------------------- top brand + trust strip + real hero
with nav_left:
    brand_sub = (
        "নগর ডেঙ্গু আগাম সতর্কতা ও জনস্বাস্থ্য সুরক্ষা প্ল্যাটফর্ম"
        if is_bn
        else "Urban Dengue Early-Warning & Public Health Portal"
    )
    st.markdown(
        f"""
        <div class="top-brand">
            <div>
                <div class="brand-title">AlertAhead</div>
                <div class="brand-sub">{brand_sub}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

h2_acc = metrics["horizons"][f"h{HORIZONS[0]}"]["lightgbm"]["tier_accuracy"]
if is_bn:
    trust_items = [
        "২ থেকে ৪ সপ্তাহ আগে আগাম পূর্বাভাস",
        f"{features['ward_id'].nunique()}টি ওয়ার্ড ও {len(cities)}টি শহর পর্যবেক্ষণ",
        f"২-সপ্তাহের টিয়ার নির্ভুলতা {h2_acc:.0%}",
        f"ডেটা পরিসর: {data_start:%b %Y} – {data_end:%b %Y}",
    ]
else:
    trust_items = [
        "2–4 Week Early-Warning Horizon",
        f"{features['ward_id'].nunique()} Wards Across {len(cities)} Cities",
        f"{h2_acc:.0%} Risk-Tier Accuracy (2-Week)",
        f"Data Span: {data_start:%b %Y} – {data_end:%b %Y}",
    ]

st.markdown(
    '<div class="trust-strip">'
    + "".join(f'<div class="trust-item"><span class="trust-dot"></span>{item}</div>' for item in trust_items)
    + "</div>",
    unsafe_allow_html=True,
)

city_display = ("সব শহর" if is_bn else "All cities") if city == ALL_CITIES else city
if overview:
    place = f"{city_display} · {len(wards)}টি ওয়ার্ড" if is_bn else f"{city_display} · {len(wards)} wards"
else:
    place = f"{engine.ward_traits(features, ward)['city']} · {ward}"

scope_line = (
    f"{place} · {issue_week:%d %b %Y}-এর পর ২–৪ সপ্তাহের ডেঙ্গু ঝুঁকি"
    if is_bn
    else f"{place} · dengue risk for the 2–4 weeks after {issue_week:%d %b %Y}"
)

hero_eyebrow = "জনস্বাস্থ্য আগাম সতর্কতা কেন্দ্র" if is_bn else "PUBLIC HEALTH EARLY WARNING"
hero_title = (
    "প্রাদুর্ভাবের আগেই আপনার এলাকার ডেঙ্গু ঝুঁকি জানুন।"
    if is_bn
    else "Predict Outbreaks Weeks Ahead. Protect Every Ward."
)
hero_desc = (
    "বৃষ্টিপাত, তাপমাত্রা, আর্দ্রতা ও ওয়ার্ডের পরিবেশগত তথ্য বিশ্লেষণ করে ২, ৩ ও ৪ সপ্তাহ আগেই উচ্চঝুঁকিপূর্ণ এলাকা শনাক্ত করুন—যাতে সময়মতো মশক নিধন ও চিকিৎসা প্রস্তুতি নেওয়া যায়।"
    if is_bn
    else "Tracking rainfall, temperature, humidity, and urban ward conditions to pinpoint High and Critical dengue risk 2 to 4 weeks early—giving health teams time to act."
)

st.markdown(
    f"""
    <div class="editorial-hero">
        <div class="hero-eyebrow">{hero_eyebrow}</div>
        <h1 class="hero-headline">{hero_title}</h1>
        <p class="hero-desc">{hero_desc}</p>
        <div class="hero-scope-pill">{scope_line}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not hero_b64:
    st.caption(
        "টিপস: ব্যানারে আসল মশা ও ত্বকের ছবি দেখাতে ছবিটি `assets/hero_mosquito.jpg` নামে সেভ করুন।"
        if is_bn
        else "Tip: Save the real mosquito photo inside `assets/hero_mosquito.jpg` to show it in the hero banner."
    )

if what_if:
    if is_bn:
        st.warning(
            f"সিমুলেশন সক্রিয়: গত ১–২ দিনের {extra_rain} মি.মি. বাড়তি বৃষ্টি এই সপ্তাহের বৃষ্টিপাতের সাথে যোগ করা হয়েছে"
            f"{' (সব ওয়ার্ডে)' if overview else ''}। প্রকৃত আবহাওয়ার পূর্বাভাস দেখতে স্লাইডার বা ইনপুট বক্স ০-এ নামিয়ে দিন।"
        )
    else:
        st.warning(
            f"What-if active: {extra_rain} mm of extra rain in the last 1–2 days, added to this week's rain"
            f"{' in every ward' if overview else ''}. Set the slider back to 0 to see the forecast from "
            "recorded weather."
        )


# ---------------------------------------------------------------- forecasts for every ward in scope
LEAD_EN = {
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
LEAD_BN = {
    2: ("এখনই ব্যবস্থা নিন",
        "নিচের উচ্চ ও চরম ঝুঁকিপূর্ণ ওয়ার্ডগুলোতে দ্রুত ফগিং ও লার্ভিসাইড টিম পাঠান, স্থানীয় ক্লিনিক ও হাসপাতালগুলোকে সতর্ক করুন এবং শয্যা, স্যালাইন ও টেস্ট কিটের মজুত যাচাই করুন।"),
    3: ("প্রজনন চক্র ভাঙুন",
        "নিচের ওয়ার্ডগুলোতে জলাবদ্ধ স্থান নিষ্কাশন, জমে থাকা পাত্র ও বন্ধ ড্রেন পরিষ্কার এবং পরিচ্ছন্নতা অভিযান চালান। এখনই করলে রোগী বাড়ার আগেই মশার প্রজনন চক্র ভেঙে দেওয়া সম্ভব।"),
    4: ("আগাম প্রস্তুতি নিন",
        "নিচের ওয়ার্ডগুলোর জন্য স্যালাইন, টেস্ট কিট ও লার্ভিসাইড অর্ডার করুন, ফিল্ড টিমের সূচি ঠিক করুন এবং সম্ভাব্য রোগীর চাপ সম্পর্কে হাসপাতালগুলোকে অবহিত করুন।"),
}
LEAD = LEAD_BN if is_bn else LEAD_EN

week_tbl = None
if not manual_mode:
    frows = features[(features["date"] == issue_week) & features["ward_id"].isin(wards)]
    if what_if:
        week_tbl = engine.predict_table(models, frows, extra_rain)
    else:
        week_tbl = preds[(preds["date"] == issue_week) & preds["ward_id"].isin(wards)]
    week_tbl = week_tbl.merge(
        frows[["ward_id", TARGET_COL]].rename(columns={TARGET_COL: "cases_now_rec"}), on="ward_id"
    )


@st.cache_data
def reliability() -> dict:
    """Per horizon, on the test year: MAE, tier accuracy, High/Critical weeks caught (alone and with watch)."""
    out = {}
    for h in HORIZONS:
        m = metrics["horizons"][f"h{h}"]["lightgbm"]
        d = preds[["ward_id", "date", f"high_h{h}"]].merge(
            features[["ward_id", "date", f"target_h{h}", f"target_date_h{h}"]], on=["ward_id", "date"]
        )
        d = d[d[f"target_date_h{h}"].dt.year == metrics["test_year"]]
        actual_high = d[f"target_h{h}"] >= DEFAULT_THRESHOLDS["high"]
        flagged = d[f"high_h{h}"] >= DEFAULT_THRESHOLDS["high"]
        out[h] = {
            "mae": m["MAE"],
            "tier_acc": m["tier_accuracy"],
            "recall": m["high_critical_recall"],
            "recall_watch": float((flagged & actual_high).sum() / max(int(actual_high.sum()), 1)),
        }
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
    when = f"{target:%d %b %Y}-এর সপ্তাহে" if is_bn else f"the week of {target:%d %b %Y}"
    lines = []
    if n_crit or n_high:
        title = (
            f"⚠ {when} {n}টির মধ্যে {n_crit + n_high}টি ওয়ার্ড উচ্চ বা চরম ঝুঁকিতে"
            if is_bn
            else f"⚠ {n_crit + n_high} of {n} wards HIGH or CRITICAL in {when}"
        )
        lines.append(
            f"সর্বোচ্চ ঝুঁকি: <b>{top['ward_id']}</b> ({top['city']}), আনুমানিক {n_cases(top['cases'])}।"
            if is_bn
            else f"Highest: <b>{top['ward_id']}</b> ({top['city']}), about {n_cases(top['cases'])}."
        )
        if n_watch:
            lines.append(
                f"আরও {n_watch}টি ওয়ার্ড উচ্চ ঝুঁকিতে পৌঁছাতে পারে (পর্যবেক্ষণ তালিকা)।"
                if is_bn
                else f"{n_watch} more wards could reach High (watch)."
            )
        color, action = TIER_META[top["tier"]]["color"], f"<b>{LEAD[h][0]}:</b> {LEAD[h][1]}"
    elif n_watch:
        title = (
            f"⚠ পর্যবেক্ষণ — {when} {n}টির মধ্যে {n_watch}টি ওয়ার্ড উচ্চ ঝুঁকিতে পৌঁছাতে পারে"
            if is_bn
            else f"⚠ WATCH - {n_watch} of {n} wards could reach HIGH in {when}"
        )
        lines.append(
            "আনুমানিক রোগীর সংখ্যা উচ্চ ঝুঁকির নিচে থাকলেও সম্ভাব্য ঊর্ধ্বসীমা উচ্চ ঝুঁকিতে পৌঁছাতে পারে।"
            if is_bn
            else "Expected cases stay below High, but the likely range reaches it."
        )
        color, action = TIER_META["High"]["color"], f"<b>{LEAD[h][0]}:</b> {LEAD[h][1]}"
    elif n_med:
        title = (
            f"{when} {n}টির মধ্যে {n_med}টি ওয়ার্ড মাঝারি ঝুঁকিতে"
            if is_bn
            else f"MEDIUM RISK in {n_med} of {n} wards in {when}"
        )
        color = TIER_META["Medium"]["color"]
        action = TIER_ACTION_BN["Medium"] if is_bn else TIER_META["Medium"]["action"]
    else:
        title = (
            f"✓ {when} সব {n}টি ওয়ার্ডই কম ঝুঁকিতে"
            if is_bn
            else f"✓ LOW RISK in all {n} wards in {when}"
        )
        color = TIER_META["Low"]["color"]
        action = TIER_ACTION_BN["Low"] if is_bn else TIER_META["Low"]["action"]

    lines.append(
        (
            f"<span style='color:#4D6670'>মোট আনুমানিক: <b>{n_cases(t['cases'].sum())}</b> "
            f"&nbsp;·&nbsp; বর্তমানে: <b>{n_cases(t['cases_now_rec'].sum())}</b></span>"
        )
        if is_bn
        else (
            f"<span style='color:#4D6670'>Expected: <b>{n_cases(t['cases'].sum())}</b> across these wards "
            f"that week &nbsp;·&nbsp; now: <b>{n_cases(t['cases_now_rec'].sum())}</b></span>"
        )
    )
    headline_box(color, title, lines, action)

    r = reliability()[h]
    if is_bn:
        st.caption(
            f"ℹ️ {h}-সপ্তাহের পূর্বাভাস কতটা নির্ভরযোগ্য? {metrics['test_year']} সালে গড় ত্রুটি ছিল সপ্তাহে মাত্র "
            f"{r['mae']:.1f} জন রোগী, ঝুঁকির মাত্রা সঠিক ছিল {r['tier_acc']:.0%} সময়, এবং উচ্চ/চরম ঝুঁকির সপ্তাহ "
            f"আগেভাগে শনাক্ত করেছে {r['recall']:.0%} (পর্যবেক্ষণ ফ্ল্যাগসহ {r['recall_watch']:.0%})।"
        )
    else:
        st.caption(
            f"ℹ️ How reliable is the {h}-week forecast? In {metrics['test_year']} it was off by "
            f"{r['mae']:.1f} cases a week on average, got the risk tier right {r['tier_acc']:.0%} of the time, "
            f"and caught {r['recall']:.0%} of High/Critical weeks ({r['recall_watch']:.0%} counting watch flags)."
        )

    k = st.columns(5)
    boxes = (
        [
            ("চরম ঝুঁকির ওয়ার্ড", n_crit, None),
            ("উচ্চ ঝুঁকির ওয়ার্ড", n_high, None),
            ("পর্যবেক্ষণ (Watch)", n_watch, "গড় পূর্বাভাস কম হলেও সম্ভাব্য ঊর্ধ্বসীমা উচ্চ বা চরম ঝুঁকিতে পৌঁছাতে পারে।"),
            ("মাঝারি ঝুঁকির ওয়ার্ড", n_med, None),
            ("কম ঝুঁকির ওয়ার্ড", n_low, None),
        ]
        if is_bn
        else [
            ("Critical wards", n_crit, None),
            ("High wards", n_high, None),
            ("Watch", n_watch, "Expected lower, but the likely range reaches High or Critical."),
            ("Medium wards", n_med, None),
            ("Low wards", n_low, None),
        ]
    )
    for col, (label, val, hlp) in zip(k, boxes):
        col.metric(label, val, border=True, help=hlp)

    if city == ALL_CITIES:
        st.markdown("**শহর অনুযায়ী সারসংক্ষেপ**" if is_bn else "**By city**")
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
        if is_bn:
            by_city_disp = by_city.rename(columns={
                "City": "শহর", "Wards": "ওয়ার্ড", "Critical": "চরম", "High": "উচ্চ",
                "Watch": "পর্যবেক্ষণ", "Medium": "মাঝারি", "Low": "কম",
                "Cases now": "বর্তমান রোগী", "Expected cases": "আনুমানিক রোগী",
            })
            st.dataframe(
                by_city_disp.style.format({"বর্তমান রোগী": "{:.0f}", "আনুমানিক রোগী": "{:.0f}"}),
                hide_index=True, key=f"city_{h}",
            )
        else:
            st.dataframe(
                by_city.style.format({"Cases now": "{:.0f}", "Expected cases": "{:.0f}"}),
                hide_index=True, key=f"city_{h}",
            )

    st.markdown(
        f"**ওয়ার্ড ঝুঁকি মানচিত্র — {when}** (প্রতিটি ঘর একটি ওয়ার্ড; সংখ্যাটি আনুমানিক রোগীর সংখ্যা)"
        if is_bn
        else f"**Ward risk map - {when}** (each box is a ward; the number is expected cases)"
    )
    t["ward_no"] = t["ward_id"].str.split("_").str[-1]
    rows = sorted(t["city"].unique())[::-1]
    cols = sorted(t["ward_no"].unique())
    grid = {
        v: t.pivot(index="city", columns="ward_no", values=v).reindex(index=rows, columns=cols)
        for v in ("rank", "cases", "ward_id")
    }
    txt = [["" if pd.isna(x) else f"{x:.1f}" for x in row] for row in grid["cases"].to_numpy(dtype=float)]
    scale = []
    for i, c in enumerate(TIER_META[x]["color"] for x in TIERS):
        scale += [[i / len(TIERS), c], [(i + 1) / len(TIERS), c]]
    tier_labels = [t_tier(x) for x in TIERS]
    hover_tpl = (
        "%{customdata}<br>আনুমানিক %{text} জন রোগী<extra></extra>"
        if is_bn
        else "%{customdata}<br>expected %{text} cases<extra></extra>"
    )
    fig = go.Figure(go.Heatmap(
        z=grid["rank"].to_numpy(dtype=float), x=cols, y=rows, zmin=-0.5, zmax=len(TIERS) - 0.5,
        colorscale=scale, text=txt, texttemplate="%{text}", xgap=3, ygap=3, hoverongaps=False,
        customdata=grid["ward_id"].fillna("").to_numpy(dtype=object),
        hovertemplate=hover_tpl,
        colorbar=dict(tickvals=list(range(len(TIERS))), ticktext=tier_labels, title="মাত্রা" if is_bn else "Tier"),
    ))
    fig.update_layout(
        **PLOTLY_LIGHT,
        height=120 + 50 * len(rows),
        margin=dict(l=14, r=14, t=22, b=14),
        xaxis=dict(side="top", title="ওয়ার্ড নম্বর" if is_bn else "Ward number", type="category"),
    )
    st.plotly_chart(fig, key=f"map_{h}")

    st.markdown(
        f"**ঝুঁকির ক্রম অনুযায়ী তালিকা — ওপরের সারিগুলোতে আগে ব্যবস্থা নিন** ({n}টি ওয়ার্ড)"
        if is_bn
        else f"**Ranked list - act on the top rows first** ({n} wards)"
    )
    risk_col = "ঝুঁকি" if is_bn else "Risk"
    # Ranked list: no colour fill - stars beside the tier (Medium 1, High 2, Critical 3, in red).
    stars = {"Low": "", "Medium": "★", "High": "★★", "Critical": "★★★"}
    risk_vals = [f"{t_tier(x)} {stars[x]}".strip() for x in t["tier"]]
    exp_col = "আনুমানিক রোগী" if is_bn else "Expected cases"
    if is_bn:
        table = pd.DataFrame({
            "শহর": t["city"],
            "ওয়ার্ড": t["ward_id"],
            risk_col: risk_vals,
            exp_col: t["cases"].round(1),
            "সম্ভাব্য রেঞ্জ": t["low"].round().astype(int).astype(str) + "–" + t["high"].round().astype(int).astype(str),
            "বর্তমান রোগী": t["cases_now_rec"].round().astype(int),
            "পর্যবেক্ষণ": t["watch"].map(lambda w: f"⚠ {t_tier(w)} হতে পারে" if w else ""),
        })
    else:
        table = pd.DataFrame({
            "City": t["city"],
            "Ward": t["ward_id"],
            risk_col: risk_vals,
            exp_col: t["cases"].round(1),
            "Likely range": t["low"].round().astype(int).astype(str) + "–" + t["high"].round().astype(int).astype(str),
            "Cases now": t["cases_now_rec"].round().astype(int),
            "Watch": t["watch"].map(lambda w: f"⚠ could reach {w}" if w else ""),
        })

    star_color = {"Medium": "#D93025", "High": "#D93025", "Critical": "#8C1D18"}
    row_tiers = list(t["tier"])
    styled = table.style.apply(
        lambda col: [f"color:{star_color[x]};font-weight:700" if x in star_color else "" for x in row_tiers],
        subset=[risk_col], axis=0,
    ).format({exp_col: "{:.1f}"})
    st.dataframe(styled, hide_index=True, height=380, key=f"rank_{h}")
    st.download_button(
        "⬇️ তালিকা ডাউনলোড করুন (CSV)" if is_bn else "⬇️ Download this list (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name=f"dengue_wards_{h}wk_{target:%Y-%m-%d}.csv",
        mime="text/csv",
        key=f"dl_{h}",
    )


DOT = {"Low": "🟢", "Medium": "⭕", "High": "🔴", "Critical": "🚨"}  # no yellow/orange: light red -> red -> siren
LEAD_WARD_EN = {
    2: "Send fogging and larvicide teams to this ward, alert its clinics and hospitals, and check beds, "
       "saline and test-kit stock.",
    3: "Drain water-logged spots, clear containers and blocked drains, and run a clean-up drive in this ward. "
       "Done now, this breaks the breeding cycle before these cases appear.",
    4: "Order saline, test kits and larvicide for this ward, schedule field teams, and brief local hospitals.",
}
LEAD_WARD_BN = {
    2: "এই ওয়ার্ডে দ্রুত ফগিং ও লার্ভিসাইড টিম পাঠান, এলাকার ক্লিনিক ও হাসপাতাল সতর্ক করুন এবং শয্যা, স্যালাইন ও টেস্ট কিটের মজুত যাচাই করুন।",
    3: "এই ওয়ার্ডের জলাবদ্ধ স্থান নিষ্কাশন, জমে থাকা পাত্র ও বন্ধ ড্রেন পরিষ্কার এবং পরিচ্ছন্নতা অভিযান চালান। এখনই করলে রোগী বাড়ার আগেই প্রজনন চক্র ভেঙে যাবে।",
    4: "এই ওয়ার্ডের জন্য স্যালাইন, টেস্ট কিট ও লার্ভিসাইড অর্ডার করুন, ফিল্ড টিমের সূচি ঠিক করুন এবং স্থানীয় হাসপাতালগুলোকে আগেভাগে জানান।",
}
LEAD_WARD = LEAD_WARD_BN if is_bn else LEAD_WARD_EN


def ward_forecasts() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The selected ward's last 4 weeks, model input and 2/3/4-week forecasts."""
    traits = engine.ward_traits(features, ward)
    if manual_mode:
        if is_bn:
            st.info(
                f"{issue_week:%d %b %Y} তারিখের জন্য সংরক্ষিত আবহাওয়া ডেটা নেই (ডেটা শেষ হয়েছে {data_end:%d %b %Y})। "
                "নিচে গত ৪ সপ্তাহের তথ্য দিন — বছরের এই সময়ের স্বাভাবিক মান দিয়ে ঘরগুলো আগে থেকেই পূরণ করা আছে।"
            )
        else:
            st.info(
                f"No recorded weather for {issue_week:%d %b %Y} (data ends {data_end:%d %b %Y}). "
                "Enter the last 4 weeks below - pre-filled with this ward's usual values for this time of year."
            )
        normals = engine.seasonal_normals(features, ward, issue_week)
        c_week = "সপ্তাহ" if is_bn else "Week of"
        c_rain = "বৃষ্টিপাত (মি.মি.)" if is_bn else "Rain (mm)"
        c_humid = "আর্দ্রতা (%)" if is_bn else "Humidity (%)"
        c_temp = "গড় তাপমাত্রা (°C)" if is_bn else "Avg temp (°C)"
        c_cases = "শনাক্ত রোগী" if is_bn else "Reported cases"

        edited = st.data_editor(
            normals.rename(columns={
                "date": c_week, RAIN_COL: c_rain, HUMID_COL: c_humid,
                TEMP_COL: c_temp, TARGET_COL: c_cases,
            }),
            hide_index=True,
            disabled=[c_week],
            key=f"manual_{ward}_{issue_week.date()}_{lang}",
            column_config={c_week: st.column_config.DateColumn(format="YYYY-MM-DD")},
        )
        weeks = edited.rename(columns={
            c_week: "date", c_rain: RAIN_COL, c_humid: HUMID_COL,
            c_temp: TEMP_COL, c_cases: TARGET_COL,
        })
        weeks = engine.add_recent_rain(weeks, extra_rain)
        X = engine.build_row(features, ward, issue_week, weeks, traits)
        fcs = engine.outlook(models, preds, features, ward, issue_week, X)
    else:
        weeks = engine.recent_weeks(features, ward, issue_week)
        X = features[(features["ward_id"] == ward) & (features["date"] == issue_week)]
        if what_if:
            # Adjust the stored feature row (works even when < 4 recorded weeks exist, e.g. Jan 2016)
            weeks = engine.add_recent_rain(weeks, extra_rain)
            X = engine.add_rain_to_features(X, extra_rain)
            fcs = engine.outlook(models, preds, features, ward, issue_week, X)
        else:
            fcs = engine.outlook(models, preds, features, ward, issue_week)
    return weeks, X, {f.horizon: f for f in fcs}


def render_ward_week(h: int, weeks: pd.DataFrame, X: pd.DataFrame, fc) -> None:
    """One ward, one lead time: headline, what to do, how reliable, why, and the rain/cases picture."""
    now_cases = float(weeks[TARGET_COL].iloc[-1])
    now_tier = assign_risk_tier(now_cases)
    if fc.cases >= now_cases * 1.25 and fc.cases - now_cases >= 1:
        trend = "বাড়ছে ↑" if is_bn else "rising ↑"
    elif fc.cases <= now_cases * 0.8 and now_cases - fc.cases >= 1:
        trend = "কমছে ↓" if is_bn else "falling ↓"
    else:
        trend = "প্রায় অপরিবর্তিত →" if is_bn else "about the same →"

    when = f"{fc.target_date:%d %b %Y}-এর সপ্তাহে" if is_bn else f"the week of {fc.target_date:%d %b %Y}"
    if fc.tier in ("High", "Critical"):
        title = (
            f"⚠ {when} {t_tier(fc.tier)} ঝুঁকি"
            if is_bn
            else f"⚠ {fc.tier.upper()} RISK in {when}"
        )
        color, action = fc.color, f"<b>{LEAD[h][0]}:</b> {LEAD_WARD[h]}"
    elif fc.watch:
        title = (
            f"⚠ পর্যবেক্ষণ — {when} {t_tier(fc.watch)} ঝুঁকিতে পৌঁছাতে পারে"
            if is_bn
            else f"⚠ WATCH - could reach {fc.watch.upper()} in {when}"
        )
        color = TIER_META[fc.watch]["color"]
        base_act = TIER_ACTION_BN.get(fc.tier, fc.action) if is_bn else fc.action
        action = (
            f"{base_act} <b>ঝুঁকি বাড়লে ({LEAD[h][0]}):</b> {LEAD_WARD[h]}"
            if is_bn
            else f"{base_act} <b>If it rises ({LEAD[h][0].lower()}):</b> {LEAD_WARD[h]}"
        )
    elif fc.tier == "Medium":
        title = f"{when} মাঝারি ঝুঁকি" if is_bn else f"MEDIUM RISK in {when}"
        color = fc.color
        action = TIER_ACTION_BN["Medium"] if is_bn else fc.action
    else:
        title = f"✓ {when} কম ঝুঁকি" if is_bn else f"✓ LOW RISK in {when}"
        color = fc.color
        action = TIER_ACTION_BN["Low"] if is_bn else fc.action

    lines = (
        [
            f"আনুমানিক প্রায় <b>{fc.cases:.1f} জন রোগী</b> (সম্ভাব্য রেঞ্জ {fc.low:.0f}–{fc.high:.0f}), "
            f"{fc.issue_date:%d %b}-এর {h} সপ্তাহ পর।",
            f"<span style='color:#4D6670'>বর্তমানে: <b>{n_cases(now_cases)}</b> ({t_tier(now_tier)} ঝুঁকি) "
            f"&nbsp;·&nbsp; বর্তমানের তুলনায়: <b>{trend}</b></span>",
        ]
        if is_bn
        else [
            f"Expected about <b>{fc.cases:.1f} cases</b> (likely {fc.low:.0f}–{fc.high:.0f}), "
            f"{h} weeks after {fc.issue_date:%d %b}.",
            f"<span style='color:#4D6670'>Now: <b>{n_cases(now_cases)}</b> this week ({now_tier}) "
            f"&nbsp;·&nbsp; compared with now: <b>{trend}</b></span>",
        ]
    )
    if fc.actual is not None and not what_if:
        lines.append(
            f"✔ ওই সপ্তাহে প্রকৃত রেকর্ডকৃত রোগী: <b>{n_cases(fc.actual)}</b> ({t_tier(assign_risk_tier(fc.actual))} ঝুঁকি)"
            if is_bn
            else f"✔ Actually recorded that week: <b>{n_cases(fc.actual)}</b> ({assign_risk_tier(fc.actual)})"
        )
    headline_box(color, title, lines, action)

    r = reliability()[h]
    if is_bn:
        st.caption(
            f"ℹ️ {h}-সপ্তাহের পূর্বাভাস কতটা নির্ভরযোগ্য? {metrics['test_year']} সালে গড় ত্রুটি ছিল সপ্তাহে মাত্র "
            f"{r['mae']:.1f} জন রোগী, ঝুঁকির মাত্রা সঠিক ছিল {r['tier_acc']:.0%} সময়, এবং উচ্চ/চরম ঝুঁকির সপ্তাহ "
            f"আগেভাগে শনাক্ত করেছে {r['recall']:.0%} (পর্যবেক্ষণ ফ্ল্যাগসহ {r['recall_watch']:.0%})।"
        )
    else:
        st.caption(
            f"ℹ️ How reliable is the {h}-week forecast? In {metrics['test_year']} it was off by "
            f"{r['mae']:.1f} cases a week on average, got the risk tier right {r['tier_acc']:.0%} of the time, "
            f"and caught {r['recall']:.0%} of High/Critical weeks ({r['recall_watch']:.0%} counting watch flags)."
        )

    if week_tbl is not None:
        order = week_tbl[f"pred_h{h}"].astype(float).rank(ascending=False, method="min")
        pos = int(order[week_tbl["ward_id"] == ward].iloc[0])
        scope_name = ("সব শহর" if is_bn else "all cities") if city == ALL_CITIES else city
        if is_bn:
            st.caption(
                f"📊 ওই সপ্তাহে {scope_name}-এর {len(week_tbl)}টি ওয়ার্ডের মধ্যে এই ওয়ার্ডের অবস্থান **#{pos}**। "
                "সম্পূর্ণ তালিকা দেখতে সাইডবার থেকে *সব ওয়ার্ড* নির্বাচন করুন।"
            )
        else:
            st.caption(
                f"📊 This ward ranks **#{pos} of {len(week_tbl)}** wards in {scope_name} for that week. "
                "Pick *All wards* to see the full list."
            )

    left, right = st.columns([1, 1.15])
    with left:
        st.markdown(
            f"**কেন এই ঝুঁকি** — {h}-সপ্তাহের পূর্বাভাসের প্রধান কারণসমূহ"
            if is_bn
            else f"**Why** - what drives the {h}-week forecast"
        )
        rain_sum = float(X["rain_sum_4w"].iloc[0])
        rain_normal = rain_sum - float(X["rain_anom_4w"].iloc[0])
        if rain_normal > 1:
            ratio = rain_sum / rain_normal
            if is_bn:
                word = "স্বাভাবিকের চেয়ে বেশি বৃষ্টি" if ratio > 1.15 else "স্বাভাবিকের চেয়ে কম বৃষ্টি" if ratio < 0.85 else "প্রায় স্বাভাবিক"
                st.caption(
                    f"🌧️ গত ৪ সপ্তাহের মোট বৃষ্টিপাত: **{rain_sum:.0f} মি.মি.**, যা এই ওয়ার্ডের স্বাভাবিক গড়ের {ratio:.1f} গুণ ({word})।"
                )
            else:
                word = "wetter than usual" if ratio > 1.15 else "drier than usual" if ratio < 0.85 else "about usual"
                st.caption(
                    f"🌧️ Rain over the last 4 weeks: **{rain_sum:.0f} mm**, {ratio:.1f}× this ward's "
                    f"usual for this time of year ({word})."
                )
        drv, base = engine.drivers(models[h], X)
        icons = {
            "Rain": "🌧️", "Unusually": "🌧️", "Humidity": "💧", "Temperature": "🌡️", "Season": "📅",
            "Cases": "🦟", "Population": "🏘️", "Water": "🏘️", "Sanitation": "🏘️",
        }
        drv["label"] = [next((i for k, i in icons.items() if d.startswith(k)), "") + " " + d for d in drv["driver"]]
        drv = drv.iloc[::-1]
        colors = ["#C62828" if v > 0 else "#1E7F74" for v in drv["effect_cases"]]
        span = max(drv["effect_cases"].abs().max(), 0.5)
        fig = go.Figure(go.Bar(
            x=drv["effect_cases"], y=drv["label"], orientation="h", marker_color=colors,
            text=[f"{v:+.1f}" for v in drv["effect_cases"]], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:+.2f}" + (" রোগী<extra></extra>" if is_bn else " cases<extra></extra>"),
        ))
        fig.update_layout(
            **PLOTLY_LIGHT,
            height=360,
            margin=dict(l=14, r=44, t=14, b=14),
            xaxis=dict(
                title=(
                    f"স্বাভাবিক সপ্তাহের ({base:.1f}) তুলনায় রোগী বাড়া (লাল) / কমা (সবুজ)"
                    if is_bn
                    else f"Cases added (red) / removed (green) vs a typical week ({base:.1f})"
                ),
                range=[-span * 1.25 if (drv["effect_cases"] < 0).any() else 0, span * 1.25],
                gridcolor="#EAF0EF",
                zerolinecolor="#CBD5E1",
            ),
        )
        fig.update_layout(dragmode=False)  # no accidental zoom-box (grey overlay) during a demo
        st.plotly_chart(fig, key=f"drivers_{h}")
        st.caption(
            "🌧️💧🌡️ আবহাওয়া · 🏘️ ওয়ার্ডের অবস্থা · 📅 মৌসুম · 🦟 সাম্প্রতিক রোগী"
            if is_bn
            else "🌧️💧🌡️ weather · 🏘️ ward conditions · 📅 time of year · 🦟 recent cases"
        )
    with right:
        st.markdown(
            f"**বৃষ্টিপাত ও ডেঙ্গু রোগী** — হালকা নীল অংশ ইনপুট সপ্তাহ; ডায়মন্ড চিহ্নটি {h}-সপ্তাহের পূর্বাভাস"
            if is_bn
            else f"**Rain and cases** - shaded weeks are the inputs; the diamond is the {h}-week forecast"
        )
        hist = features[(features["ward_id"] == ward) & (features["date"] <= issue_week)].tail(20)
        fig = go.Figure()
        fig.add_bar(
            x=hist["date"], y=hist[RAIN_COL],
            name="বৃষ্টিপাত (মি.মি.)" if is_bn else "Rain (mm)",
            marker_color="rgba(36, 106, 115, 0.28)",
        )
        if manual_mode or what_if:
            fig.add_bar(
                x=weeks["date"], y=weeks[RAIN_COL],
                name="ব্যবহৃত বৃষ্টি (সিমুলেশন / ইনপুট)" if is_bn else "Rain used (what-if / entered)",
                marker_color="rgba(15, 76, 92, 0.75)",
            )
        fig.add_scatter(
            x=hist["date"], y=hist[TARGET_COL],
            name="রোগী" if is_bn else "Cases",
            mode="lines+markers", line=dict(color="#0F4C5C", width=3), yaxis="y2",
        )
        future = features[
            (features["ward_id"] == ward)
            & (features["date"] > issue_week)
            & (features["date"] <= fc.target_date)
        ]
        if len(future) and not what_if:
            fig.add_scatter(
                x=future["date"], y=future[TARGET_COL],
                name="প্রকৃত রোগী (পরবর্তী)" if is_bn else "Cases (actual, later)",
                mode="lines+markers", line=dict(color="#78909C", dash="dot"), yaxis="y2",
            )
        fig.add_scatter(
            x=[fc.target_date], y=[fc.cases], mode="markers",
            name=f"পূর্বাভাস (+{h} সপ্তাহ)" if is_bn else f"Forecast (+{h} wk)",
            yaxis="y2",
            marker=dict(color=fc.color, size=15, symbol="diamond"),
            error_y=dict(
                type="data", symmetric=False,
                array=[fc.high - fc.cases], arrayminus=[fc.cases - fc.low], color="#64748B",
            ),
        )
        fig.add_vrect(
            x0=issue_week - pd.Timedelta(days=24), x1=issue_week + pd.Timedelta(days=3),
            fillcolor="rgba(36, 106, 115, 0.10)", line_width=0,
        )
        fig.update_layout(
            **PLOTLY_LIGHT,
            height=360,
            margin=dict(l=14, r=14, t=14, b=14),
            barmode="overlay",
            xaxis=dict(gridcolor="#EAF0EF"),
            yaxis=dict(
                title="বৃষ্টিপাত (মি.মি.)" if is_bn else "Rain (mm)",
                side="right", showgrid=False, rangemode="tozero",
            ),
            yaxis2=dict(
                title="সাপ্তাহিক রোগী" if is_bn else "Weekly cases",
                overlaying="y", side="left", rangemode="tozero", gridcolor="#EAF0EF",
            ),
            legend=dict(orientation="h", y=1.15),
        )
        fig.update_layout(dragmode=False)
        st.plotly_chart(fig, key=f"rain_{h}")


def render_accuracy() -> None:
    st.subheader(
        f"মডেলটি কতটা নির্ভুল? {metrics['test_year']} সালের অদেখা ডেটায় পরীক্ষিত"
        if is_bn
        else f"How accurate is it? Tested on {metrics['test_year']} (never used in training)"
    )
    st.caption(
        "ট্রেইন: ২০২৩ পর্যন্ত · টিউন/ভ্যালিডেশন: ২০২৪ · টেস্ট: ২০২৫। তারিখ অনুযায়ী ভাগ করা, কখনো শাফল করা হয়নি—তাই মডেল ভবিষ্যতের ডেটা দেখতে পায় না।"
        if is_bn
        else "Train: target weeks up to 2023 · Tune / early stopping: 2024 · Test: 2025. "
             "Split by date, never shuffled, so the model never sees the future."
    )
    names = (
        {
            "lightgbm": "LightGBM (আমাদের মডেল)",
            "negative_binomial_glm": "নেগেটিভ বাইনোমিয়াল GLM",
            "same_week_last_year": "গত বছরের একই সপ্তাহ",
            "persistence_cases_now": "এই সপ্তাহের রোগী",
        }
        if is_bn
        else {
            "lightgbm": "LightGBM (ours)",
            "negative_binomial_glm": "Negative binomial GLM",
            "same_week_last_year": "Same week last year",
            "persistence_cases_now": "This week's cases",
        }
    )
    col_wk = "কত সপ্তাহ আগে" if is_bn else "Weeks ahead"
    col_mod = "মডেল" if is_bn else "Model"
    col_tier = "টিয়ার নির্ভুলতা" if is_bn else "Tier accuracy"
    col_hc = "উচ্চ+চরম শনাক্ত" if is_bn else "High+Critical caught"
    col_cov = "৮০% রেঞ্জ কভারেজ" if is_bn else "80% range coverage"

    rows = []
    for hk, m in metrics["horizons"].items():
        for key, label in names.items():
            r = m[key]
            rows.append({
                col_wk: int(hk[1:]),
                col_mod: label,
                "MAE": r["MAE"],
                "RMSE": r["RMSE"],
                col_tier: r["tier_accuracy"],
                col_hc: r["high_critical_recall"],
                col_cov: r.get("range_80_coverage"),
            })
    table = pd.DataFrame(rows)
    palette = ["#0F4C5C", "#2E938B", "#6250D6", "#94A3B8"]
    fig = go.Figure()
    for idx, label in enumerate(names.values()):
        sub = table[table[col_mod] == label]
        x_vals = [f"{h} সপ্তাহ" if is_bn else f"{h} weeks" for h in sub[col_wk]]
        fig.add_bar(x=x_vals, y=sub["MAE"], name=label, marker_color=palette[idx % len(palette)])
    fig.update_layout(
        **PLOTLY_LIGHT,
        height=330,
        barmode="group",
        yaxis_title="গড় ত্রুটি (রোগী/সপ্তাহ, যত কম তত ভালো)" if is_bn else "Average error (cases/week, lower is better)",
        yaxis=dict(gridcolor="#EAF0EF"),
        margin=dict(l=14, r=14, t=18, b=14),
        legend=dict(orientation="h", y=1.16),
    )
    st.plotly_chart(fig)
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            col_tier: st.column_config.NumberColumn(format="percent"),
            col_hc: st.column_config.NumberColumn(format="percent"),
            col_cov: st.column_config.NumberColumn(format="percent"),
        },
    )

    vh = 3
    scope_wards = wards if overview else [ward]
    label = (
        (f"{city_display}, মোট {len(scope_wards)}টি ওয়ার্ড" if is_bn else f"{city_display}, all {len(scope_wards)} wards added up")
        if overview
        else ward
    )
    st.markdown(
        f"**{label}: {metrics['test_year']} সালে {vh}-সপ্তাহের পূর্বাভাস বনাম প্রকৃত রোগী**"
        if is_bn
        else f"**{label}: {vh}-week forecast vs actual through {metrics['test_year']}**"
    )
    w_pred = preds[preds["ward_id"].isin(scope_wards)][["ward_id", "date", f"pred_h{vh}", f"low_h{vh}", f"high_h{vh}"]].copy()
    w_pred["target_date"] = w_pred["date"] + pd.Timedelta(weeks=vh)
    w_act = features[features["ward_id"].isin(scope_wards)][["ward_id", "date", TARGET_COL]]
    w = w_pred.merge(w_act, left_on=["ward_id", "target_date"], right_on=["ward_id", "date"], suffixes=("", "_t"))
    w = w[w["target_date"].dt.year == metrics["test_year"]]
    w = w.groupby("target_date", as_index=False)[[f"pred_h{vh}", f"low_h{vh}", f"high_h{vh}", TARGET_COL]].sum()
    fig = go.Figure()
    if not overview:
        fig.add_scatter(x=w["target_date"], y=w[f"high_h{vh}"], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(
            x=w["target_date"], y=w[f"low_h{vh}"], line=dict(width=0), fill="tonexty",
            fillcolor="rgba(36, 106, 115, 0.16)", name="৮০% রেঞ্জ" if is_bn else "80% range",
        )
    fig.add_scatter(
        x=w["target_date"], y=w[TARGET_COL],
        name="প্রকৃত রোগী" if is_bn else "Actual",
        line=dict(color="#0F4C5C", width=2.4),
    )
    fig.add_scatter(
        x=w["target_date"], y=w[f"pred_h{vh}"],
        name="পূর্বাভাস" if is_bn else "Forecast",
        line=dict(color="#D93025", width=2.2, dash="dash"),
    )
    fig.update_layout(
        **PLOTLY_LIGHT,
        height=330,
        margin=dict(l=14, r=14, t=18, b=14),
        xaxis=dict(gridcolor="#EAF0EF"),
        yaxis=dict(title="সাপ্তাহিক রোগী" if is_bn else "Weekly cases", gridcolor="#EAF0EF"),
        legend=dict(orientation="h", y=1.15),
    )
    st.plotly_chart(fig)


# ---------------------------------------------------------------- layout (2 / 3 / 4 weeks ahead + Accuracy)
if not overview:
    w_weeks, w_X, w_fcs = ward_forecasts()


def tab_label(h: int) -> str:
    dt = issue_week + pd.Timedelta(weeks=h)
    label = f"⏱️ {h} সপ্তাহ পর · {dt:%d %b}" if is_bn else f"⏱️ {h} weeks ahead · {dt:%d %b}"
    if not overview:
        fc = w_fcs[h]
        label += f" · {DOT[fc.tier]} {t_tier(fc.tier)}" + (" ⚠" if fc.watch else "")
    return label


tabs = st.tabs(
    [tab_label(h) for h in HORIZONS]
    + ["📊 নির্ভুলতা (রিভিউয়ারদের জন্য)" if is_bn else "📊 Accuracy (for reviewers)"]
    + ["🚑 কমান্ড সেন্টার" if is_bn else "🚑 Command Center"]
)
for h, tab in zip(HORIZONS, tabs):
    with tab:
        if not overview:
            render_ward_week(h, w_weeks, w_X, w_fcs[h])
        elif manual_mode:
            if is_bn:
                st.info(
                    f"'সব ওয়ার্ড' ভিউতে সংরক্ষিত ডেটা ({data_end:%d %b %Y} পর্যন্ত) দেখানো হয়। "
                    "পরবর্তী তারিখের পূর্বাভাস দেখতে সাইডবার থেকে যেকোনো একটি ওয়ার্ড বেছে তার সাম্প্রতিক আবহাওয়া লিখুন।"
                )
            else:
                st.info(
                    f"The all-wards view uses recorded data (up to {data_end:%d %b %Y}). "
                    "For later dates, pick one ward and enter its recent weather."
                )
        else:
            render_horizon(h)
            render_map_and_dispatch(h, week_tbl, features, issue_week, is_bn)

with tabs[-2]:
    render_accuracy()

with tabs[-1]:
    render_command_center(None if manual_mode else issue_week, city, is_bn, data_end)
    render_historical_replay_and_ethics(preds, features, is_bn)