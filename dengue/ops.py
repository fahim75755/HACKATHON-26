"""Operational decision layer for AlertAhead:
- 60-ward city coordinates for interactive Street/Satellite maps
- Smart Team Allocator with Human-in-the-Loop overrides & vulnerability mode
- Historical backtest replay (Even Split vs AlertAhead vs Perfect Foresight)
- Live 7-day weather via Open-Meteo with guaranteed offline fallback
- Lightweight Bangla/English field-report keyword flagging
- Out-of-range weather guardrails & Bangla SMS alert builder
"""
from __future__ import annotations

import datetime as dt
import json
import urllib.request
from typing import Iterable

import numpy as np
import pandas as pd

from dengue.tiers import TIER_META, assign_risk_tier

# ------------------------------------------------------------------ 1. Ward coordinates (urban centroids)
CITY_CENTERS: dict[str, tuple[float, float]] = {
    "Dhaka North": (23.8150, 90.4020),
    "Dhaka South": (23.7320, 90.4100),
    "Chattogram": (22.3569, 91.8150),
    "Gazipur": (23.9920, 90.4150),
    "Narayanganj": (23.6280, 90.5000),
}

# Realistic non-overlapping urban ward reference points inside each city boundary
WARD_COORDS: dict[str, tuple[float, float]] = {
    # Chattogram (10 wards — urban core & inland neighborhoods)
    "CCC_Ward_01": (22.3920, 91.8120),
    "CCC_Ward_02": (22.3820, 91.8260),
    "CCC_Ward_03": (22.3710, 91.8050),
    "CCC_Ward_04": (22.3660, 91.8320),
    "CCC_Ward_05": (22.3580, 91.8190),
    "CCC_Ward_06": (22.3520, 91.8420),
    "CCC_Ward_07": (22.3450, 91.8080),
    "CCC_Ward_08": (22.3390, 91.8290),
    "CCC_Ward_09": (22.3310, 91.7960),
    "CCC_Ward_10": (22.3260, 91.8160),
    # Dhaka North (20 wards — Uttara, Mirpur, Gulshan, Banani, Mohammadpur, Badda)
    "DNCC_Ward_01": (23.8740, 90.3980),
    "DNCC_Ward_02": (23.8640, 90.3820),
    "DNCC_Ward_03": (23.8580, 90.4120),
    "DNCC_Ward_04": (23.8480, 90.3690),
    "DNCC_Ward_05": (23.8420, 90.3920),
    "DNCC_Ward_06": (23.8360, 90.4220),
    "DNCC_Ward_07": (23.8290, 90.3620),
    "DNCC_Ward_08": (23.8240, 90.3810),
    "DNCC_Ward_09": (23.8190, 90.4060),
    "DNCC_Ward_10": (23.8140, 90.4280),
    "DNCC_Ward_11": (23.8080, 90.3580),
    "DNCC_Ward_12": (23.8030, 90.3760),
    "DNCC_Ward_13": (23.7980, 90.3990),
    "DNCC_Ward_14": (23.7940, 90.4190),
    "DNCC_Ward_15": (23.7880, 90.3640),
    "DNCC_Ward_16": (23.7830, 90.3860),
    "DNCC_Ward_17": (23.7790, 90.4110),
    "DNCC_Ward_18": (23.7750, 90.4310),
    "DNCC_Ward_19": (23.7710, 90.3680),
    "DNCC_Ward_20": (23.7680, 90.3950),
    # Dhaka South (20 wards — Dhanmondi, Ramna, Motijheel, Khilgaon, Lalbagh, Jatrabari)
    "DSCC_Ward_01": (23.7580, 90.3720),
    "DSCC_Ward_02": (23.7560, 90.3920),
    "DSCC_Ward_03": (23.7550, 90.4140),
    "DSCC_Ward_04": (23.7530, 90.4340),
    "DSCC_Ward_05": (23.7470, 90.3780),
    "DSCC_Ward_06": (23.7450, 90.4010),
    "DSCC_Ward_07": (23.7440, 90.4220),
    "DSCC_Ward_08": (23.7410, 90.4410),
    "DSCC_Ward_09": (23.7360, 90.3840),
    "DSCC_Ward_10": (23.7350, 90.4080),
    "DSCC_Ward_11": (23.7330, 90.4290),
    "DSCC_Ward_12": (23.7270, 90.3760),
    "DSCC_Ward_13": (23.7250, 90.3970),
    "DSCC_Ward_14": (23.7240, 90.4170),
    "DSCC_Ward_15": (23.7220, 90.4380),
    "DSCC_Ward_16": (23.7160, 90.3880),
    "DSCC_Ward_17": (23.7150, 90.4090),
    "DSCC_Ward_18": (23.7120, 90.4310),
    "DSCC_Ward_19": (23.7070, 90.3990),
    "DSCC_Ward_20": (23.7040, 90.4210),
    # Gazipur (5 wards)
    "GCC_Ward_01": (24.0120, 90.4180),
    "GCC_Ward_02": (23.9980, 90.4020),
    "GCC_Ward_03": (23.9890, 90.4260),
    "GCC_Ward_04": (23.9720, 90.3960),
    "GCC_Ward_05": (23.9560, 90.4140),
    # Narayanganj (5 wards)
    "NCC_Ward_01": (23.6520, 90.4940),
    "NCC_Ward_02": (23.6390, 90.5080),
    "NCC_Ward_03": (23.6270, 90.4960),
    "NCC_Ward_04": (23.6160, 90.5110),
    "NCC_Ward_05": (23.6060, 90.4920),
}


def attach_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Attach latitude/longitude columns for each ward safely."""
    out = df.copy()
    lats, lons = [], []
    for _, r in out.iterrows():
        wid = str(r.get("ward_id", ""))
        city = str(r.get("city", ""))
        lat, lon = WARD_COORDS.get(wid, CITY_CENTERS.get(city, (23.78, 90.40)))
        lats.append(lat)
        lons.append(lon)
    out["lat"] = lats
    out["lon"] = lons
    return out


# ------------------------------------------------------------------ 2. Smart Team Allocator (VIGIA-style)
def allocate_teams(
    df: pd.DataFrame,
    n_teams: int,
    pred_col: str,
    mode: str = "cases",
    locked_wards: Iterable[str] = (),
    excluded_wards: Iterable[str] = (),
) -> dict:
    """Select optimal wards for `n_teams` fogging/larvicide teams.

    Supports:
    - mode="cases": maximizes expected dengue cases covered.
    - mode="vulnerable": prioritizes expected cases weighted by population density & waterlogging risk.
    - Human-in-the-Loop: `locked_wards` are pinned in; `excluded_wards` are skipped.
    """
    if df.empty or n_teams <= 0 or pred_col not in df.columns:
        return {
            "selected": pd.DataFrame(),
            "covered_cases": 0.0,
            "total_cases": 0.0,
            "coverage_pct": 0.0,
            "even_split_pct": 0.0,
            "lift_vs_even": 0.0,
            "teams_assigned": 0,
        }

    work = df.copy()
    work[pred_col] = pd.to_numeric(work[pred_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    total_cases = float(work[pred_col].sum())
    total_wards = len(work)

    excl = set(excluded_wards)
    eligible = work[~work["ward_id"].isin(excl)].copy()
    if eligible.empty:
        return {
            "selected": pd.DataFrame(),
            "covered_cases": 0.0,
            "total_cases": total_cases,
            "coverage_pct": 0.0,
            "even_split_pct": 0.0,
            "lift_vs_even": 0.0,
            "teams_assigned": 0,
        }

    # Priority score
    if mode == "vulnerable" and {"population_density_sqkm", "water_logging_risk_index"}.issubset(eligible.columns):
        dens = pd.to_numeric(eligible["population_density_sqkm"], errors="coerce").fillna(0.0)
        dmax = float(dens.max()) if float(dens.max()) > 0 else 1.0
        wlog = pd.to_numeric(eligible["water_logging_risk_index"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        eligible["priority_score"] = eligible[pred_col] * (1.0 + 0.4 * (dens / dmax) + 0.4 * wlog)
    else:
        eligible["priority_score"] = eligible[pred_col]

    cap = min(int(n_teams), len(eligible))
    lock_set = [w for w in locked_wards if w not in excl and w in set(eligible["ward_id"])]

    locked_df = eligible[eligible["ward_id"].isin(lock_set)].copy()
    locked_df["dispatch_reason"] = "Pinned by Officer"
    locked_df = locked_df.sort_values("priority_score", ascending=False).head(cap)

    remaining_slots = max(0, cap - len(locked_df))
    pool = eligible[~eligible["ward_id"].isin(set(locked_df["ward_id"]))].copy()
    auto_df = pool.sort_values("priority_score", ascending=False).head(remaining_slots)
    auto_df["dispatch_reason"] = (
        "High Vulnerability + Risk" if mode == "vulnerable" else "Highest Expected Cases"
    )

    selected = pd.concat([locked_df, auto_df], ignore_index=True)
    selected["tier"] = [assign_risk_tier(v) for v in selected[pred_col]]
    selected["rank"] = np.arange(1, len(selected) + 1)

    covered_cases = float(selected[pred_col].sum())
    coverage_pct = (covered_cases / total_cases * 100.0) if total_cases > 0 else 0.0
    even_split_pct = min(100.0, (len(selected) / total_wards) * 100.0) if total_wards > 0 else 0.0
    lift = (coverage_pct / even_split_pct) if even_split_pct > 0 else 0.0

    return {
        "selected": selected,
        "covered_cases": round(covered_cases, 1),
        "total_cases": round(total_cases, 1),
        "coverage_pct": round(coverage_pct, 1),
        "even_split_pct": round(even_split_pct, 1),
        "lift_vs_even": round(lift, 2),
        "teams_assigned": len(selected),
    }


# ------------------------------------------------------------------ 3. Historical Replay Backtester
def backtest_allocation(
    preds: pd.DataFrame,
    features: pd.DataFrame,
    horizon: int,
    n_teams: int = 10,
    test_year: int = 2025,
    issue_date: pd.Timestamp | None = None,
) -> dict:
    """Compare Even Split vs AlertAhead vs Perfect Foresight (Oracle) on recorded actuals."""
    pred_col = f"pred_h{horizon}"
    actual_col = f"target_h{horizon}"

    merged = preds[["date", "city", "ward_id", pred_col]].merge(
        features[["date", "ward_id", actual_col]],
        on=["date", "ward_id"],
        how="inner",
    )
    merged["target_date"] = pd.to_datetime(merged["date"]) + pd.Timedelta(weeks=horizon)
    merged = merged.dropna(subset=[pred_col, actual_col]).copy()

    if issue_date is not None:
        merged = merged[merged["date"] == pd.Timestamp(issue_date)]
    else:
        merged = merged[merged["target_date"].dt.year == test_year]

    if merged.empty or n_teams <= 0:
        return {
            "weeks": 0,
            "even_split_pct": 0.0,
            "alertahead_pct": 0.0,
            "oracle_pct": 0.0,
            "total_actual": 0.0,
            "alertahead_caught": 0.0,
            "oracle_caught": 0.0,
            "week_detail": pd.DataFrame(),
        }

    total_actual = 0.0
    alertahead_caught = 0.0
    oracle_caught = 0.0
    even_caught = 0.0
    week_count = 0

    for _, grp in merged.groupby("date"):
        wk_total = float(grp[actual_col].sum())
        if wk_total <= 0:
            continue
        k = min(int(n_teams), len(grp))
        week_count += 1
        total_actual += wk_total
        even_caught += wk_total * (k / len(grp))
        alertahead_caught += float(grp.nlargest(k, pred_col)[actual_col].sum())
        oracle_caught += float(grp.nlargest(k, actual_col)[actual_col].sum())

    if total_actual <= 0:
        return {
            "weeks": week_count,
            "even_split_pct": 0.0,
            "alertahead_pct": 0.0,
            "oracle_pct": 0.0,
            "total_actual": 0.0,
            "alertahead_caught": 0.0,
            "oracle_caught": 0.0,
            "week_detail": merged,
        }

    return {
        "weeks": week_count,
        "even_split_pct": round(even_caught / total_actual * 100.0, 1),
        "alertahead_pct": round(alertahead_caught / total_actual * 100.0, 1),
        "oracle_pct": round(oracle_caught / total_actual * 100.0, 1),
        "total_actual": round(total_actual, 1),
        "alertahead_caught": round(alertahead_caught, 1),
        "oracle_caught": round(oracle_caught, 1),
        "week_detail": merged.sort_values(actual_col, ascending=False).reset_index(drop=True),
    }


# ------------------------------------------------------------------ 4. Live Weather (Open-Meteo + Offline Fallback)
OFFLINE_WEATHER_FALLBACK: dict[str, dict] = {
    "Dhaka North": {"rain_7d_mm": 6.4, "avg_temp_c": 24.8, "avg_humid_pct": 71.0},
    "Dhaka South": {"rain_7d_mm": 6.8, "avg_temp_c": 25.0, "avg_humid_pct": 72.0},
    "Chattogram": {"rain_7d_mm": 12.5, "avg_temp_c": 25.4, "avg_humid_pct": 75.0},
    "Gazipur": {"rain_7d_mm": 5.2, "avg_temp_c": 24.3, "avg_humid_pct": 70.0},
    "Narayanganj": {"rain_7d_mm": 7.1, "avg_temp_c": 24.9, "avg_humid_pct": 72.5},
}


def fetch_live_weather(city: str, timeout: float = 3.5) -> dict:
    """Fetch 7-day forecast weather from Open-Meteo (free, no API key) with instant offline fallback."""
    lat, lon = CITY_CENTERS.get(city, CITY_CENTERS["Dhaka North"])
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum"
        "&timezone=Asia%2FDhaka&forecast_days=7"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AlertAhead-EDU-Hackathon/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        daily = payload.get("daily", {})
        rain = [float(x) for x in daily.get("precipitation_sum", []) if x is not None]
        temp = [float(x) for x in daily.get("temperature_2m_mean", []) if x is not None]
        humid = [float(x) for x in daily.get("relative_humidity_2m_mean", []) if x is not None]
        if not (rain and temp and humid):
            raise ValueError("Incomplete Open-Meteo payload")
        return {
            "city": city,
            "rain_7d_mm": round(sum(rain), 1),
            "avg_temp_c": round(float(np.mean(temp)), 1),
            "avg_humid_pct": round(float(np.mean(humid)), 1),
            "source": "live_open_meteo",
            "fetched_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    except Exception:
        fb = OFFLINE_WEATHER_FALLBACK.get(city, OFFLINE_WEATHER_FALLBACK["Dhaka North"])
        return {
            "city": city,
            **fb,
            "source": "offline_cache",
            "fetched_at": "Offline backup",
        }


# ------------------------------------------------------------------ 5. Quick Field Report Keyword Flag
HIGH_RISK_KEYWORDS: dict[str, str] = {
    "লার্ভা": "Larvae spotted",
    "larva": "Larvae spotted",
    "larvae": "Larvae spotted",
    "জলাবদ্ধতা": "Waterlogging",
    "জমা পানি": "Stagnant water",
    "পানি জমে": "Stagnant water",
    "waterlog": "Waterlogging",
    "stagnant": "Stagnant water",
    "নির্মাণাধীন": "Construction site water",
    "construction": "Construction site water",
    "ড্রেন": "Blocked drain",
    "drain": "Blocked drain",
    "টায়ার": "Standing water in tires",
    "tire": "Standing water in tires",
    "মশা": "High mosquito density",
    "mosquito": "High mosquito density",
    "ডেঙ্গু": "Suspected dengue cases",
    "dengue": "Suspected dengue cases",
}


def flag_field_report(note: str) -> dict:
    """Detect vector-breeding risk keywords in Bangla/English field notes (never alters ML prediction)."""
    text = (note or "").strip().lower()
    matched: list[str] = []
    for kw, tag in HIGH_RISK_KEYWORDS.items():
        if kw in text and tag not in matched:
            matched.append(tag)
    if len(matched) >= 2:
        level = "Urgent Citizen Signal"
    elif len(matched) == 1:
        level = "Citizen Signal"
    else:
        level = "General Note"
    return {"flagged": len(matched) > 0, "level": level, "signals": matched}


# ------------------------------------------------------------------ 6. Out-of-Range Guardrail & SMS Builder
def check_weather_bounds(
    rain_mm: float,
    humid_pct: float,
    temp_c: float,
    bounds: dict | None = None,
) -> list[str]:
    """Return human-readable warnings if what-if weather falls outside historical training range."""
    b = bounds or {
        "rain": (0.0, 450.0),
        "humid": (40.0, 98.0),
        "temp": (14.0, 36.0),
    }
    warnings: list[str] = []
    if rain_mm < b["rain"][0] or rain_mm > b["rain"][1]:
        warnings.append(f"Rainfall ({rain_mm:.1f} mm) is outside training range {b['rain'][0]}–{b['rain'][1]} mm.")
    if humid_pct < b["humid"][0] or humid_pct > b["humid"][1]:
        warnings.append(f"Humidity ({humid_pct:.1f}%) is outside training range {b['humid'][0]}–{b['humid'][1]}%.")
    if temp_c < b["temp"][0] or temp_c > b["temp"][1]:
        warnings.append(f"Temperature ({temp_c:.1f}°C) is outside training range {b['temp'][0]}–{b['temp'][1]}°C.")
    return warnings


def build_sms_preview(selected_df: pd.DataFrame, horizon: int, pred_col: str, is_bn: bool = True) -> str:
    """Generate a concise dispatch SMS for field supervisors covering the top priority wards."""
    if selected_df.empty:
        return "কোনো ওয়ার্ড নির্বাচিত হয়নি।" if is_bn else "No wards selected."
    top = selected_df.head(5)
    ward_list = ", ".join(f"{r['ward_id']} ({r[pred_col]:.0f})" for _, r in top.iterrows())
    extra = len(selected_df) - len(top)
    if is_bn:
        more = f" ও আরও {extra}টি ওয়ার্ড" if extra > 0 else ""
        return (
            f"[AlertAhead জরুরি সতর্কতা] আগামী {horizon} সপ্তাহে ডেঙ্গু ঝুঁকি বেশি: "
            f"{ward_list}{more}। আজই অগ্রাধিকার ভিত্তিতে লার্ভিসাইড ও ফগিং টিম পাঠান।"
        )
    more = f" (+{extra} more)" if extra > 0 else ""
    return (
        f"[AlertAhead Dispatch Alert] High dengue risk in +{horizon}w: "
        f"{ward_list}{more}. Prioritize larvicide & fogging deployment today."
    )


def tier_color(tier: str) -> str:
    return TIER_META.get(tier, TIER_META["Low"])["color"]