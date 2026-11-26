"""Interactive map, VIGIA-style team allocator, historical replay, live weather & field-report UI.

Designed as modular plug-ins for app.py with full English/Bangla support and zero extra dependencies.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dengue.ops import (
    CITY_CENTERS,
    allocate_teams,
    attach_coordinates,
    backtest_allocation,
    build_sms_preview,
    check_weather_bounds,
    fetch_live_weather,
    flag_field_report,
    tier_color,
)
from dengue.tiers import TIERS, assign_risk_tier

TIER_BN = {"Low": "কম", "Medium": "মাঝারি", "High": "উচ্চ", "Critical": "চরম"}
ESRI_SATELLITE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)


def _tier_label(tier: str, is_bn: bool) -> str:
    return TIER_BN.get(tier, tier) if is_bn else tier


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_live_weather(city: str) -> dict:
    return fetch_live_weather(city)


def _merge_ward_traits(week_df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    needed = ["population_density_sqkm", "water_logging_risk_index", "sanitation_score"]
    missing = [c for c in needed if c not in week_df.columns]
    if not missing or features.empty:
        return week_df.copy()
    traits = (
        features[["ward_id", *missing]]
        .drop_duplicates(subset=["ward_id"], keep="last")
    )
    return week_df.merge(traits, on="ward_id", how="left")


def _build_ward_map(
    df: pd.DataFrame,
    pred_col: str,
    map_style: str,
    selected_wards: set[str],
    signal_map: dict[str, list[str]],
    is_bn: bool,
) -> go.Figure:
    mapped = attach_coordinates(df)
    mapped["cases_val"] = pd.to_numeric(mapped[pred_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    mapped["tier"] = [assign_risk_tier(v) for v in mapped["cases_val"]]
    mapped["in_dispatch"] = mapped["ward_id"].isin(selected_wards)

    fig = go.Figure()
    for tier in TIERS:
        sub = mapped[mapped["tier"] == tier]
        if sub.empty:
            continue
        sizes = (11 + sub["cases_val"].clip(upper=30) * 0.75).tolist()
        hover_texts = []
        for _, r in sub.iterrows():
            wid = str(r["ward_id"])
            sigs = signal_map.get(wid, [])
            sig_line = (
                f"<br><b>{'নাগরিক সিগন্যাল' if is_bn else 'Citizen Signal'}:</b> {', '.join(sigs)}"
                if sigs
                else ""
            )
            disp_line = (
                f"<br><b>{'টিম বরাদ্দ' if is_bn else 'Dispatch'}:</b> "
                + ("হ্যাঁ (অগ্রাধিকার)" if is_bn else "YES (Priority)")
                if r["in_dispatch"]
                else ""
            )
            hover_texts.append(
                f"<b>{wid}</b> ({r['city']})<br>"
                f"{'সম্ভাব্য কেস' if is_bn else 'Expected cases'}: <b>{r['cases_val']:.1f}</b><br>"
                f"{'ঝুঁকির স্তর' if is_bn else 'Risk tier'}: <b>{_tier_label(tier, is_bn)}</b>"
                f"{disp_line}{sig_line}"
            )
        fig.add_trace(
            go.Scattermap(
                lat=sub["lat"],
                lon=sub["lon"],
                mode="markers",
                name=_tier_label(tier, is_bn),
                marker=dict(size=sizes, color=tier_color(tier), opacity=0.9),
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    # Highlight wards that have an active Citizen Signal from field reports
    flagged_df = mapped[mapped["ward_id"].isin(signal_map.keys())]
    if not flagged_df.empty:
        fig.add_trace(
            go.Scattermap(
                lat=flagged_df["lat"],
                lon=flagged_df["lon"],
                mode="markers",
                name="নাগরিক সিগন্যাল" if is_bn else "Citizen Signal",
                marker=dict(size=22, color="#1D4ED8", opacity=0.45),
                hoverinfo="skip",
            )
        )

    center_lat = float((mapped["lat"].min() + mapped["lat"].max()) / 2) if not mapped.empty else 23.78
    center_lon = float((mapped["lon"].min() + mapped["lon"].max()) / 2) if not mapped.empty else 90.40
    cities = mapped["city"].nunique() if "city" in mapped.columns else 1
    zoom = 5.9 if cities > 2 else 10.6

    if map_style == "satellite":
        map_cfg = dict(
            style="white-bg",
            layers=[
                {
                    "below": "traces",
                    "sourcetype": "raster",
                    "sourceattribution": "Esri World Imagery",
                    "source": [ESRI_SATELLITE_URL],
                }
            ],
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom,
        )
    elif map_style == "light":
        map_cfg = dict(
            style="carto-positron",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom,
        )
    else:
        map_cfg = dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom,
        )

    fig.update_layout(
        map=map_cfg,
        height=430,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


def render_map_and_dispatch(
    h: int,
    week_df: pd.DataFrame,
    features: pd.DataFrame,
    issue_week: pd.Timestamp,
    is_bn: bool,
) -> None:
    """Render 60-ward Street/Satellite Risk Map, Smart Team Allocator + Audit Log, Live Weather & Field Reports."""
    pred_col = f"pred_h{h}"
    if week_df.empty or pred_col not in week_df.columns:
        return

    st.session_state.setdefault("dispatch_audit_log", [])
    st.session_state.setdefault("field_reports", [])

    enriched = _merge_ward_traits(week_df, features)
    all_ward_ids = sorted(enriched["ward_id"].astype(str).unique().tolist())

    # Collect active citizen signals per ward
    signal_map: dict[str, list[str]] = {}
    for rep in st.session_state["field_reports"]:
        if rep.get("flagged"):
            signal_map.setdefault(rep["ward_id"], [])
            for s in rep.get("signals", []):
                if s not in signal_map[rep["ward_id"]]:
                    signal_map[rep["ward_id"]].append(s)

    st.divider()
    st.subheader(
        f"🗺️ ওয়ার্ড রিস্ক ম্যাপ ও স্মার্ট টিম ডিসপ্যাচ (+{h} সপ্তাহ)"
        if is_bn
        else f"🗺️ Ward Risk Map & Smart Team Dispatch (+{h} weeks)"
    )

    # ---------------------------------------------------------- Smart Team Allocator Controls (VIGIA-style)
    c1, c2, c3 = st.columns([1.2, 1.3, 1.5])
    max_teams = max(1, min(25, len(enriched)))
    default_teams = min(10, max_teams)
    with c1:
        n_teams = st.slider(
            "ফগিং/সার্ভে টিমের সংখ্যা" if is_bn else "Available fogging/survey teams",
            min_value=1,
            max_value=max_teams,
            value=default_teams,
            key=f"ops_teams_h{h}",
        )
    with c2:
        mode_choice = st.radio(
            "অগ্রাধিকারের ধরন" if is_bn else "Allocation strategy",
            options=["cases", "vulnerable"],
            format_func=lambda x: (
                ("সর্বোচ্চ সম্ভাব্য রোগী কভার" if x == "cases" else "ঝুঁকিপূর্ণ এলাকা (ঘনত্ব + জলাবদ্ধতা)")
                if is_bn
                else ("Maximize Cases Covered" if x == "cases" else "Vulnerable Zones (Density + Waterlogging)")
            ),
            horizontal=False,
            key=f"ops_mode_h{h}",
        )
    with c3:
        map_style = st.radio(
            "ম্যাপ ভিউ" if is_bn else "Map style",
            options=["street", "satellite", "light"],
            format_func=lambda x: (
                {"street": "স্ট্রিট ম্যাপ", "satellite": "স্যাটেলাইট", "light": "হালকা"}[x]
                if is_bn
                else {"street": "Street", "satellite": "Satellite", "light": "Clean Light"}[x]
            ),
            horizontal=True,
            key=f"ops_mapstyle_h{h}",
        )

    ov1, ov2 = st.columns(2)
    with ov1:
        locked_wards = st.multiselect(
            "আবশ্যিকভাবে রাখুন (Officer Pin)" if is_bn else "Pin wards into plan (Officer Override)",
            options=all_ward_ids,
            default=[],
            key=f"ops_lock_h{h}",
        )
    with ov2:
        excluded_wards = st.multiselect(
            "সাময়িকভাবে বাদ দিন (ইতোমধ্যে ফগিং হয়েছে / রাস্তা বন্ধ)"
            if is_bn
            else "Exclude wards (already treated / blocked)",
            options=all_ward_ids,
            default=[],
            key=f"ops_excl_h{h}",
        )

    plan = allocate_teams(
        enriched,
        n_teams=n_teams,
        pred_col=pred_col,
        mode=mode_choice,
        locked_wards=locked_wards,
        excluded_wards=excluded_wards,
    )
    selected_df = plan["selected"]
    selected_set = set(selected_df["ward_id"].tolist()) if not selected_df.empty else set()

    # ---------------------------------------------------------- KPI Strip
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "বরাদ্দকৃত টিম" if is_bn else "Teams Dispatched",
        f"{plan['teams_assigned']} / {len(enriched)}",
    )
    k2.metric(
        "AlertAhead কভারেজ" if is_bn else "AlertAhead Case Coverage",
        f"{plan['coverage_pct']:.1f}%",
        delta=f"+{plan['coverage_pct'] - plan['even_split_pct']:.1f}% vs {'সমান বণ্টন' if is_bn else 'even split'}",
    )
    k3.metric(
        "কার্যকারিতা বৃদ্ধি" if is_bn else "Efficiency Lift",
        f"{plan['lift_vs_even']:.2f}x",
        help=(
            "র‍্যান্ডম বা সমান ভাগে টিম পাঠানোর তুলনায় কত গুণ বেশি সম্ভাব্য কেস কভার হচ্ছে।"
            if is_bn
            else "How many times more expected cases are covered vs. an even/random split."
        ),
    )
    k4.metric(
        "সম্ভাব্য কভারকৃত রোগী" if is_bn else "Expected Cases Covered",
        f"{plan['covered_cases']:.0f} / {plan['total_cases']:.0f}",
    )

    # ---------------------------------------------------------- Map + Priority Table
    m_col, t_col = st.columns([1.35, 1.0])
    with m_col:
        fig_map = _build_ward_map(enriched, pred_col, map_style, selected_set, signal_map, is_bn)
        st.plotly_chart(fig_map, key=f"ops_map_chart_h{h}")
        st.caption(
            "ম্যাপের প্রতিটি পয়েন্টের আকার সম্ভাব্য ডেঙ্গু রোগীর সংখ্যা এবং রঙ ঝুঁকির স্তর নির্দেশ করে। "
            "কোনো ওয়ার্ডে নাগরিক রিপোর্ট পাওয়া গেলে নীল বৃত্ত দেখাবে।"
            if is_bn
            else "Marker size reflects expected cases and color indicates risk tier. "
                 "Blue halo marks wards with active citizen field signals."
        )

    with t_col:
        st.markdown(
            "**অগ্রাধিকার ভিত্তিতে টিম পাঠানোর তালিকা**" if is_bn else "**Priority Dispatch Queue**"
        )
        if not selected_df.empty:
            disp = selected_df[["rank", "ward_id", "city", pred_col, "tier", "dispatch_reason"]].copy()
            disp[pred_col] = disp[pred_col].round(1)
            disp["tier"] = [_tier_label(x, is_bn) for x in disp["tier"]]
            disp.columns = (
                ["ক্রম", "ওয়ার্ড", "শহর", "সম্ভাব্য কেস", "স্তর", "কারণ"]
                if is_bn
                else ["Rank", "Ward", "City", "Pred. Cases", "Tier", "Reason"]
            )
            st.dataframe(disp, hide_index=True, height=330)
        else:
            st.warning("কোনো ওয়ার্ড নির্বাচিত হয়নি।" if is_bn else "No eligible wards selected.")

        # Human-in-the-Loop Approval + CSV Download
        b1, b2 = st.columns(2)
        with b1:
            if st.button(
                "✅ প্ল্যান অনুমোদন ও লগ করুন" if is_bn else "✅ Approve & Log Plan",
                key=f"ops_approve_h{h}",
                type="primary",
            ):
                entry = {
                    "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "issue_week": str(pd.Timestamp(issue_week).date()),
                    "horizon": f"+{h}w",
                    "strategy": mode_choice,
                    "teams": plan["teams_assigned"],
                    "coverage_pct": plan["coverage_pct"],
                    "pinned": ", ".join(locked_wards) or "-",
                    "excluded": ", ".join(excluded_wards) or "-",
                    "top_wards": ", ".join(selected_df["ward_id"].head(5).tolist()) if not selected_df.empty else "-",
                }
                st.session_state["dispatch_audit_log"].insert(0, entry)
                st.success(
                    "ডিসপ্যাচ প্ল্যান অনুমোদিত এবং অডিট লগে সংরক্ষিত হয়েছে!"
                    if is_bn
                    else "Dispatch plan approved and recorded in Audit Log!"
                )
        with b2:
            if not selected_df.empty:
                csv_bytes = selected_df[
                    ["rank", "ward_id", "city", pred_col, "tier", "dispatch_reason"]
                ].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ CSV ডাউনলোড" if is_bn else "⬇️ Export CSV",
                    data=csv_bytes,
                    file_name=f"alertahead_dispatch_h{h}_{pd.Timestamp(issue_week):%Y%m%d}.csv",
                    mime="text/csv",
                    key=f"ops_csv_h{h}",
                )

    # ---------------------------------------------------------- SMS Preview & Audit Log
    sms_text = build_sms_preview(selected_df, horizon=h, pred_col=pred_col, is_bn=is_bn)
    with st.expander(
        "📩 মাঠপর্যায়ের SMS অ্যালার্ট প্রিভিউ ও অনুমোদনের অডিট লগ"
        if is_bn
        else "📩 Field Supervisor SMS Preview & Human Approval Audit Log"
    ):
        st.code(sms_text, language=None)
        if st.session_state["dispatch_audit_log"]:
            log_df = pd.DataFrame(st.session_state["dispatch_audit_log"])
            st.dataframe(log_df, hide_index=True)
        else:
            st.caption(
                "এখনও কোনো প্ল্যান অনুমোদন করা হয়নি। ওপরে 'প্ল্যান অনুমোদন ও লগ করুন' বাটনে ক্লিক করলে এখানে অডিট লগ জমবে।"
                if is_bn
                else "No plans approved yet in this session. Click 'Approve & Log Plan' above to record an audit entry."
            )

    # ---------------------------------------------------------- Live Weather + Quick Field Report
    with st.expander(
        "🌦️ লাইভ ৭ দিনের আবহাওয়া (Open-Meteo) ও কুইক ফিল্ড রিপোর্ট"
        if is_bn
        else "🌦️ Live 7-Day Weather (Open-Meteo + Offline Fallback) & Quick Field Report"
    ):
        w_col, f_col = st.columns(2)
        with w_col:
            st.markdown(
                "**লাইভ আবহাওয়া ও আউট-অফ-রেঞ্জ গার্ডরেইল**"
                if is_bn
                else "**Live Weather & Out-of-Range Guardrail**"
            )
            city_pick = st.selectbox(
                "শহর (লাইভ আবহাওয়া)" if is_bn else "City (Live Weather)",
                options=list(CITY_CENTERS.keys()),
                key=f"ops_wcity_h{h}",
            )
            if st.button("🔄 লাইভ আবহাওয়া দেখুন" if is_bn else "🔄 Fetch Live 7-Day Weather", key=f"ops_wbtn_h{h}"):
                st.session_state[f"ops_weather_last_h{h}"] = _cached_live_weather(city_pick)
            wdata = st.session_state.get(f"ops_weather_last_h{h}")
            if wdata:
                badge = (
                    "🟢 Live Open-Meteo" if wdata["source"] == "live_open_meteo" else "🟡 Offline Backup Cache"
                )
                st.caption(f"{wdata['city']} · {badge} ({wdata['fetched_at']})")
                wm1, wm2, wm3 = st.columns(3)
                wm1.metric("৭ দিনে বৃষ্টি" if is_bn else "7d Rain", f"{wdata['rain_7d_mm']} mm")
                wm2.metric("গড় তাপমাত্রা" if is_bn else "Avg Temp", f"{wdata['avg_temp_c']}°C")
                wm3.metric("গড় আর্দ্রতা" if is_bn else "Humidity", f"{wdata['avg_humid_pct']}%")
                warns = check_weather_bounds(wdata["rain_7d_mm"], wdata["avg_humid_pct"], wdata["avg_temp_c"])
                if warns:
                    for wmsg in warns:
                        st.warning(wmsg)
                else:
                    st.success(
                        "আবহাওয়ার মান ট্রেনিং রেঞ্জের ভেতরে রয়েছে (নিরাপদ পূর্বাভাস)।"
                        if is_bn
                        else "Weather inputs are within historical training bounds (safe forecast)."
                    )

        with f_col:
            st.markdown(
                "**কুইক ফিল্ড রিপোর্ট (নাগরিক সিগন্যাল)**"
                if is_bn
                else "**Quick Field Report (Citizen Signal)**"
            )
            rep_ward = st.selectbox(
                "ওয়ার্ড নির্বাচন (ফিল্ড রিপোর্ট)" if is_bn else "Ward (Field Report)",
                options=all_ward_ids,
                key=f"ops_rward_h{h}",
            )
            rep_note = st.text_input(
                "মাঠের পর্যবেক্ষণ লিখুন" if is_bn else "Enter ground observation",
                placeholder=(
                    "যেমন: ড্রেনে জমা পানি ও মশার লার্ভা দেখা গেছে"
                    if is_bn
                    else "e.g. Stagnant water and mosquito larvae near construction site"
                ),
                key=f"ops_rnote_h{h}",
            )
            if st.button("📌 রিপোর্ট যোগ করুন" if is_bn else "📌 Log Field Signal", key=f"ops_rbtn_h{h}"):
                if rep_note.strip():
                    flag = flag_field_report(rep_note)
                    st.session_state["field_reports"].insert(
                        0,
                        {
                            "time": dt.datetime.now().strftime("%H:%M"),
                            "ward_id": rep_ward,
                            "note": rep_note.strip(),
                            "level": flag["level"],
                            "flagged": flag["flagged"],
                            "signals": flag["signals"],
                        },
                    )
                    if flag["flagged"]:
                        st.warning(
                            f"🚩 {flag['level']} শনাক্ত হয়েছে ({rep_ward}): {', '.join(flag['signals'])}"
                        )
                    else:
                        st.info(
                            "সাধারণ নোট হিসেবে সংরক্ষিত হয়েছে (মূল ML পূর্বাভাস অপরিবর্তিত)।"
                            if is_bn
                            else "Logged as general note (core ML forecast remains unchanged)."
                        )
            if st.session_state["field_reports"]:
                st.dataframe(
                    pd.DataFrame(st.session_state["field_reports"])[["time", "ward_id", "level", "note"]].head(5),
                    hide_index=True,
                )


def render_historical_replay_and_ethics(
    preds: pd.DataFrame,
    features: pd.DataFrame,
    is_bn: bool,
) -> None:
    """Render VIGIA-style 2025 historical dispatch backtest + Responsible AI / Privacy note."""
    st.divider()
    st.subheader(
        "🔁 ঐতিহাসিক ব্যাকটেস্ট রিপ্লে (২০২৫ বাস্তব ডেটায় টিম ডিসপ্যাচ প্রমাণ)"
        if is_bn
        else "🔁 Historical Dispatch Replay (2025 Unseen Test-Year Proof)"
    )
    st.caption(
        "অতীতের অদেখা ২০২৫ সালের প্রতিটি সপ্তাহে সমানভাবে টিম পাঠানো বনাম AlertAhead-এর ডিসপ্যাচ প্ল্যান বনাম "
        "ভবিষ্যৎ জানা থাকলে সর্বোচ্চ সম্ভব কভারেজের (Oracle Ceiling) সরাসরি তুলনা।"
        if is_bn
        else "Compares Even Split vs. AlertAhead Dispatch vs. Perfect Foresight (Oracle Ceiling) "
             "on unseen 2025 recorded dengue cases."
    )

    rc1, rc2, rc3 = st.columns([1.0, 1.0, 1.4])
    with rc1:
        bt_h = st.selectbox(
            "লিড টাইম (ব্যাকটেস্ট)" if is_bn else "Lead time (backtest)",
            options=[2, 3, 4],
            format_func=lambda x: f"+{x} সপ্তাহ" if is_bn else f"+{x} weeks",
            key="ops_bt_h",
        )
    with rc2:
        bt_teams = st.slider(
            "টিম সংখ্যা (ব্যাকটেস্ট)" if is_bn else "Teams (backtest)",
            min_value=5,
            max_value=20,
            value=10,
            step=5,
            key="ops_bt_teams",
        )

    # Available active 2025 issue weeks
    valid_weeks = sorted(
        preds.loc[
            (pd.to_datetime(preds["date"]) + pd.Timedelta(weeks=bt_h)).dt.year == 2025,
            "date",
        ].unique().tolist()
    )
    week_options = ["ALL_2025"] + [pd.Timestamp(d) for d in valid_weeks]
    with rc3:
        wk_pick = st.selectbox(
            "সপ্তাহ নির্বাচন" if is_bn else "Replay scope",
            options=week_options,
            format_func=lambda x: (
                ("সম্পূর্ণ ২০২৫ সিজন (সব সক্রিয় সপ্তাহ)" if is_bn else "Full 2025 Season (all active weeks)")
                if x == "ALL_2025"
                else f"{pd.Timestamp(x):%d %b %Y}"
            ),
            key="ops_bt_week",
        )

    bt = backtest_allocation(
        preds=preds,
        features=features,
        horizon=bt_h,
        n_teams=bt_teams,
        test_year=2025,
        issue_date=None if wk_pick == "ALL_2025" else wk_pick,
    )

    bm1, bm2, bm3 = st.columns(3)
    bm1.metric(
        "সমান বণ্টন (বেসলাইন)" if is_bn else "Even Split (Baseline)",
        f"{bt['even_split_pct']:.1f}%",
    )
    bm2.metric(
        "AlertAhead ডিসপ্যাচ" if is_bn else "AlertAhead Plan",
        f"{bt['alertahead_pct']:.1f}%",
        delta=f"+{bt['alertahead_pct'] - bt['even_split_pct']:.1f}%",
    )
    bm3.metric(
        "সর্বোচ্চ তাত্ত্বিক সীমা (Oracle)" if is_bn else "Perfect Foresight (Ceiling)",
        f"{bt['oracle_pct']:.1f}%",
    )

    labels = (
        ["সমান বণ্টন", "AlertAhead প্ল্যান", "সর্বোচ্চ সীমা (Oracle)"]
        if is_bn
        else ["Even Split", "AlertAhead Plan", "Perfect Foresight (Oracle)"]
    )
    vals = [bt["even_split_pct"], bt["alertahead_pct"], bt["oracle_pct"]]
    fig_bt = go.Figure(
        go.Bar(
            x=labels,
            y=vals,
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside",
            marker_color=["#94A3B8", "#0F4C5C", "#2E938B"],
        )
    )
    fig_bt.update_layout(
        height=270,
        margin=dict(l=14, r=14, t=24, b=14),
        yaxis=dict(
            title="বাস্তব রোগী কভারেজ (%)" if is_bn else "Share of Real Cases Covered (%)",
            range=[0, max(vals + [35]) * 1.25],
            gridcolor="#EAF0EF",
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    st.plotly_chart(fig_bt, key="ops_bt_chart")

    if wk_pick != "ALL_2025" and not bt["week_detail"].empty:
        pred_col = f"pred_h{bt_h}"
        act_col = f"target_h{bt_h}"
        top_rep = bt["week_detail"].sort_values(pred_col, ascending=False).head(bt_teams)[
            ["ward_id", "city", pred_col, act_col]
        ].copy()
        top_rep[pred_col] = top_rep[pred_col].round(1)
        top_rep[act_col] = top_rep[act_col].round(0).astype(int)
        top_rep.columns = (
            ["ওয়ার্ড", "শহর", "পূর্বাভাস", "বাস্তব রোগী"]
            if is_bn
            else ["Ward", "City", "Predicted", "Actual Cases"]
        )
        st.dataframe(top_rep, hide_index=True)

    # ---------------------------------------------------------- Responsible AI & Privacy-by-Design Note
    st.info(
        "🛡️ **দায়িত্বশীল AI ও গোপনীয়তা নোট:** এই সিস্টেমটি শুধুমাত্র ওয়ার্ড-ভিত্তিক অ্যাগ্রিগেটেড জনস্বাস্থ্য ও আবহাওয়া ডেটা ব্যবহার করে—কোনো ব্যক্তিগত রোগীর তথ্য সংগ্রহ বা সংরক্ষণ করা হয় না। "
        "AlertAhead একটি সিদ্ধান্ত-সহায়ক টুল (Decision-Support System), মেডিকেল ডায়াগনসিস নয়; প্রতিটি ডিসপ্যাচ প্ল্যানে মাঠপর্যায়ের কর্মকর্তার অনুমোদন ও অডিট লগ বাধ্যতামূলক।"
        if is_bn
        else "🛡️ **Responsible AI & Privacy-by-Design:** AlertAhead uses strictly ward-level aggregated public-health and climate data with zero patient PII. "
             "It is a human-in-the-loop decision-support system (not a medical diagnostic tool); field officers approve and audit every dispatch plan."
    )