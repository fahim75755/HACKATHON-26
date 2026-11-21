import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dengue.features import TARGET_COL, build_features, load_raw_data
from dengue.model import DEFAULT_MODEL_PATH, load_model, predict_row
from dengue.tiers import TIER_META, assign_risk_tier

# --- Page Configuration & Cinematic Theme Styling ---
st.set_page_config(
    page_title="XPO-DENGUE | Urban Early-Warning System",
    page_icon="🦟",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .hero-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        padding: 32px;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        margin-bottom: 25px;
    }
    .stButton>button {
        background-color: #E29578;
        color: #0F172A;
        font-weight: 700;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
    }
    .stButton>button:hover {
        background-color: #FFDDD2;
        color: #0F172A;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_artifact():
    return load_model(DEFAULT_MODEL_PATH)


@st.cache_data
def get_datasets(category_maps: dict):
    raw_df = load_raw_data("data/raw/dengue_data.csv")
    feat_df, _, _, _ = build_features(
        raw_df, category_maps=category_maps, drop_na_lags=True
    )
    return raw_df, feat_df


artifact = get_artifact()
raw_df, feat_df = get_datasets(artifact["category_maps"])
weather_cols = artifact["weather_cols"]
rain_col = weather_cols["rainfall"]
humid_col = weather_cols["humidity"]
metrics = artifact["metrics"]

# Define cities list globally so it can be accessed across all tabs
cities = sorted(feat_df["city"].unique())

# --- Top Header & Language Toggle (EN / বাংলা) ---
top_col1, top_col2 = st.columns([5, 1])
with top_col1:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 14px;">
            <span style="font-size: 2.4rem;">🦟</span>
            <div>
                <h2 style="margin: 0; color: #F8FAFC; font-family: sans-serif; font-weight: 700;">XPO-DENGUE | Early-Warning Engine</h2>
                <p style="margin: 0; color: #94A3B8; font-size: 0.95rem;">Ward-level outbreak prediction powered by LightGBM & 1–4 week weather lags</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with top_col2:
    lang = st.selectbox("Language", ["English", "বাংলা"], label_visibility="collapsed")

is_bn = lang == "বাংলা"

# --- Hero Cinematic Banner ---
st.markdown(
    f"""
    <div class="hero-card">
        <h1 style="color: #FFDDD2; font-size: 2.4rem; margin-bottom: 10px; font-family: sans-serif;">
            { 'পূর্বাভাস দিন। প্রতিরোধ করুন। জীবন বাঁচান।' if is_bn else 'PREDICT. PREVENT. PROTECT. DENGUE.' }
        </h1>
        <p style="color: #CBD5E1; font-size: 1.05rem; line-height: 1.6; max-width: 950px; margin: 0;">
            { 'আমাদের উন্নত এআই ইঞ্জিন ও মাল্টি-উইক ল্যাগ পাইপলাইন ব্যবহার করে ২ থেকে ৪ সপ্তাহ আগেই উচ্চঝুঁকিপূর্ণ ওয়ার্ডগুলো শনাক্ত করুন। সাধারণ নাগরিক ও স্বাস্থ্যকর্মীদের জন্য তৈরি রিয়াল-টাইম ড্যাশবোর্ড।' if is_bn else 'Welcome to the Dengue Early Warning Dashboard. Utilizing multi-week environmental lag features and advanced LightGBM modeling, our localized engine provides robust 2-4 week outbreak forecasting to safeguard urban communities.' }
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar Controls (Strictly EN or BN) ---
st.sidebar.header("⚙️ " + ("এলাকা ও আবহাওয়া সিমুলেটর" if is_bn else "Area & Climate Simulator"))

selected_city = st.sidebar.selectbox("শহর নির্বাচন করুন" if is_bn else "Select City", cities)
city_wards = sorted(feat_df.loc[feat_df["city"] == selected_city, "ward_id"].unique())
selected_ward = st.sidebar.selectbox("ওয়ার্ড নির্বাচন করুন" if is_bn else "Select Ward", city_wards)

available_dates = sorted(feat_df["date"].dt.strftime("%Y-%m-%d").unique())
selected_date_str = st.sidebar.selectbox(
    "পূর্বাভাসের সপ্তাহের তারিখ" if is_bn else "Forecast Week Date", available_dates, index=len(available_dates) - 1
)
selected_ts = pd.Timestamp(selected_date_str)

ward_static = raw_df[raw_df["ward_id"] == selected_ward].iloc[-1]
base_wl = float(ward_static["water_logging_risk_index"])
base_san = float(ward_static["sanitation_score"])
base_pop = int(ward_static["population_density_sqkm"])

st.sidebar.divider()
st.sidebar.subheader("⚡ " + ("পরিবেশ ও আবহাওয়া পরিবর্তন" if is_bn else "Climate Shock Simulator"))
st.sidebar.caption("স্লাইডার বা ইনপুট বক্সে মান পরিবর্তন করে প্রভাব দেখুন।" if is_bn else "Adjust sliders or input boxes to simulate shocks.")

rain_pct = st.sidebar.slider(
    "বৃষ্টির পরিমাণ পরিবর্তন (%)" if is_bn else "Rainfall Increase/Decrease (%)", 
    -50, 150, 0, step=5
)
rain_pct = st.sidebar.number_input(
    "সঠিক বৃষ্টির পরিবর্তন (%) লিখুন" if is_bn else "Enter exact rainfall change (%)", 
    -50, 150, rain_pct, step=1
)

temp_delta = st.sidebar.slider(
    "তাপমাত্রার তারতম্য (°C)" if is_bn else "Temperature Shift (°C)", 
    -3.0, 5.0, 0.0, step=0.5
)
temp_delta = st.sidebar.number_input(
    "সঠিক তাপমাত্রার পরিবর্তন লিখুন" if is_bn else "Enter exact temperature shift (°C)", 
    -3.0, 5.0, temp_delta, step=0.1
)

humid_delta = st.sidebar.slider(
    "আর্দ্রতার পরিবর্তন (%)" if is_bn else "Humidity Shift (%)", 
    -20, 20, 0, step=1
)
humid_delta = st.sidebar.number_input(
    "সঠিক আর্দ্রতার পরিবর্তন লিখুন" if is_bn else "Enter exact humidity shift (%)", 
    -20, 20, humid_delta, step=1
)

wl_override = st.sidebar.slider(
    "জলাবদ্ধতার ঝুঁকি (০ থেকে ১)" if is_bn else "Water-Logging Risk (0 to 1)", 
    0.0, 1.0, round(base_wl, 2), step=0.05, key=f"wl_{selected_ward}"
)
wl_override = st.sidebar.number_input(
    "সঠিক জলাবদ্ধতা সূচক লিখুন" if is_bn else "Enter exact water-logging index", 
    0.0, 1.0, float(wl_override), step=0.01, key=f"wl_num_{selected_ward}"
)

san_override = st.sidebar.slider(
    "পরিচ্ছন্নতা স্কোর (০ থেকে ১)" if is_bn else "Sanitation Score (0 to 1)", 
    0.0, 1.0, round(base_san, 2), step=0.05, key=f"san_{selected_ward}"
)
san_override = st.sidebar.number_input(
    "সঠিক পরিচ্ছন্নতা স্কোর লিখুন" if is_bn else "Enter exact sanitation score", 
    0.0, 1.0, float(san_override), step=0.01, key=f"san_num_{selected_ward}"
)

is_what_if = (
    rain_pct != 0
    or temp_delta != 0.0
    or humid_delta != 0
    or abs(wl_override - base_wl) > 1e-6
    or abs(san_override - base_san) > 1e-6
)

ward_slice = (
    raw_df[(raw_df["ward_id"] == selected_ward) & (raw_df["date"] <= selected_ts)]
    .sort_values("date")
    .tail(8)
    .copy()
)

if is_what_if:
    shock_idx = ward_slice.index[-5:]
    ward_slice.loc[shock_idx, rain_col] = np.clip(
        ward_slice.loc[shock_idx, rain_col] * (1.0 + rain_pct / 100.0), 0.0, None
    )
    for t_col in ("avg_temp_c", "max_temp_c", "min_temp_c"):
        ward_slice.loc[shock_idx, t_col] = ward_slice.loc[shock_idx, t_col] + temp_delta
    ward_slice.loc[shock_idx, humid_col] = np.clip(
        ward_slice.loc[shock_idx, humid_col] + humid_delta, 10.0, 100.0
    )
    ward_slice["water_logging_risk_index"] = wl_override
    ward_slice["sanitation_score"] = san_override

sim_feat_df, _, _, _ = build_features(
    ward_slice, category_maps=artifact["category_maps"], drop_na_lags=True
)
target_row = sim_feat_df.iloc[[-1]]
prediction = predict_row(artifact, target_row, top_k=7)
actual_cases = int(ward_slice.iloc[-1][TARGET_COL])

# --- Main Tabs ---
tab1, tab2, tab3 = st.tabs(
    [
        "📍 " + ("ওয়ার্ড পূর্বাভাস ও মূল কারণ" if is_bn else "Ward Forecast & Key Drivers"),
        "🗺️ " + ("৬০-ওয়ার্ড হটস্পট রাডার" if is_bn else "60-Ward Hotspot Radar"),
        "📊 " + ("মডেল পারফরম্যান্স" if is_bn else "Model Performance"),
    ]
)

with tab1:
    if is_what_if:
        sim_msg = f"⚡ সিমুলেশন সক্রিয় — বৃষ্টি {rain_pct:+d}%, তাপমাত্রা {temp_delta:+.1f}°C, আর্দ্রতা {humid_delta:+d}%, জলাবদ্ধতা {wl_override:.2f}" if is_bn else f"⚡ Simulation Active — Rainfall {rain_pct:+d}%, Temp {temp_delta:+.1f}°C, Humidity {humid_delta:+d}%, Water-Logging {wl_override:.2f}"
        st.info(sim_msg)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("আনুমানিক সাপ্তাহিক রোগী" if is_bn else "Predicted Weekly Cases", f"{prediction['predicted_cases']:.1f}")
    c2.metric("৮০% কনফিডেন্স রেঞ্জ" if is_bn else "80% Confidence Range", f"{prediction['ci_low']:.1f} – {prediction['ci_high']:.1f}")
    c3.metric("ঝুঁকির মাত্রা" if is_bn else "Risk Tier", prediction["risk_tier"])
    c4.metric("প্রকৃত রোগী" if is_bn else ("Recorded Actual" if not is_what_if else "Baseline Actual"), f"{actual_cases} cases")

    rec_label = "সুপারিশকৃত পদক্ষেপ:" if is_bn else "Recommended Action:"
    pop_label = "জনঘনত্ব" if is_bn else "Population Density"
    wl_label = "জলাবদ্ধতা সূচক" if is_bn else "Water-Logging Index"
    san_label = "পরিচ্ছন্নতা স্কোর" if is_bn else "Sanitation Score"

    st.markdown(
        f"""
        <div style="padding:16px 20px;border-radius:10px;border-left:6px solid {prediction['tier_color']};background-color:#1E293B;margin:15px 0 20px 0;">
            <strong style="font-size: 1.1rem; color: #F8FAFC;">{selected_ward} ({selected_city}) — {prediction['risk_tier'].upper()} RISK</strong><br/>
            <span style="color: #CBD5E1; display: block; margin-top: 5px;"><strong>{rec_label}</strong> {prediction['recommended_action']}</span>
            <small style="color: #94A3B8; display: block; margin-top: 6px;">{pop_label}: {base_pop:,}/sqkm | {wl_label}: {wl_override:.2f} | {san_label}: {san_override:.2f}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1.15, 1.0])

    with col_left:
        st.subheader("৫২ সপ্তাহের ডেঙ্গু প্রাদুর্ভাবের গতিপথ" if is_bn else "52-Week Outbreak Trajectory")
        hist_df = (
            feat_df[
                (feat_df["ward_id"] == selected_ward) & (feat_df["date"] <= selected_ts)
            ]
            .sort_values("date")
            .tail(52)
            .copy()
        )
        X_hist = hist_df[artifact["features"]]
        hist_df["pred"] = np.clip(artifact["main_model"].predict(X_hist), 0.0, None)
        hist_df["low"] = np.clip(artifact["lower_model"].predict(X_hist), 0.0, None)
        hist_df["high"] = np.maximum(
            hist_df["pred"], np.clip(artifact["upper_model"].predict(X_hist), 0.0, None)
        )

        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Scatter(
                x=hist_df["date"],
                y=hist_df["high"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig_trend.add_trace(
            go.Scatter(
                x=hist_df["date"],
                y=hist_df["low"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(226, 149, 120, 0.18)",
                name="৮০% আত্মবিশ্বাসের ব্যান্ড" if is_bn else "80% Confidence Band",
            )
        )
        fig_trend.add_trace(
            go.Scatter(
                x=hist_df["date"],
                y=hist_df[TARGET_COL],
                mode="lines+markers",
                name="প্রকৃত রোগী" if is_bn else "Actual Cases",
                line=dict(color="#38BDF8", width=2),
                marker=dict(size=4),
            )
        )
        fig_trend.add_trace(
            go.Scatter(
                x=hist_df["date"],
                y=hist_df["pred"],
                mode="lines",
                name="মডেল পূর্বাভাস" if is_bn else "Model Forecast",
                line=dict(color="#E29578", width=2, dash="dash"),
            )
        )
        if is_what_if:
            fig_trend.add_trace(
                go.Scatter(
                    x=[selected_ts],
                    y=[prediction["predicted_cases"]],
                    mode="markers",
                    name="সিমুলেটেড পূর্বাভাস" if is_bn else "Simulated Forecast",
                    marker=dict(color="#EF4444", size=12, symbol="diamond"),
                )
            )
        fig_trend.update_layout(
            height=370,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis_title="সাপ্তাহিক ডেঙ্গু রোগী" if is_bn else "Weekly Dengue Cases",
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_trend)

    with col_right:
        st.subheader("ঝুঁকি বাড়ার প্রধান কারণসমূহ" if is_bn else "Key Factors Driving Risk")
        
        feature_friendly_names = {
            "rain_roll_4w_sum": "গত ৪ সপ্তাহের মোট বৃষ্টিপাত" if is_bn else "4-Week Total Rainfall",
            "rain_4w_x_waterlogging": "বৃষ্টিপাত ও জলাবদ্ধতার মিথস্ক্রিয়া" if is_bn else "Rainfall & Water-Logging",
            "humid_roll_4w_mean": "গত ৪ সপ্তাহের গড় আর্দ্রতা" if is_bn else "4-Week Mean Humidity",
            "relative_humidity_pct_lag_2_weeks": "২ সপ্তাহ আগের আর্দ্রতা" if is_bn else "Humidity Lag 2 Weeks",
            "rainfall_mm_lag_3_weeks": "৩ সপ্তাহ আগের বৃষ্টিপাত" if is_bn else "Rainfall Lag 3 Weeks",
            "avg_temp_c": "গড় তাপমাত্রা" if is_bn else "Average Temperature",
            "density_x_poor_sanitation": "জনঘনত্ব ও অপরিচ্ছন্নতা" if is_bn else "Population Density & Sanitation",
            "min_temp_c": "সর্বনিম্ন তাপমাত্রা" if is_bn else "Minimum Temperature",
            "max_temp_c": "সর্বোচ্চ তাপমাত্রা" if is_bn else "Maximum Temperature"
        }

        drivers_df = pd.DataFrame(prediction["top_drivers"]).iloc[::-1]
        drivers_df["friendly_feature"] = drivers_df["feature"].map(lambda x: feature_friendly_names.get(x, x))
        drivers_df["direction"] = np.where(
            drivers_df["impact"] >= 0, "ঝুঁকি বাড়ায়" if is_bn else "Increases Risk", "ঝুঁকি কমায়" if is_bn else "Reduces Risk"
        )
        
        fig_shap = px.bar(
            drivers_df,
            x="impact",
            y="friendly_feature",
            orientation="h",
            color="direction",
            color_discrete_map={
                "Increases Risk": "#EF4444",
                "Reduces Risk": "#22C55E",
                "ঝুঁকি বাড়ায়": "#EF4444",
                "ঝুঁকি কমায়": "#22C55E"
            },
            labels={"impact": "রোগীর সংখ্যার ওপর প্রভাব" if is_bn else "Impact on Case Count", "friendly_feature": "পরিবেশগত কারণ" if is_bn else "Environmental Factor"},
        )
        fig_shap.update_layout(
            height=370,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", y=1.12, title=None),
        )
        st.plotly_chart(fig_shap)

with tab2:
    st.subheader(f"All-Ward Risk Triage — Week of {selected_date_str}")
    week_df = feat_df[feat_df["date"] == selected_ts].copy()
    X_week = week_df[artifact["features"]]

    week_df["predicted_cases"] = np.round(
        np.clip(artifact["main_model"].predict(X_week), 0.0, None), 2
    )
    week_df["ci_low"] = np.round(
        np.clip(artifact["lower_model"].predict(X_week), 0.0, None), 2
    )
    week_df["ci_high"] = np.round(
        np.maximum(
            week_df["predicted_cases"],
            np.clip(artifact["upper_model"].predict(X_week), 0.0, None),
        ),
        2,
    )
    week_df["risk_tier"] = [
        assign_risk_tier(v, artifact["thresholds"]) for v in week_df["predicted_cases"]
    ]
    week_df = week_df.sort_values("predicted_cases", ascending=False)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Critical Wards", int((week_df["risk_tier"] == "Critical").sum()))
    k2.metric("High-Risk Wards", int((week_df["risk_tier"] == "High").sum()))
    k3.metric("Total Predicted Cases", f"{week_df['predicted_cases'].sum():.0f}")
    k4.metric("Total Recorded Cases", int(week_df[TARGET_COL].sum()))

    city_filter = st.multiselect("Filter Cities", options=cities, default=cities)
    filtered_week = week_df[week_df["city"].isin(city_filter)]

    fig_hotspots = px.bar(
        filtered_week.head(20),
        x="ward_id",
        y="predicted_cases",
        color="risk_tier",
        color_discrete_map={k: v["color"] for k, v in TIER_META.items()},
        hover_data=["city", TARGET_COL, "ci_low", "ci_high", "water_logging_risk_index"],
        labels={"predicted_cases": "Predicted Cases", "ward_id": "Ward"},
        title="Top 20 Highest-Risk Wards",
    )
    fig_hotspots.update_layout(
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_hotspots)

    display_cols = [
        "city",
        "ward_id",
        "risk_tier",
        "predicted_cases",
        "ci_low",
        "ci_high",
        TARGET_COL,
        "water_logging_risk_index",
        "sanitation_score",
    ]
    st.dataframe(filtered_week[display_cols], hide_index=True)

with tab3:
    st.subheader("Chronological Out-of-Time Validation (2024–2025)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("MAE", f"{metrics['MAE']:.3f}")
    m2.metric("RMSE", f"{metrics['RMSE']:.3f}")
    m3.metric("R² Score", f"{metrics['R2']:.4f}")
    m4.metric("Tier Accuracy", f"{metrics['tier_accuracy'] * 100:.1f}%")
    m5.metric("CPU Inference", f"{metrics['single_inference_ms']:.1f} ms")

    st.caption(
        f"Train rows: {artifact['split_info']['train_rows']:,} | "
        f"Validation rows: {artifact['split_info']['val_rows']:,} | "
        f"Cutoff date: {artifact['split_info']['cutoff_date']} | "
        f"80% CI Coverage: {metrics['ci_80_coverage'] * 100:.1f}%"
    )

    top_feat_df = pd.DataFrame(
        artifact["top_features"], columns=["Feature", "Split Importance"]
    ).iloc[::-1]
    fig_imp = px.bar(
        top_feat_df,
        x="Split Importance",
        y="Feature",
        orientation="h",
        title="Top 10 Predictive Features (Multi-Week Weather Lags & Urban Interactions)",
    )
    fig_imp.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_imp)