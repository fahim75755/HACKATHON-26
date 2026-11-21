import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dengue.features import TARGET_COL, build_features, load_raw_data
from dengue.model import DEFAULT_MODEL_PATH, load_model, predict_row
from dengue.tiers import TIER_META, assign_risk_tier

st.set_page_config(
    page_title="Urban Dengue Early-Warning System",
    page_icon="🦟",
    layout="wide",
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

# --- Header ---
st.title("🦟 Urban Dengue Outbreak Early-Warning Engine")
st.caption(
    "Ward-level LightGBM forecasting with 1–4 week weather lags, 80% quantile confidence intervals, and local TreeSHAP explainability."
)

# --- Sidebar Controls ---
st.sidebar.header("Forecast Controls")
cities = sorted(feat_df["city"].unique())
selected_city = st.sidebar.selectbox("City", cities)

city_wards = sorted(feat_df.loc[feat_df["city"] == selected_city, "ward_id"].unique())
selected_ward = st.sidebar.selectbox("Ward ID", city_wards)

available_dates = sorted(feat_df["date"].dt.strftime("%Y-%m-%d").unique())
selected_date_str = st.sidebar.selectbox(
    "Forecast Week", available_dates, index=len(available_dates) - 1
)
selected_ts = pd.Timestamp(selected_date_str)

# Baseline static attributes for selected ward
ward_static = raw_df[raw_df["ward_id"] == selected_ward].iloc[-1]
base_wl = float(ward_static["water_logging_risk_index"])
base_san = float(ward_static["sanitation_score"])
base_pop = int(ward_static["population_density_sqkm"])

st.sidebar.divider()
st.sidebar.subheader("⚡ What-If Climate & Urban Simulator")
st.sidebar.caption("Adjust recent 4-week weather or ward infrastructure to simulate shocks.")

rain_pct = st.sidebar.slider("Rainfall Change (%)", -50, 150, 0, step=10)
temp_delta = st.sidebar.slider("Temperature Shift (°C)", -3.0, 5.0, 0.0, step=0.5)
humid_delta = st.sidebar.slider("Humidity Shift (%)", -20, 20, 0, step=2)
wl_override = st.sidebar.slider(
    "Water-Logging Risk Index",
    0.0,
    1.0,
    round(base_wl, 2),
    step=0.05,
    key=f"wl_{selected_ward}",
)
san_override = st.sidebar.slider(
    "Sanitation Score",
    0.0,
    1.0,
    round(base_san, 2),
    step=0.05,
    key=f"san_{selected_ward}",
)

is_what_if = (
    rain_pct != 0
    or temp_delta != 0.0
    or humid_delta != 0
    or abs(wl_override - base_wl) > 1e-6
    or abs(san_override - base_san) > 1e-6
)

# Build target feature row (supports live What-If recomputation with zero drift)
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
        "📍 Ward Forecast & Drivers",
        "🗺️ 60-Ward Hotspot Radar",
        "📊 Model Performance",
    ]
)

with tab1:
    if is_what_if:
        st.info(
            f"What-If Simulation Active — Rainfall {rain_pct:+d}%, Temp {temp_delta:+.1f}°C, "
            f"Humidity {humid_delta:+d}%, Water-Logging {wl_override:.2f}, Sanitation {san_override:.2f}"
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Weekly Cases", f"{prediction['predicted_cases']:.1f}")
    c2.metric(
        "80% Confidence Range",
        f"{prediction['ci_low']:.1f} – {prediction['ci_high']:.1f}",
    )
    c3.metric("Risk Tier", prediction["risk_tier"])
    c4.metric(
        "Recorded Actual" if not is_what_if else "Baseline Actual",
        f"{actual_cases} cases",
    )

    st.markdown(
        f"""
        <div style="padding:14px 18px;border-radius:8px;border-left:6px solid {prediction['tier_color']};background-color:rgba(128,128,128,0.12);margin:10px 0 18px 0;">
            <strong>{selected_ward} ({selected_city}) — {prediction['risk_tier'].upper()} RISK</strong><br/>
            <span>Recommended Action: {prediction['recommended_action']}</span><br/>
            <small>Population Density: {base_pop:,}/sqkm | Water-Logging: {wl_override:.2f} | Sanitation: {san_override:.2f}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1.15, 1.0])

    with col_left:
        st.subheader("52-Week Outbreak Trajectory")
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
                fillcolor="rgba(239, 108, 0, 0.18)",
                name="80% Confidence Band",
            )
        )
        fig_trend.add_trace(
            go.Scatter(
                x=hist_df["date"],
                y=hist_df[TARGET_COL],
                mode="lines+markers",
                name="Actual Cases",
                line=dict(color="#1E88E5", width=2),
                marker=dict(size=4),
            )
        )
        fig_trend.add_trace(
            go.Scatter(
                x=hist_df["date"],
                y=hist_df["pred"],
                mode="lines",
                name="Model Forecast",
                line=dict(color="#EF6C00", width=2, dash="dash"),
            )
        )
        if is_what_if:
            fig_trend.add_trace(
                go.Scatter(
                    x=[selected_ts],
                    y=[prediction["predicted_cases"]],
                    mode="markers",
                    name="Simulated Forecast",
                    marker=dict(color="#C62828", size=12, symbol="diamond"),
                )
            )
        fig_trend.update_layout(
            height=370,
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis_title="Weekly Dengue Cases",
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_trend)

    with col_right:
        st.subheader("Top Risk Drivers (TreeSHAP)")
        drivers_df = pd.DataFrame(prediction["top_drivers"]).iloc[::-1]
        drivers_df["direction"] = np.where(
            drivers_df["impact"] >= 0, "Increases Risk", "Reduces Risk"
        )
        fig_shap = px.bar(
            drivers_df,
            x="impact",
            y="feature",
            orientation="h",
            color="direction",
            color_discrete_map={
                "Increases Risk": "#C62828",
                "Reduces Risk": "#2E7D32",
            },
            labels={"impact": "Case Impact vs Baseline", "feature": "Feature"},
        )
        fig_shap.update_layout(
            height=370,
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
    fig_hotspots.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
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
    fig_imp.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_imp)