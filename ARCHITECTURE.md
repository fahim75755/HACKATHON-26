# System Architecture — Urban Dengue Outbreak Early-Warning Engine

## 1. End-to-End Pipeline

    Raw Weekly CSV (31,320 rows, 60 wards, 2016–2025)
            │
            ▼
    [dengue/features.py]
      ├─ Strict ward-level sorting & grouping (prevents cross-ward leakage)
      ├─ 1–4 week lags: avg/max/min temp, rainfall, humidity
      ├─ 2w & 4w rolling windows (shifted by 1w to capture mosquito lifecycle)
      ├─ Cyclical seasonality: sin/cos of ISO week
      └─ Urban interaction terms: rain_4w × water_logging, density × (1 - sanitation)
            │
            ▼
    [dengue/model.py]  ── Chronological 80/20 Split (Cutoff: 2024-01-08)
      ├─ Main Regressor:   LightGBM L2 (point forecast + TreeSHAP pred_contrib)
      ├─ Lower Bound:      LightGBM Quantile (alpha = 0.10)
      └─ Upper Bound:      LightGBM Quantile (alpha = 0.90)
            │
            ▼
    [dengue/tiers.py]  ── 4-Tier Epidemiological Triage
      ├─ Low      (less than 2 cases/wk)
      ├─ Medium   (2 to less than 5 cases/wk)
      ├─ High     (5 to less than 10 cases/wk)
      └─ Critical (10 or more cases/wk)
            │
            ▼
    [app.py]  ── Streamlit + Plotly Interactive Dashboard
      ├─ Live ward forecasting & 52-week confidence bands
      ├─ Real-time What-If climate & infrastructure simulator
      ├─ Local TreeSHAP driver attribution (~12.6 ms/prediction)
      └─ 60-ward municipal hotspot triage radar

## 2. Anti-Leakage Design
- **Spatial Isolation:** All `.shift()` and `.rolling()` operations run inside `groupby("ward_id")` so weather from Ward A never leaks into Ward B.
- **Temporal Integrity:** Rolling windows are shifted by 1 week (`shift(1).rolling(k)`) and evaluation uses a strict forward-in-time split (2016–2023 train, 2024–2025 validation) instead of random shuffling.
- **Deterministic Encoding:** `city` and `ward_id` mappings are persisted inside the serialized `.joblib` artifact to prevent train/inference category drift during single-row What-If simulation.

## 3. Why LightGBM + Built-In TreeSHAP?
- **Tabular & Non-Linear Interactions:** Excels on mixed static urban indices and lagged meteorological variables without requiring GPU hardware.
- **Exact Fast Attribution:** Using LightGBM's native `pred_contrib=True` computes exact TreeSHAP contributions in the same unit as dengue cases in ~12.6 ms on standard CPU, avoiding heavy external SHAP overhead.
- **Lightweight Deployment:** All three models plus metadata compress into a single ~1.15 MB `.joblib` file for instant offline execution.