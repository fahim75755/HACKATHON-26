# Urban Dengue Outbreak Early-Warning Engine

A fast, ward-level dengue outbreak forecasting and decision-support system built for urban municipalities across Bangladesh. Powered by a multi-week weather-lag feature pipeline, a 3-model LightGBM quantile ensemble, local TreeSHAP explainability, and an interactive Streamlit dashboard.

## Key Capabilities
- **Ward-Level Granularity:** Covers 60 wards across 5 major cities (Chattogram, Dhaka North, Dhaka South, Gazipur, Narayanganj) using 10 years of weekly historical data (31,320 rows, 2016–2025).
- **Zero-Leakage Multi-Week Lag Pipeline:** Strictly ward-grouped 1–4 week weather lags, 2- and 4-week rolling accumulations, monsoon seasonality (`sin`/`cos`), and urban vulnerability interactions (rainfall × water-logging, population density × poor sanitation).
- **Calibrated Uncertainty & Actionable Tiers:** Predicts weekly cases alongside an **80% confidence interval** (10th–90th quantile LightGBM models) and maps risk into 4 tiers (`Low`, `Medium`, `High`, `Critical`) with municipal response playbooks.
- **Local Explainability & What-If Simulator:** Built-in TreeSHAP (`pred_contrib=True`) explains top risk drivers for every prediction in ~12.6 ms on CPU, paired with live climate and infrastructure shock sliders.

## Out-of-Time Validation Metrics (2024–2025)
Evaluated on a strict chronological 80/20 split (`cutoff_date: 2024-01-08` — 24,840 training rows, 6,240 unseen validation rows):

| Metric | Validation Score |
| :--- | :--- |
| **MAE** | `0.846` weekly cases |
| **RMSE** | `1.594` weekly cases |
| **R² Score** | `0.8306` |
| **4-Tier Risk Accuracy** | `83.89%` |
| **80% Confidence Interval Coverage** | `78.48%` |
| **Full Training Time (CPU)** | `10.19 s` |
| **Single Prediction + TreeSHAP Latency** | `12.62 ms` |

## Quickstart

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt