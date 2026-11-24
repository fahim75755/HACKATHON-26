# AlertAhead: Urban Dengue Early-Warning System

Ward-level dengue forecasting for Bangladesh city corporations. A public health officer picks a
**city, ward and date**; the app forecasts dengue cases **2, 3 and 4 weeks later** (all made on the same day),
assigns a risk tier (Low / Medium / High / Critical) to each, gives an early-warning headline and recommended
action, and shows **which weather factors drive the forecast**.

Rain does not cause cases the next day: water must stand for 7-14 days for Aedes eggs to become
mosquitoes, then the virus incubates. In this data, cases follow rain most strongly **about 3 weeks
later** (correlation 0.75 vs 0.53 for same-week rain). The system uses that delay to warn officers before the spike.

## Quickstart (under 3 minutes)

Requires **Python 3.11-3.13**. Trained models and precomputed forecasts are included, so no training is needed to run the app.

```bash
git clone https://github.com/fahim75755/HACKATHON-26.git
cd HACKATHON-26
python -m venv .venv
# Windows:            .venv\Scripts\activate
# macOS / Linux:      source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at http://localhost:8501.

### Retrain and test (optional)

```bash
python train.py                         # ~20 s on a laptop CPU; rebuilds models/ and data/processed/
python -m unittest discover -s tests -v # 25 tests: lags, leakage, split, baselines, outlook, watch flag, what-if, dashboard (English/Bangla)
```

## Using the dashboard

The sidebar only takes input; everything else is output. The whole page switches between **English and বাংলা**
(language box at the top right).

| Input (sidebar) | Output |
|---|---|
| City: **All cities** or one of 5 | **All wards** (default): one tab each for **2, 3 and 4 weeks ahead** - headline such as "⚠ 23 of 60 wards HIGH or CRITICAL in the week of 28 Jul 2025", what to do with that much warning time, how reliable that lead time was in 2025, tier counts, per-city summary, a ward risk map (city x ward number) and a ranked ward list (CSV download) |
| Ward: **All wards** or one ward | **One ward:** the same 2, 3 and 4-week tabs show only that ward (tab names show its tier, e.g. "2 weeks ahead · 28 Jul · 🟠 High") - expected cases and 80% range, cases now and trend, what to do, how reliable that lead time is, its rank among the wards, driver chart (TreeSHAP) for that week's model, and rain vs cases |
| "Forecast from week of" date | All forecasts are made from this week's data |
| What-if: extra rain in the last 1–2 days (mm) | Forecasts rerun live with the extra rain (in every ward shown) |

* **Action by lead time:** 4 weeks ahead = *plan* (order saline, kits, larvicide; schedule teams);
  3 weeks ahead = *stop the breeding* (drain water, clear containers - breaks the cycle);
  2 weeks ahead = *respond* (fogging, alert clinics, free beds).
* **Extra rain what-if:** rain that fell in the last day or two, not yet in the weekly record, is added to this
  week's rain and the forecasts rerun live. Example: DNCC_Ward_01 on 14 Jul 2025, +100 mm moves the 3-week
  forecast from 8.4 (Medium) to 16.1 (High); across all cities, High/Critical wards 3 weeks ahead go from 14 to 53 of 60.

* **Watch flag:** "⚠ Could reach High" when the expected cases are lower but the top of the 80% range reaches High
  or Critical. On 2025 data this catches 91–95% of High/Critical weeks vs 51–63% from the expected value alone.
* **Dates in the data (Jan 2016 - Dec 2025):** forecasts are looked up instantly; "Actually recorded" shows the real outcome.
* **Dates after Dec 2025:** the officer enters the last 4 weeks of weather (pre-filled with the ward's seasonal normals) and the model predicts live.
* **Accuracy tab (for reviewers):** 2025 test-year MAE / RMSE vs three baselines, and forecast vs actual for the selected ward (or the selected wards added up).

## Results on 2025 (held out: never used for training or tuning)

| Weeks ahead | LightGBM MAE | Neg. binomial GLM | Same week last year | This week's cases | LightGBM tier accuracy | 80% range coverage |
|---|---|---|---|---|---|---|
| 2 | **1.10** | 1.90 | 1.48 | 1.73 | 87.8% | 83.0% |
| 3 | **1.13** | 1.76 | 1.48 | 1.99 | 86.9% | 82.2% |
| 4 | **1.22** | 1.80 | 1.48 | 2.16 | 85.1% | 86.7% |

MAE = average error in cases per ward per week. Full metrics (RMSE, High+Critical recall/precision) are in `models/metrics.json`.

## Project structure

```
app.py                 Streamlit dashboard (Task 4)
train.py               Trains all models, writes models/ and data/processed/
dengue/config.py       Paths, horizons, split dates, feature list, model settings
dengue/data.py         Load CSV/Excel, parse dates, fail-fast validation (Task 1)
dengue/features.py     Ward-grouped weather lags, rolling sums, "unusually wet" signal (Task 1)
dengue/model.py        LightGBM per horizon + quantile range models, evaluation (Task 2)
dengue/analysis.py     Metrics and baselines: negative binomial GLM, last year, persistence (Task 2)
dengue/tiers.py        Risk tiers and recommended actions (Task 2)
dengue/engine.py       Lookups, live what-if predictions, driver breakdown (Tasks 3-4)
data/raw/              Source dataset (31,320 rows, 60 wards, weekly 2016-2025)
data/processed/        features.parquet, predictions.parquet (generated)
models/                lgbm_h2/h3/h4.joblib, metrics.json (generated)
tests/                 Unit tests
ARCHITECTURE.md        Data flow, validation split, model choice, cost controls
```

## Team

| Member | GitHub | Area |
|---|---|---|
| Fahim | fahim75755 | Risk prediction engine (Task 2) |
| Omar Faruk Naim | naim2003programming | User interface (Task 4) |

## AI assistants and external APIs (disclosure)

* **Claude (Anthropic), via Claude Code** - used to review the dataset, design and write the multi-horizon
  feature pipeline, training code, baselines, tests, dashboard changes and documentation.
* **Other AI assistants used by team members:** _team members add any other tools here (name and what it was used for)._
* **External APIs:** none. The app runs fully offline: no weather API and no LLM calls at runtime.
* **Open-source libraries:** pandas, NumPy, LightGBM, scikit-learn, statsmodels, Streamlit, Plotly, PyArrow, joblib.
