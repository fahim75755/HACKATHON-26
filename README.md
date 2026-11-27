# AlertAhead: Urban Dengue Early-Warning System

Ward-level dengue forecasting for Bangladesh city corporations. A public health officer picks a
**city, ward and date**; the app forecasts dengue cases **2, 3 and 4 weeks later** (all made on the same day),
assigns a risk tier (Low / Medium / High / Critical) to each, gives an early-warning headline and recommended
action, and shows **which weather factors drive the forecast**. It then helps decide **where to send a limited
number of fogging / larvicide teams** this week.

**Live demo:** https://edu-hackathon-26-alert-ahead-dengue-coming.streamlit.app/

Rain does not cause cases the next day: water must stand for 7-14 days for Aedes eggs to become
mosquitoes, then the virus incubates. In this data, cases follow rain most strongly **about 3 weeks
later** (correlation 0.75 vs 0.53 for same-week rain). The system uses that delay to warn officers before the spike.

## Quickstart (under 3 minutes)

Requires **Python 3.11-3.13** (tested on 3.12 locally and 3.13 on Streamlit Community Cloud). Trained models and precomputed forecasts are included, so no training is needed to run the app.

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
python -m unittest discover -s tests -v # 95 tests: Phase 1 (lags, leakage, split, baselines, what-if, dashboard) + Phase 2 (allocation, routing, simulator, notices, officer portal, offline)
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

### Field operations (in every 2, 3 and 4-week tab, all-wards view)

* **Ward risk map:** all 60 wards on a street, light or satellite map, coloured by tier and sized by expected cases.
  Ward positions are approximate reference points inside each city (the dataset has no coordinates).
* **Smart team dispatch:** set how many fogging / survey teams are available; AlertAhead picks the wards that cover
  the most expected cases (or, in *vulnerability* mode, weights them by population density and water-logging).
  Officers can **pin** or **exclude** wards, every plan goes to an audit log, and a Bangla / English **SMS alert**
  text is generated for copying.
* **Live weather check (optional button):** fetches the next 7 days of rain, temperature and humidity for a city
  from Open-Meteo and warns if values fall outside the range the model was trained on. It is shown for context only:
  it does **not** change the forecasts. If the internet is unavailable, offline backup values are shown instead.
* **Quick field report:** officers type what they saw (Bangla or English); notes mentioning larvae, standing water,
  blocked drains, tyres and similar are flagged as a *Citizen Signal* on the map. Notes never change the ML forecast
  and are kept only for the current browser session.
* **Dispatch replay (Accuracy tab):** for every 2025 week, compares sending teams to AlertAhead's top wards vs
  splitting them evenly vs perfect hindsight. With 10 teams across 60 wards, AlertAhead's picks covered
  **22-23%** of the real 2025 cases vs **16.7%** for an even split; perfect hindsight would reach **30.3%**.

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
dengue/ops.py          Ward map positions, team allocator, 2025 dispatch backtest, live weather, field-report flags, SMS text
dengue/ui_extras.py    Dashboard sections for the risk map, team dispatch, live weather, field reports and dispatch replay
data/raw/              Source dataset (31,320 rows, 60 wards, weekly 2016-2025)
data/processed/        features.parquet, predictions.parquet (generated)
models/                lgbm_h2/h3/h4.joblib, metrics.json (generated)
tests/                 Unit tests
packages.txt           System library for LightGBM on Streamlit Community Cloud (libgomp1)
.devcontainer/         One-click GitHub Codespaces setup
.streamlit/config.toml Theme and toolbar settings
ARCHITECTURE.md        Data flow, validation split, model choice, cost controls
```

## Phase 2 (onsite): 🚑 Command Center

Open it from the sidebar: **View → 🚑 Command Center**, right below "Forecast from week of" (it uses the same city
and week). It turns the forecasts into an operations plan for ward commissioners and health-centre managers
across all 60 wards. Everything runs locally (no internet, no LLM).

| Requirement | In the app | Code |
|---|---|---|
| **R1 Constrained allocation** | Sliders: available regional stock 30–100 %, safety buffer, per-resource override, total spraying crews, crew efficacy. Saline / kits / beds per ward with fill bars, shortfall · red alert · CODE BLACK flags, fill-by-tier chart, +2/+3/+4-week procurement table | `dengue/resources.py` |
| **R2 Overflow routing** | Ward bed load vs capacity (85 % surge / 100 % overflow bands), intra-city transfers (integer MILP, shortest distance), city bed load, "city pool exhausted" alerts, bed-capacity slider | `dengue/routing.py` |
| **R3 Vector-control simulator** | Crews cut breeding-water features; the same h2/h3/h4 LightGBM models re-run → cases, beds and saline prevented at +2/+3/+4 weeks, ROI (cases prevented per crew) ranking; crews auto-assigned by ROI × density or placed by the officer on selected wards | `dengue/simulator.py` |
| **R4 Dispatch notices** | SMS (English 1 × GSM-7 ≤ 160, Bangla ≤ 3 parts), WhatsApp, printable brief (Bangla font embedded), city summary, civil-surgeon escalation, ZIP of everything | `dengue/notify.py`, `make_exports.py`, [`exports/`](exports/) |
| **R5 Offline** | Sidebar **Offline mode (kill-switch)** or `ALERTAHEAD_OFFLINE=1`: weather uses the offline cache, maps drop tiles; forecast ladder parquet → live LightGBM → same week last year | `dengue/fallback.py`, `tests/test_offline.py` |
| **⑥ Officer portal** (extra) | **Login** (user name + password) → **alert SMS queue**: every ward with an alert (⚫ CODE BLACK, 🔴 red, 🟠 shortfall) → SMS to **that ward's medical centre**, plus ⚫ escalation SMS to the **civil surgeon** of a city with a CODE BLACK ward or no beds left, most urgent first. A ward officer sees their ward, a city health officer their city, an admin everything. **Open in SMS app** opens the phone's own SMS app with the centre's number and text filled in; the officer sends it and marks it sent / centre confirmed. **Admin panel:** add / edit / disable officers, reset passwords, save medical-centre phone numbers, SMS log, audit log — in a small local SQLite database. The app sends nothing itself, so it works offline | `dengue/ui_outbox.py`, `dengue/db.py`, `dengue/outbox.py`, `dengue/access.py`, [`exports/outbox/`](exports/outbox/) |

* **Exact supply math:** the dataset's resource columns are a fixed rate × same-week cases. Fitted on 2016–2023
  (saline 2.384, kits 1.158, beds 0.2955 per case), the rates reproduce 2025 exactly (3120/3120, 3119/3120, 3120/3120 rows).
* **Demo scenario:** forecast from **30 Jun 2025**, **+3 weeks**, stock 60 %, 8 crews → 27 red-alert wards, 3 intra-city
  transfers, Dhaka North's bed pool exhausted (1 CODE BLACK ward); High wards get 79 % of their saline vs 35 % for
  Medium, and 8 crews prevent ≈ 163 cases over +2–4 weeks (crew efficacy 30 % is an explicit assumption).
* **Exports:** `python make_exports.py` rewrites [`exports/`](exports/) offline (byte-identical on re-run) and prints sample briefs to PDF.
* **Officer login (portal tab):** with no `ADMIN_PASSWORD` set the portal runs in **demo mode** and lists these public
  demo accounts on the login screen, so judges are never locked out:

  | User name | Password | Role |
  |---|---|---|
  | `admin` | `admin-demo-2026` | Control room (admin panel) |
  | `dhaka.north` | `city-demo-2026` | City health officer, Dhaka North |
  | `dncc10` | `ward-demo-2026` | Ward officer, DNCC_Ward_10 |

  For a private setup add `ADMIN_PASSWORD = "<strong password>"` (and optionally `ADMIN_USER`) to
  `.streamlit/secrets.toml` (git-ignored) or Streamlit Cloud → App settings → Secrets before the first start: only
  that admin is created, and the admin adds every other officer in the admin panel. Accounts, phone numbers and
  SMS status live in `data/alertahead.db` (SQLite, git-ignored). On Streamlit Cloud that file resets when the app restarts.
* **Model check (accuracy first):** 12 models compared (LightGBM variants, XGBoost, CatBoost, HistGradientBoosting,
  random forest, extra trees, Poisson GLM, ensembles), chosen on 2024 only and reported on 2025: none beat the
  Phase-1 LightGBM by more than 2 % MAE, so the models are unchanged (details in ARCHITECTURE.md).

```bash
python -m unittest discover -s tests -v   # 95 tests incl. allocation, routing, simulator, notices, officer portal (login, SMS queue, admin), offline kill-switch
python make_exports.py                    # regenerate exports/ + sample PDFs (needs no internet)
```

## Team

| Member | GitHub | Area |
|---|---|---|
| Fahim | fahim75755 | Risk prediction engine (Task 2) |
| Omar Faruk Naim | naim2003programming | User interface (Task 4), field operations: map, team dispatch, backtest; Phase 2 backend (contract, allocation, routing) |
| Ittehad Bin Hossain | OkIttehad | Phase 2 Command Center frontend, dispatch notices and exports, offline kill-switch, simulator integration |

## AI assistants and external APIs (disclosure)

* **Claude (Anthropic), via Claude Code** - used to review the dataset, design and write the multi-horizon
  feature pipeline, training code, baselines, tests, dashboard changes and documentation.
* **Claude (Anthropic), via Claude Code - Phase 2:** checked the onsite dataset (resource rates, bed capacity),
  ran the model comparison, and wrote the Command Center UI, notice templates, exports script, simulator,
  offline tests and these docs, which the team reviewed.
* **Other AI assistants used by team members:** _team members add any other tools here (name and what it was used for)._
* **External APIs (all free, no API key, nothing sent about patients):**
  * **Open-Meteo** weather forecast - only when an officer clicks "Fetch Live 7-Day Weather"; cached 30 minutes
    per city, 3.5 s timeout, offline backup values if it fails. Display only; forecasts never depend on it.
  * **Map tiles** from OpenStreetMap, CARTO and Esri World Imagery - loaded by the viewer's browser for the ward map.
  * **No LLM calls** at runtime. Forecasts, tiers, drivers and dispatch all run locally from the saved models.
* **Open-source libraries:** pandas, NumPy, LightGBM, scikit-learn, SciPy, statsmodels, Streamlit, Plotly, PyArrow, joblib.
* **Font:** Noto Sans Bengali (SIL Open Font License, `assets/fonts/OFL.txt`), subset to 45 KB for the printable briefs.
