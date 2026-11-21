"""Shared paths and constants for the dengue early-warning system."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = ROOT / "data" / "raw" / "dengue_data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"
PREDICTIONS_PATH = PROCESSED_DIR / "predictions.parquet"

MODELS_DIR = ROOT / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"


def model_path(horizon: int) -> Path:
    return MODELS_DIR / f"lgbm_h{horizon}.joblib"


# Forecast horizons in weeks: predict cases 2, 3 and 4 weeks after the issue week.
HORIZONS = (2, 3, 4)

# Chronological split by TARGET date (the week being predicted).
TRAIN_END = "2023-12-31"   # train: target week <= 2023
VAL_YEAR = 2024            # early stopping / tuning
TEST_YEAR = 2025           # final, untouched evaluation

TARGET_COL = "dengue_cases"
RAIN_COL = "rainfall_mm"
HUMID_COL = "relative_humidity_pct"
TEMP_COL = "avg_temp_c"

REQUIRED_COLUMNS = [
    "date", "city", "ward_id",
    "population_density_sqkm", "water_logging_risk_index", "sanitation_score",
    "avg_temp_c", "max_temp_c", "min_temp_c",
    RAIN_COL, HUMID_COL, TARGET_COL,
]

WARD_TRAITS = ["population_density_sqkm", "water_logging_risk_index", "sanitation_score"]

# Model inputs. Every value is known on the issue week (no future weather).
FEATURES = [
    "rain_lag_0w",       # rain this week
    "rain_lag_1w",       # rain 1 week ago
    "rain_lag_2w",       # rain 2 weeks ago
    "rain_lag_3w",       # rain 3 weeks ago
    "rain_sum_4w",       # total rain over the last 4 weeks
    "rain_anom_4w",      # last-4-week rain minus this ward's normal for the season
    "humid_mean_4w",     # average humidity over the last 4 weeks
    "temp_mean_4w",      # average temperature over the last 4 weeks
    "cases_now",         # cases reported this week
    "cases_mean_4w",     # average weekly cases over the last 4 weeks
    "week_sin",          # season (week of year)
    "week_cos",
    *WARD_TRAITS,
]

FEATURE_LABELS = {
    "rain_lag_0w": "Rain this week",
    "rain_lag_1w": "Rain 1 week ago",
    "rain_lag_2w": "Rain 2 weeks ago",
    "rain_lag_3w": "Rain 3 weeks ago",
    "rain_sum_4w": "Rain total, last 4 weeks",
    "rain_anom_4w": "Unusually wet (vs ward normal)",
    "humid_mean_4w": "Humidity, last 4 weeks",
    "temp_mean_4w": "Temperature, last 4 weeks",
    "cases_now": "Cases this week",
    "cases_mean_4w": "Cases, last 4 weeks",
    "week_sin": "Season",
    "week_cos": "Season",
    "population_density_sqkm": "Population density",
    "water_logging_risk_index": "Water-logging risk",
    "sanitation_score": "Sanitation score",
}

LGBM_PARAMS = {
    "objective": "poisson",
    "learning_rate": 0.03,
    "n_estimators": 3000,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}
EARLY_STOPPING_ROUNDS = 100
