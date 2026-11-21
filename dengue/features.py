import numpy as np
import pandas as pd

STATIC_WARD_COLS = [
    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score",
]
TARGET_COL = "dengue_cases"
LAG_WEEKS = (1, 2, 3, 4)


def detect_weather_columns(df: pd.DataFrame) -> dict:
    """Auto-detect exact rainfall and humidity column names from the dataset."""
    rain_cols = [c for c in df.columns if "rain" in c.lower()]
    humid_cols = [c for c in df.columns if "humid" in c.lower()]
    if not rain_cols or not humid_cols:
        raise ValueError(f"Could not detect rainfall/humidity columns in: {list(df.columns)}")
    return {
        "avg_temp": "avg_temp_c",
        "max_temp": "max_temp_c",
        "min_temp": "min_temp_c",
        "rainfall": rain_cols[0],
        "humidity": humid_cols[0],
    }


def load_raw_data(csv_path: str = "data/raw/dengue_data.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ward_id", "date"]).reset_index(drop=True)


def build_features(
    df: pd.DataFrame,
    category_maps: dict | None = None,
    drop_na_lags: bool = True,
) -> tuple[pd.DataFrame, list[str], dict, dict]:
    """
    Build ward-grouped 1-4 week weather lag features, rolling accumulations,
    seasonality terms, and urban vulnerability interactions without leakage.
    """
    weather_cols = detect_weather_columns(df)
    rain_col = weather_cols["rainfall"]
    humid_col = weather_cols["humidity"]
    temp_col = weather_cols["avg_temp"]

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["ward_id", "date"]).reset_index(drop=True)

    # Derived current-week weather indicators
    data["temp_range_c"] = data["max_temp_c"] - data["min_temp_c"]

    # Cyclical seasonality (captures Bangladesh monsoon cycle)
    iso_week = data["date"].dt.isocalendar().week.astype(int)
    data["week_sin"] = np.sin(2 * np.pi * iso_week / 52.0)
    data["week_cos"] = np.cos(2 * np.pi * iso_week / 52.0)

    # Multi-week weather lags strictly grouped by ward_id (prevents cross-ward leakage)
    grouped = data.groupby("ward_id", sort=False)
    lag_source_cols = [temp_col, "max_temp_c", "min_temp_c", rain_col, humid_col]

    for col in lag_source_cols:
        for w in LAG_WEEKS:
            data[f"{col}_lag_{w}w"] = grouped[col].shift(w)

    # Rolling multi-week weather windows (mosquito breeding + incubation window: 2-4 weeks)
    data["rain_roll_2w_sum"] = grouped[rain_col].transform(
        lambda s: s.shift(1).rolling(2).sum()
    )
    data["rain_roll_4w_sum"] = grouped[rain_col].transform(
        lambda s: s.shift(1).rolling(4).sum()
    )
    data["temp_roll_4w_mean"] = grouped[temp_col].transform(
        lambda s: s.shift(1).rolling(4).mean()
    )
    data["humid_roll_4w_mean"] = grouped[humid_col].transform(
        lambda s: s.shift(1).rolling(4).mean()
    )

    # Domain interaction features (standing water + optimal Aedes aegypti climate)
    data["rain_4w_x_waterlogging"] = (
        data["rain_roll_4w_sum"] * data["water_logging_risk_index"]
    )
    data["rain_lag2_x_waterlogging"] = (
        data[f"{rain_col}_lag_2w"] * data["water_logging_risk_index"]
    )
    data["temp_humid_lag2_index"] = (
        data[f"{temp_col}_lag_2w"] * data[f"{humid_col}_lag_2w"] / 100.0
    )
    data["density_x_poor_sanitation"] = (
        data["population_density_sqkm"] * (1.0 - data["sanitation_score"])
    )

    # Deterministic integer encoding for city & ward_id (safe for single-row inference)
    if category_maps is None:
        category_maps = {
            "city": {v: i for i, v in enumerate(sorted(data["city"].unique()))},
            "ward_id": {v: i for i, v in enumerate(sorted(data["ward_id"].unique()))},
        }
    data["city_code"] = data["city"].map(category_maps["city"]).fillna(-1).astype(int)
    data["ward_code"] = data["ward_id"].map(category_maps["ward_id"]).fillna(-1).astype(int)

    exclude = {"date", "city", "ward_id", TARGET_COL}
    feature_cols = [c for c in data.columns if c not in exclude]

    if drop_na_lags:
        data = data.dropna(subset=feature_cols).reset_index(drop=True)

    return data, feature_cols, category_maps, weather_cols