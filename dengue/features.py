"""Task 1 - weather processing: ward-grouped multi-week lags known on the issue week.

Every feature for issue week t uses only data from weeks t, t-1, t-2, t-3 (or earlier).
The forecast targets are cases at t+2, t+3 and t+4. Combined, this means rain
1-6 weeks before the predicted week drives the forecast - matching the 2-4 week
mosquito breeding + incubation delay without peeking at future weather.
"""
import numpy as np
import pandas as pd

from dengue.config import (
    FEATURES, HORIZONS, HUMID_COL, RAIN_COL, TARGET_COL, TEMP_COL, WARD_TRAITS,
)

KEY_COLS = ["date", "city", "ward_id"]
RAW_KEEP = [RAIN_COL, HUMID_COL, TEMP_COL, TARGET_COL]


def season_terms(dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    week = pd.to_datetime(dates).dt.isocalendar().week.astype(int).to_numpy()
    angle = 2 * np.pi * week / 52.18
    return np.sin(angle), np.cos(angle)


def season_week(dates: pd.Series) -> pd.Series:
    """Week-of-year used for 'normal for this time of year' (week 53 folded into 52)."""
    return pd.to_datetime(dates).dt.isocalendar().week.astype(int).clip(upper=52)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (ward, issue week) with model features and h-week-ahead targets."""
    df = raw.sort_values(["ward_id", "date"]).reset_index(drop=True).copy()
    g = df.groupby("ward_id", sort=False)

    for k in range(4):
        df[f"rain_lag_{k}w"] = g[RAIN_COL].shift(k)

    df["rain_sum_4w"] = g[RAIN_COL].transform(lambda s: s.rolling(4).sum())
    df["humid_mean_4w"] = g[HUMID_COL].transform(lambda s: s.rolling(4).mean())
    df["temp_mean_4w"] = g[TEMP_COL].transform(lambda s: s.rolling(4).mean())
    df["cases_now"] = df[TARGET_COL].astype(float)
    df["cases_mean_4w"] = g[TARGET_COL].transform(lambda s: s.rolling(4).mean())

    # "Unusually wet": this ward's 4-week rain minus its average for the same
    # week of year in PREVIOUS years only (expanding mean, shifted -> no leakage).
    df["season_week"] = season_week(df["date"])
    df["rain_normal_4w"] = df.groupby(["ward_id", "season_week"], sort=False)[
        "rain_sum_4w"
    ].transform(lambda s: s.shift(1).expanding().mean())
    df["rain_anom_4w"] = (df["rain_sum_4w"] - df["rain_normal_4w"]).fillna(0.0)

    df["week_sin"], df["week_cos"] = season_terms(df["date"])

    for h in HORIZONS:
        df[f"target_h{h}"] = g[TARGET_COL].shift(-h)
        df[f"target_date_h{h}"] = df["date"] + pd.Timedelta(weeks=h)

    # Drop the first 3 weeks of each ward (incomplete 4-week windows).
    df = df.dropna(subset=[c for c in FEATURES if c != "rain_anom_4w"])
    cols = (
        KEY_COLS + RAW_KEEP + ["season_week", "rain_normal_4w"] + FEATURES
        + [f"target_h{h}" for h in HORIZONS] + [f"target_date_h{h}" for h in HORIZONS]
    )
    out = df[cols].reset_index(drop=True)
    float_cols = [c for c in out.columns if out[c].dtype == "float64"]
    out[float_cols] = out[float_cols].astype("float32")
    return out


def ward_rain_normal(history: pd.DataFrame, ward_id: str, issue_date: pd.Timestamp) -> float | None:
    """Average 4-week rain for this ward and week of year, from years before issue_date."""
    wk = int(season_week(pd.Series([issue_date])).iloc[0])
    past = history[
        (history["ward_id"] == ward_id)
        & (history["season_week"] == wk)
        & (history["date"] < issue_date)
    ]["rain_sum_4w"]
    return float(past.mean()) if len(past) else None


def make_feature_row(
    *,
    issue_date: pd.Timestamp,
    rain_last4: list[float],
    humid_last4: list[float],
    temp_last4: list[float],
    cases_last4: list[float],
    traits: dict,
    rain_normal_4w: float | None,
) -> pd.DataFrame:
    """Build one model input row from raw weekly values.

    The *_last4 lists are ordered oldest -> newest: [t-3, t-2, t-1, t].
    Used for what-if simulation and for weeks after the dataset ends.
    Produces exactly the same values as build_features for the same inputs.
    """
    rain = [float(v) for v in rain_last4]
    rain_sum = float(sum(rain))
    sin, cos = season_terms(pd.Series([issue_date]))
    row = {
        "rain_lag_0w": rain[3],
        "rain_lag_1w": rain[2],
        "rain_lag_2w": rain[1],
        "rain_lag_3w": rain[0],
        "rain_sum_4w": rain_sum,
        "rain_anom_4w": 0.0 if rain_normal_4w is None else rain_sum - rain_normal_4w,
        "humid_mean_4w": float(np.mean(humid_last4)),
        "temp_mean_4w": float(np.mean(temp_last4)),
        "cases_now": float(cases_last4[3]),
        "cases_mean_4w": float(np.mean(cases_last4)),
        "week_sin": float(sin[0]),
        "week_cos": float(cos[0]),
        **{k: float(traits[k]) for k in WARD_TRAITS},
    }
    return pd.DataFrame([row], columns=FEATURES).astype("float32")
