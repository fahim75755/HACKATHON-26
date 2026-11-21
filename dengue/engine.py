"""Serving layer used by the dashboard: lookups, live what-if predictions, drivers.

Cost design: models and tables are loaded once (cached by the app). A normal
forecast is a table lookup in precomputed predictions - no model call. The model
runs live only when the officer changes what-if inputs or enters new weather.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dengue.config import FEATURE_LABELS, FEATURES, HUMID_COL, RAIN_COL, TARGET_COL, TEMP_COL, WARD_TRAITS
from dengue.features import make_feature_row, season_week, ward_rain_normal
from dengue.tiers import assign_risk_tier, get_tier_info


@dataclass
class Forecast:
    issue_date: pd.Timestamp
    target_date: pd.Timestamp
    horizon: int
    cases: float
    low: float
    high: float
    tier: str
    color: str
    action: str
    source: str            # "precomputed" or "live"
    actual: float | None = None


def _make(issue_date, h, cases, low, high, source, actual=None) -> Forecast:
    tier = assign_risk_tier(cases)
    meta = get_tier_info(tier)
    return Forecast(
        issue_date=pd.Timestamp(issue_date),
        target_date=pd.Timestamp(issue_date) + pd.Timedelta(weeks=h),
        horizon=h, cases=float(cases), low=float(low), high=float(max(high, cases)),
        tier=tier, color=meta["color"], action=meta["action"], source=source,
        actual=None if actual is None or pd.isna(actual) else float(actual),
    )


def snap_to_week(features: pd.DataFrame, ward_id: str, chosen: pd.Timestamp) -> pd.Timestamp | None:
    """Latest data week for this ward on or before the chosen date (None if outside data)."""
    dates = features.loc[features["ward_id"] == ward_id, "date"]
    if dates.empty or chosen < dates.min() or chosen > dates.max() + pd.Timedelta(days=6):
        return None
    return dates[dates <= chosen].max()


def lookup(preds: pd.DataFrame, features: pd.DataFrame, ward_id: str, issue_date, h: int) -> Forecast:
    row = preds[(preds["ward_id"] == ward_id) & (preds["date"] == issue_date)].iloc[0]
    frow = features[(features["ward_id"] == ward_id) & (features["date"] == issue_date)].iloc[0]
    return _make(issue_date, h, row[f"pred_h{h}"], row[f"low_h{h}"], row[f"high_h{h}"],
                 "precomputed", frow[f"target_h{h}"])


def predict_live(artifact: dict, X: pd.DataFrame, issue_date, actual=None) -> Forecast:
    X = X[FEATURES]
    cases = max(0.0, float(artifact["main"].predict(X)[0]))
    low = max(0.0, float(artifact["low"].predict(X)[0]))
    high = max(0.0, float(artifact["high"].predict(X)[0]))
    return _make(issue_date, artifact["horizon"], cases, low, high, "live", actual)


def drivers(artifact: dict, X: pd.DataFrame, top_k: int = 7) -> tuple[pd.DataFrame, float]:
    """Per-feature effect on the forecast using LightGBM's built-in TreeSHAP.

    The Poisson model works on a log scale: log(forecast) = base + sum(contributions).
    We convert to case units by scaling every log contribution by the logarithmic
    mean of (forecast, baseline), so the effects add up exactly to
    forecast - baseline. Returns (table, baseline_cases).
    """
    contrib = artifact["main"].predict(X[FEATURES], pred_contrib=True)[0]
    base = float(np.exp(contrib[-1]))
    pred = float(np.exp(contrib.sum()))
    log_ratio = np.log(pred / base)
    scale = (pred - base) / log_ratio if abs(log_ratio) > 1e-9 else pred
    by_label: dict[str, float] = {}
    for f, c in zip(FEATURES, contrib[:-1]):
        label = FEATURE_LABELS.get(f, f)
        by_label[label] = by_label.get(label, 0.0) + float(c)
    df = pd.DataFrame({"driver": list(by_label), "log_effect": list(by_label.values())})
    df["effect_cases"] = df["log_effect"] * scale
    df = df.reindex(df["log_effect"].abs().sort_values(ascending=False).index).head(top_k)
    return df.reset_index(drop=True), base


def ward_traits(features: pd.DataFrame, ward_id: str) -> dict:
    row = features[features["ward_id"] == ward_id].iloc[-1]
    return {k: float(row[k]) for k in WARD_TRAITS} | {"city": row["city"]}


def recent_weeks(features: pd.DataFrame, ward_id: str, issue_date) -> pd.DataFrame:
    """The 4 recorded weeks t-3..t (oldest first) for this ward."""
    w = features[(features["ward_id"] == ward_id) & (features["date"] <= issue_date)]
    return w.sort_values("date").tail(4)


def seasonal_normals(features: pd.DataFrame, ward_id: str, issue_date) -> pd.DataFrame:
    """Typical weather/cases for the 4 weeks ending at issue_date (all past years).

    Used to pre-fill the manual-entry form for dates after the dataset ends.
    """
    issue_date = pd.Timestamp(issue_date)
    weeks = [issue_date - pd.Timedelta(weeks=k) for k in (3, 2, 1, 0)]
    sw = season_week(pd.Series(weeks)).tolist()
    w = features[features["ward_id"] == ward_id]
    rows = []
    for d, s in zip(weeks, sw):
        m = w[w["season_week"] == s][[RAIN_COL, HUMID_COL, TEMP_COL, TARGET_COL]].mean()
        rows.append({"date": d, **m.round(1).to_dict()})
    return pd.DataFrame(rows)


def build_row(
    features: pd.DataFrame,
    ward_id: str,
    issue_date,
    weeks: pd.DataFrame,
    traits: dict,
) -> pd.DataFrame:
    """Turn 4 weekly rows (rain, humidity, temp, cases; oldest first) into a model input."""
    issue_date = pd.Timestamp(issue_date)
    return make_feature_row(
        issue_date=issue_date,
        rain_last4=weeks[RAIN_COL].tolist(),
        humid_last4=weeks[HUMID_COL].tolist(),
        temp_last4=weeks[TEMP_COL].tolist(),
        cases_last4=weeks[TARGET_COL].tolist(),
        traits=traits,
        rain_normal_4w=ward_rain_normal(features, ward_id, issue_date),
    )


def apply_what_if(weeks: pd.DataFrame, rain_pct: float, humid_delta: float, temp_delta: float) -> pd.DataFrame:
    w = weeks.copy()
    w[RAIN_COL] = np.clip(w[RAIN_COL] * (1 + rain_pct / 100.0), 0, None)
    w[HUMID_COL] = np.clip(w[HUMID_COL] + humid_delta, 0, 100)
    w[TEMP_COL] = w[TEMP_COL] + temp_delta
    return w
