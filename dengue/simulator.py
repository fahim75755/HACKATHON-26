"""R3 - "What-if" vector-control simulator: model-driven counterfactual + intervention ROI.

A larvicide/fogging crew removes breeding sites, which in the trained models is the standing-water
signal: recent rain (this week and 1-3 weeks back), the 4-week rain total and the ward's
water-logging index. n crews in a ward scale those features by (1 - e)^n (e = efficacy,
diminishing returns), the 4-week anomaly is recomputed against the ward's seasonal normal, and the
SAME h2/h3/h4 LightGBM models are re-run. Baseline and counterfactual come from one live
prediction call, so 0 crews always gives exactly 0 prevented (no float32/float64 drift).

Output feeds R1 crew assignment (marginal cases prevented per extra crew) and the R3 charts.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

import numpy as np
import pandas as pd

from dengue.config import FEATURES, FEATURES_PATH, HORIZONS

BREEDING_FEATURES = ["rain_lag_0w", "rain_lag_1w", "rain_lag_2w", "rain_lag_3w", "rain_sum_4w", "water_logging_risk_index"]
MAX_CREWS_PER_WARD = 6


@lru_cache(maxsize=1)
def _models() -> dict:
    from dengue.model import load_artifacts
    return load_artifacts()


@lru_cache(maxsize=1)
def _features() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PATH)


def counterfactual_rows(rows: pd.DataFrame, n_crews: int, efficacy: float) -> pd.DataFrame:
    """Feature rows with breeding-water signals cut by (1 - e)^n; anomaly recomputed consistently."""
    factor = (1.0 - float(efficacy)) ** int(n_crews)
    out = rows.copy()
    for c in BREEDING_FEATURES:
        out[c] = out[c].astype("float64") * factor
    if "rain_normal_4w" in out.columns:
        normal = out["rain_normal_4w"].astype("float64")
        out["rain_anom_4w"] = (out["rain_sum_4w"] - normal).where(normal.notna(), 0.0)
    return out


def run_simulator(wards: pd.DataFrame, params: Any, rates: Dict[str, float] | None = None,
                  features: pd.DataFrame | None = None, models: dict | None = None,
                  max_crews: int = MAX_CREWS_PER_WARD) -> Dict[str, Any]:
    """Cases prevented at +2/+3/+4 weeks for 0..max_crews crews in every ward.

    Returns {"cum": {ward: array[(max_crews+1) x 3]}, "marginal_roi": {ward: [m1..mK]}, ...}.
    marginal_roi[k-1] = extra cases prevented (summed over +2..+4 weeks) by the k-th crew.
    """
    feats = features if features is not None else _features()
    mdl = models if models is not None else _models()
    issue = pd.Timestamp(getattr(params, "issue_date"))
    e = float(getattr(params, "efficacy", 0.30))
    ids = [str(w) for w in wards["ward_id"]]
    rows = feats[(feats["date"] == issue) & feats["ward_id"].isin(ids)].drop_duplicates("ward_id").set_index("ward_id")
    ids = [w for w in ids if w in rows.index]
    if not ids:
        return {}
    rows = rows.loc[ids]
    k = int(max_crews)
    stack = pd.concat([counterfactual_rows(rows, n, e) for n in range(k + 1)], ignore_index=True)
    preds = {h: np.clip(mdl[h]["main"].predict(stack[FEATURES]), 0, None).reshape(k + 1, len(ids)) for h in HORIZONS}

    cum, marginal, non_monotone = {}, {}, 0
    for i, wid in enumerate(ids):
        arr = np.zeros((k + 1, len(HORIZONS)))
        for j, h in enumerate(HORIZONS):
            col = preds[h][:, i]
            non_monotone += int((col[1:] > col[0] + 1e-9).sum())
            prevented = np.maximum(0.0, col[0] - col)            # clip >= 0
            prevented = np.maximum.accumulate(prevented)          # an extra crew never un-prevents cases
            prevented[0] = 0.0                                    # 0 crews -> exactly 0
            arr[:, j] = prevented
        cum[wid] = arr
        tot = arr.sum(axis=1)
        marginal[wid] = [float(tot[n] - tot[n - 1]) for n in range(1, k + 1)]
    return {"cum": cum, "marginal_roi": marginal, "efficacy": e, "max_crews": k, "horizons": list(HORIZONS),
            "non_monotone_points": non_monotone, "source": "live LightGBM h2/h3/h4 counterfactual"}


def attach_simulator_columns(wards: pd.DataFrame, sim_output: Dict[str, Any], rates: Dict[str, float]) -> pd.DataFrame:
    """Prevented cases / beds / saline for the crews actually assigned, plus ROI per crew."""
    w = wards.copy()
    cum = (sim_output or {}).get("cum", {})
    marg = (sim_output or {}).get("marginal_roi", {})
    k = int((sim_output or {}).get("max_crews", MAX_CREWS_PER_WARD))
    crews = w["crews_assigned"].astype(int).to_numpy() if "crews_assigned" in w.columns else np.zeros(len(w), dtype=int)
    per_h = np.zeros((len(w), len(HORIZONS)))
    first = np.zeros(len(w))
    for i, wid in enumerate(w["ward_id"].astype(str)):
        arr = cum.get(wid)
        if arr is None:
            continue
        per_h[i] = arr[min(crews[i], k)]
        first[i] = marg.get(wid, [0.0])[0]
    for j, h in enumerate(HORIZONS):
        w[f"prevented_h{h}"] = np.round(per_h[:, j], 3)
    total = per_h.sum(axis=1)
    w["cases_prevented"] = np.round(total, 3)
    w["beds_prevented"] = np.round(total * rates["beds"], 3)
    w["saline_prevented"] = np.round(total * rates["saline"], 3)
    w["roi_first_crew"] = np.round(first, 3)
    w["roi_per_crew"] = np.round(np.where(crews > 0, total / np.maximum(crews, 1), first), 3)
    return w
