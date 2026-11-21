"""Task 2 - risk prediction engine: one LightGBM model per forecast horizon.

For each horizon h in (2, 3, 4) weeks:
  * main model   - LightGBM, Poisson objective (count data with many zeros)
  * low / high   - LightGBM quantile models (10th / 90th percentile) -> 80% range
Split is chronological by the TARGET week: train <= 2023, early-stop on 2024,
report on 2025 (never seen during training or tuning).
"""
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from dengue.analysis import (
    NegBinBaseline, evaluate, last_year_baseline, persistence_baseline,
)
from dengue.config import (
    EARLY_STOPPING_ROUNDS, FEATURES, FEATURES_PATH, HORIZONS, LGBM_PARAMS,
    METRICS_PATH, MODELS_DIR, PREDICTIONS_PATH, PROCESSED_DIR, RAW_DATA_PATH,
    TEST_YEAR, TRAIN_END, VAL_YEAR, model_path,
)
from dengue.data import load_raw_data
from dengue.features import build_features
from dengue.tiers import DEFAULT_THRESHOLDS


def split_by_target_date(feat: pd.DataFrame, h: int) -> dict[str, pd.DataFrame]:
    rows = feat.dropna(subset=[f"target_h{h}"])
    tdate = rows[f"target_date_h{h}"]
    return {
        "train": rows[tdate <= pd.Timestamp(TRAIN_END)],
        "val": rows[tdate.dt.year == VAL_YEAR],
        "test": rows[tdate.dt.year == TEST_YEAR],
    }


def _fit_lgbm(params: dict, train: pd.DataFrame, val: pd.DataFrame, target: str):
    model = lgb.LGBMRegressor(**params)
    model.fit(
        train[FEATURES], train[target],
        eval_X=(val[FEATURES],),
        eval_y=(val[target],),
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


def train_horizon(feat: pd.DataFrame, h: int) -> tuple[dict, dict]:
    target = f"target_h{h}"
    parts = split_by_target_date(feat, h)
    train, val, test = parts["train"], parts["val"], parts["test"]

    t0 = time.perf_counter()
    main = _fit_lgbm(LGBM_PARAMS, train, val, target)
    q_params = {**LGBM_PARAMS, "objective": "quantile"}
    low = _fit_lgbm({**q_params, "alpha": 0.10}, train, val, target)
    high = _fit_lgbm({**q_params, "alpha": 0.90}, train, val, target)
    lgbm_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    negbin = NegBinBaseline(FEATURES).fit(train[FEATURES], train[target])
    negbin_seconds = time.perf_counter() - t0

    # ---- Evaluate on the untouched 2025 test year ----
    y = test[target].to_numpy()
    p = np.clip(main.predict(test[FEATURES]), 0, None)
    lo = np.clip(low.predict(test[FEATURES]), 0, None)
    hi = np.maximum(np.clip(high.predict(test[FEATURES]), 0, None), p)

    ly = last_year_baseline(feat, h).loc[test.index]
    ly_mask = ly.notna().to_numpy()

    metrics = {
        "lightgbm": evaluate(y, p, lo, hi),
        "negative_binomial_glm": evaluate(y, negbin.predict(test[FEATURES])),
        "same_week_last_year": evaluate(y[ly_mask], ly.to_numpy()[ly_mask]),
        "persistence_cases_now": evaluate(y, persistence_baseline(test)),
        "split": {
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "test_rows": int(len(test)),
            "train_target_weeks": f"{train[f'target_date_h{h}'].min().date()} to {train[f'target_date_h{h}'].max().date()}",
            "val_target_weeks": f"{val[f'target_date_h{h}'].min().date()} to {val[f'target_date_h{h}'].max().date()}",
            "test_target_weeks": f"{test[f'target_date_h{h}'].min().date()} to {test[f'target_date_h{h}'].max().date()}",
        },
        "best_iteration": {
            "main": int(main.best_iteration_ or main.n_estimators),
            "q10": int(low.best_iteration_ or low.n_estimators),
            "q90": int(high.best_iteration_ or high.n_estimators),
        },
        "train_seconds": {"lightgbm_3_models": round(lgbm_seconds, 2), "negbin": round(negbin_seconds, 2)},
        "feature_importance_gain": dict(
            sorted(
                zip(FEATURES, (main.booster_.feature_importance("gain") / 1.0).round(1).tolist()),
                key=lambda kv: kv[1], reverse=True,
            )
        ),
    }

    artifact = {
        "horizon": h,
        "main": main,
        "low": low,
        "high": high,
        "features": FEATURES,
        "thresholds": DEFAULT_THRESHOLDS,
        "trained_until": TRAIN_END,
    }
    return artifact, metrics


def predict_all(feat: pd.DataFrame, artifacts: dict[int, dict]) -> pd.DataFrame:
    """Precompute forecasts for every (ward, issue week) so the app only looks them up."""
    out = feat[["date", "city", "ward_id"]].copy()
    X = feat[FEATURES]
    for h, art in artifacts.items():
        p = np.clip(art["main"].predict(X), 0, None)
        out[f"pred_h{h}"] = p
        out[f"low_h{h}"] = np.clip(art["low"].predict(X), 0, None)
        out[f"high_h{h}"] = np.maximum(np.clip(art["high"].predict(X), 0, None), p)
    float_cols = [c for c in out.columns if c.startswith(("pred_", "low_", "high_"))]
    out[float_cols] = out[float_cols].astype("float32")
    return out


def train_all(raw_path: str | Path = RAW_DATA_PATH, verbose: bool = True) -> dict:
    t_total = time.perf_counter()
    raw = load_raw_data(raw_path)
    feat = build_features(raw)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(FEATURES_PATH, index=False)

    artifacts, all_metrics = {}, {}
    for h in HORIZONS:
        if verbose:
            print(f"Training horizon {h} weeks ahead ...")
        art, m = train_horizon(feat, h)
        joblib.dump(art, model_path(h), compress=3)
        artifacts[h], all_metrics[f"h{h}"] = art, m

    t0 = time.perf_counter()
    preds = predict_all(feat, artifacts)
    batch_ms = (time.perf_counter() - t0) * 1000
    preds.to_parquet(PREDICTIONS_PATH, index=False)

    # Single-row latency (what-if path): predict + driver breakdown.
    row = feat[FEATURES].iloc[[-1]]
    t0 = time.perf_counter()
    for _ in range(20):
        artifacts[2]["main"].predict(row, pred_contrib=True)
        artifacts[2]["low"].predict(row)
        artifacts[2]["high"].predict(row)
    single_ms = (time.perf_counter() - t0) * 1000 / 20

    report = {
        "test_year": TEST_YEAR,
        "horizons": all_metrics,
        "runtime": {
            "total_train_seconds": round(time.perf_counter() - t_total, 2),
            "precompute_all_predictions_ms": round(batch_ms, 1),
            "precomputed_rows": int(len(preds)),
            "single_whatif_prediction_ms": round(single_ms, 2),
        },
        "artifact_sizes_kb": {
            **{model_path(h).name: round(model_path(h).stat().st_size / 1024, 1) for h in HORIZONS},
            FEATURES_PATH.name: round(FEATURES_PATH.stat().st_size / 1024, 1),
            PREDICTIONS_PATH.name: round(PREDICTIONS_PATH.stat().st_size / 1024, 1),
        },
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2))
    return report


def load_artifacts() -> dict[int, dict]:
    return {h: joblib.load(model_path(h)) for h in HORIZONS}
