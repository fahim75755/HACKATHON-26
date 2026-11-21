import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from dengue.features import TARGET_COL, build_features, load_raw_data
from dengue.tiers import DEFAULT_THRESHOLDS, assign_risk_tier, get_tier_info

DEFAULT_MODEL_PATH = Path("models/lgbm_dengue_model.joblib")


def train_model(
    csv_path: str = "data/raw/dengue_data.csv",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    train_ratio: float = 0.80,
) -> dict:
    raw_df = load_raw_data(csv_path)
    feat_df, feature_cols, category_maps, weather_cols = build_features(raw_df)

    # Strict chronological split across all 60 wards (no temporal leakage)
    unique_dates = np.sort(feat_df["date"].unique())
    cutoff_date = pd.Timestamp(unique_dates[int(len(unique_dates) * train_ratio)])

    train_df = feat_df[feat_df["date"] < cutoff_date]
    val_df = feat_df[feat_df["date"] >= cutoff_date]

    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
    X_val, y_val = val_df[feature_cols], val_df[TARGET_COL]

    base_params = {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 30,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    t0 = time.perf_counter()

    # Main point-prediction regressor (L2 keeps pred_contrib in direct dengue-case units)
    main_model = lgb.LGBMRegressor(objective="regression", **base_params)
    main_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )

    # 80% Confidence Interval models (10th and 90th percentile quantile regressors)
    q_params = {**base_params, "n_estimators": max(100, main_model.best_iteration_ or 300)}
    lower_model = lgb.LGBMRegressor(objective="quantile", alpha=0.10, **q_params)
    upper_model = lgb.LGBMRegressor(objective="quantile", alpha=0.90, **q_params)
    lower_model.fit(X_train, y_train)
    upper_model.fit(X_train, y_train)

    train_seconds = time.perf_counter() - t0

    # Validation metrics
    val_preds = np.clip(main_model.predict(X_val), 0.0, None)
    val_low = np.clip(lower_model.predict(X_val), 0.0, None)
    val_high = np.maximum(val_preds, np.clip(upper_model.predict(X_val), 0.0, None))

    mae = float(mean_absolute_error(y_val, val_preds))
    rmse = float(root_mean_squared_error(y_val, val_preds))
    r2 = float(r2_score(y_val, val_preds))
    ci_coverage = float(np.mean((y_val >= val_low) & (y_val <= val_high)))

    actual_tiers = [assign_risk_tier(v, DEFAULT_THRESHOLDS) for v in y_val]
    pred_tiers = [assign_risk_tier(v, DEFAULT_THRESHOLDS) for v in val_preds]
    tier_acc = float(np.mean([a == p for a, p in zip(actual_tiers, pred_tiers)]))

    # Single-row inference latency check
    sample_row = X_val.iloc[[0]]
    t_inf = time.perf_counter()
    _ = main_model.predict(sample_row, pred_contrib=True)
    inf_ms = (time.perf_counter() - t_inf) * 1000.0

    # Feature importance ranking
    importances = sorted(
        zip(feature_cols, main_model.feature_importances_.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    artifact = {
        "main_model": main_model,
        "lower_model": lower_model,
        "upper_model": upper_model,
        "features": feature_cols,
        "category_maps": category_maps,
        "weather_cols": weather_cols,
        "thresholds": DEFAULT_THRESHOLDS,
        "split_info": {
            "cutoff_date": str(cutoff_date.date()),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
        },
        "metrics": {
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R2": round(r2, 4),
            "tier_accuracy": round(tier_acc, 4),
            "ci_80_coverage": round(ci_coverage, 4),
            "train_seconds": round(train_seconds, 2),
            "single_inference_ms": round(inf_ms, 2),
        },
        "top_features": importances[:10],
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path, compress=3)
    return artifact


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    return joblib.load(model_path)


def predict_row(artifact: dict, feature_row: pd.DataFrame, top_k: int = 5) -> dict:
    """Fast local CPU prediction with 80% confidence interval and SHAP driver breakdown."""
    features = artifact["features"]
    X = feature_row[features].iloc[[0]]

    pred = max(0.0, float(artifact["main_model"].predict(X)[0]))
    low = max(0.0, float(artifact["lower_model"].predict(X)[0]))
    high = max(pred, float(artifact["upper_model"].predict(X)[0]))
    tier = assign_risk_tier(pred, artifact["thresholds"])
    meta = get_tier_info(tier)

    # Built-in LightGBM TreeSHAP values (last entry is base expected value)
    contribs = artifact["main_model"].predict(X, pred_contrib=True)[0]
    base_value = float(contribs[-1])
    feature_impacts = sorted(
        zip(features, contribs[:-1]),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    return {
        "predicted_cases": round(pred, 2),
        "ci_low": round(low, 2),
        "ci_high": round(high, 2),
        "risk_tier": tier,
        "tier_color": meta["color"],
        "recommended_action": meta["action"],
        "base_value": round(base_value, 2),
        "top_drivers": [
            {"feature": f, "impact": round(float(v), 3)}
            for f, v in feature_impacts[:top_k]
        ],
    }