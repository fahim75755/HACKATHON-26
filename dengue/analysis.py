"""Evaluation metrics and baseline models used to judge the LightGBM engine."""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from dengue.tiers import assign_risk_tier


def evaluate(y_true, y_pred, low=None, high=None) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    true_t = np.array([assign_risk_tier(v) for v in y_true])
    pred_t = np.array([assign_risk_tier(v) for v in y_pred])
    alert_true = np.isin(true_t, ["High", "Critical"])
    alert_pred = np.isin(pred_t, ["High", "Critical"])
    out = {
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "tier_accuracy": float(np.mean(true_t == pred_t)),
        # Of the weeks that really were High/Critical, how many did we flag?
        "high_critical_recall": float(alert_pred[alert_true].mean()) if alert_true.any() else None,
        # Of the weeks we flagged High/Critical, how many really were?
        "high_critical_precision": float(alert_true[alert_pred].mean()) if alert_pred.any() else None,
        "rows": int(len(y_true)),
    }
    if low is not None and high is not None:
        out["range_80_coverage"] = float(np.mean((y_true >= low) & (y_true <= high)))
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


class NegBinBaseline:
    """Negative-binomial GLM on standardized features (the classic disease-count model)."""

    def __init__(self, features: list[str]):
        self.features = features

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        Z = (X[self.features].to_numpy(dtype=float) - self.mean_) / self.std_
        return sm.add_constant(Z, has_constant="add")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NegBinBaseline":
        arr = X[self.features].to_numpy(dtype=float)
        self.mean_ = arr.mean(axis=0)
        self.std_ = arr.std(axis=0) + 1e-9
        y = np.asarray(y, dtype=float)
        # Estimate the dispersion (alpha) by method of moments: var = mu + alpha * mu^2.
        mu, var = y.mean(), y.var()
        self.alpha_ = float(max((var - mu) / mu**2, 1e-3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_ = sm.GLM(
                y, self._design(X), family=sm.families.NegativeBinomial(alpha=self.alpha_)
            ).fit(maxiter=200)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.predict(self._design(X)))


def last_year_baseline(features: pd.DataFrame, horizon: int) -> pd.Series:
    """Cases in the same ward, same week one year (52 weeks) before the target week."""
    shifted = features.groupby("ward_id")["dengue_cases"].shift(52 - horizon)
    return shifted


def persistence_baseline(features: pd.DataFrame) -> pd.Series:
    """'Nothing changes': predict this week's cases for the target week."""
    return features["cases_now"]
