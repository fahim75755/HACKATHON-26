"""Train the dengue early-warning models.

Usage:  python train.py
Builds data/processed/features.parquet, trains 2/3/4-week LightGBM models plus
baselines, writes models/*.joblib, models/metrics.json and
data/processed/predictions.parquet.
"""
from dengue.model import train_all


def main():
    report = train_all()
    print("\n=== 2025 test-year results (never used for training or tuning) ===")
    header = f"{'horizon':<8}{'model':<26}{'MAE':>7}{'RMSE':>7}{'tier acc':>10}{'High+Crit recall':>18}"
    print(header)
    print("-" * len(header))
    for hk, m in report["horizons"].items():
        for name in ("lightgbm", "negative_binomial_glm", "same_week_last_year", "persistence_cases_now"):
            r = m[name]
            rec = r["high_critical_recall"]
            print(
                f"{hk:<8}{name:<26}{r['MAE']:>7.3f}{r['RMSE']:>7.3f}"
                f"{r['tier_accuracy']:>10.3f}{(rec if rec is not None else float('nan')):>18.3f}"
            )
        cov = m["lightgbm"].get("range_80_coverage")
        print(f"{'':<8}80% range coverage: {cov:.3f}   best iterations: {m['best_iteration']}")
    print("\nRuntime:", report["runtime"])
    print("Artifact sizes (KB):", report["artifact_sizes_kb"])


if __name__ == "__main__":
    main()
