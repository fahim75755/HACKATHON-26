from pathlib import Path
from dengue.model import DEFAULT_MODEL_PATH, train_model


def main():
    print("Training LightGBM Dengue Early-Warning Engine...")
    artifact = train_model()

    size_kb = Path(DEFAULT_MODEL_PATH).stat().st_size / 1024
    print("\n=== Detected Weather Columns ===")
    print(artifact["weather_cols"])

    print("\n=== Chronological Split ===")
    print(artifact["split_info"])

    print("\n=== Validation Metrics ===")
    for k, v in artifact["metrics"].items():
        print(f"  {k}: {v}")

    print(f"\nModel saved to: {DEFAULT_MODEL_PATH} ({size_kb:.1f} KB)")
    print("\n=== Top 10 Features ===")
    for feat, score in artifact["top_features"]:
        print(f"  {feat}: {score}")


if __name__ == "__main__":
    main()