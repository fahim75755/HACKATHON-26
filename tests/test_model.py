import unittest
from pathlib import Path

from dengue.features import build_features, load_raw_data
from dengue.model import DEFAULT_MODEL_PATH, load_model, predict_row
from dengue.tiers import assign_risk_tier, get_tier_info


class TestDengueModel(unittest.TestCase):
    def test_risk_tier_boundaries(self):
        self.assertEqual(assign_risk_tier(0.0), "Low")
        self.assertEqual(assign_risk_tier(2.0), "Medium")
        self.assertEqual(assign_risk_tier(5.0), "High")
        self.assertEqual(assign_risk_tier(10.0), "Critical")
        self.assertIn("action", get_tier_info("Critical"))

    def test_saved_model_inference_and_drivers(self):
        self.assertTrue(
            Path(DEFAULT_MODEL_PATH).exists(),
            "Trained model artifact missing. Run `python train.py` first.",
        )
        artifact = load_model(DEFAULT_MODEL_PATH)
        raw_df = load_raw_data("data/raw/dengue_data.csv")
        feat_df, _, _, _ = build_features(
            raw_df, category_maps=artifact["category_maps"]
        )

        result = predict_row(artifact, feat_df.iloc[[-1]], top_k=5)
        self.assertGreaterEqual(result["predicted_cases"], 0.0)
        self.assertGreaterEqual(result["ci_low"], 0.0)
        self.assertGreaterEqual(result["ci_high"], result["predicted_cases"])
        self.assertIn(result["risk_tier"], {"Low", "Medium", "High", "Critical"})
        self.assertEqual(len(result["top_drivers"]), 5)


if __name__ == "__main__":
    unittest.main()