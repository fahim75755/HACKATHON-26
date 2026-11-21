import unittest
import pandas as pd

from dengue.features import build_features, load_raw_data


class TestFeaturesPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_df = load_raw_data("data/raw/dengue_data.csv")
        cls.feat_df, cls.feature_cols, cls.cat_maps, cls.weather_cols = build_features(
            cls.raw_df
        )

    def test_lag_and_interaction_columns_exist(self):
        rain_col = self.weather_cols["rainfall"]
        humid_col = self.weather_cols["humidity"]
        expected = [
            f"{rain_col}_lag_1w",
            f"{rain_col}_lag_4w",
            f"{humid_col}_lag_2w",
            "rain_roll_4w_sum",
            "rain_4w_x_waterlogging",
            "density_x_poor_sanitation",
        ]
        for col in expected:
            self.assertIn(col, self.feature_cols)

    def test_no_missing_values_after_lag_drop(self):
        self.assertEqual(int(self.feat_df[self.feature_cols].isnull().sum().sum()), 0)

    def test_ward_isolated_lags_no_leakage(self):
        # Verify lag_1w matches previous week's value within the exact same ward
        rain_col = self.weather_cols["rainfall"]
        sample_ward = self.raw_df["ward_id"].iloc[0]
        raw_w = self.raw_df[self.raw_df["ward_id"] == sample_ward].sort_values("date")
        feat_w = self.feat_df[self.feat_df["ward_id"] == sample_ward].sort_values("date")

        first_feat_date = feat_w["date"].iloc[0]
        prev_date = first_feat_date - pd.Timedelta(days=7)
        expected_lag1 = raw_w.loc[raw_w["date"] == prev_date, rain_col].iloc[0]
        actual_lag1 = feat_w[f"{rain_col}_lag_1w"].iloc[0]
        self.assertAlmostEqual(float(actual_lag1), float(expected_lag1), places=5)


if __name__ == "__main__":
    unittest.main()