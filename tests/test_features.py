import unittest

import numpy as np
import pandas as pd

from dengue import engine
from dengue.config import FEATURES, HORIZONS, RAIN_COL, TARGET_COL
from dengue.data import _parse_dates, load_raw_data, validate_raw
from dengue.features import build_features


class TestFeaturesPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_raw_data()
        cls.feat = build_features(cls.raw)
        cls.ward = cls.raw["ward_id"].iloc[0]

    def _ward(self, df):
        return df[df["ward_id"] == self.ward].sort_values("date").reset_index(drop=True)

    def test_all_features_present_and_complete(self):
        for col in FEATURES:
            self.assertIn(col, self.feat.columns)
        self.assertEqual(int(self.feat[FEATURES].isna().sum().sum()), 0)

    def test_rain_lags_are_ward_grouped(self):
        raw_w, feat_w = self._ward(self.raw), self._ward(self.feat)
        row = feat_w.iloc[10]
        idx = raw_w.index[raw_w["date"] == row["date"]][0]
        for k in range(4):
            self.assertAlmostEqual(float(row[f"rain_lag_{k}w"]), float(raw_w.loc[idx - k, RAIN_COL]), places=3)
        # First feature row of every ward starts at its 4th week (no borrowing from another ward)
        firsts = self.feat.groupby("ward_id")["date"].min()
        self.assertTrue((firsts == self.raw["date"].min() + pd.Timedelta(weeks=3)).all())

    def test_targets_are_future_cases(self):
        raw_w, feat_w = self._ward(self.raw), self._ward(self.feat)
        row = feat_w.iloc[100]
        for h in HORIZONS:
            later = raw_w.loc[raw_w["date"] == row["date"] + pd.Timedelta(weeks=h), TARGET_COL].iloc[0]
            self.assertEqual(float(row[f"target_h{h}"]), float(later))

    def test_no_future_leakage(self):
        """Changing everything after week d must not change the features for week d."""
        cut = pd.Timestamp("2022-06-06")
        changed = self.raw.copy()
        future = changed["date"] > cut
        changed.loc[future, [RAIN_COL, "relative_humidity_pct", "avg_temp_c"]] *= 3
        changed.loc[future, TARGET_COL] += 50
        a = build_features(self.raw)
        b = build_features(changed)
        a_cut = a[a["date"] <= cut].reset_index(drop=True)[FEATURES]
        b_cut = b[b["date"] <= cut].reset_index(drop=True)[FEATURES]
        pd.testing.assert_frame_equal(a_cut, b_cut)

    def test_single_row_builder_matches_batch(self):
        """The what-if / manual-entry path must produce the same inputs as training."""
        feat_w = self._ward(self.feat)
        for i in (5, 200, 400):
            row = feat_w.iloc[i]
            weeks = engine.recent_weeks(self.feat, self.ward, row["date"])
            traits = engine.ward_traits(self.feat, self.ward)
            X = engine.build_row(self.feat, self.ward, row["date"], weeks, traits)
            np.testing.assert_allclose(
                X[FEATURES].to_numpy(dtype=float)[0], row[FEATURES].to_numpy(dtype=float), rtol=1e-4, atol=1e-3
            )


class TestDataValidation(unittest.TestCase):
    def test_excel_serial_dates(self):
        parsed = _parse_dates(pd.Series([42373, 46020]))
        self.assertEqual(parsed.iloc[0], pd.Timestamp("2016-01-04"))
        self.assertEqual(parsed.iloc[1], pd.Timestamp("2025-12-29"))

    def test_rejects_duplicates(self):
        raw = load_raw_data()
        bad = pd.concat([raw, raw.iloc[[0]]])
        with self.assertRaises(ValueError):
            validate_raw(bad)


if __name__ == "__main__":
    unittest.main()
