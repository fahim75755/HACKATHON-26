import json
import unittest

import numpy as np
import pandas as pd

from dengue import engine
from dengue.config import FEATURES, FEATURES_PATH, HORIZONS, METRICS_PATH, PREDICTIONS_PATH, model_path
from dengue.model import load_artifacts, split_by_target_date
from dengue.tiers import assign_risk_tier, get_tier_info


class TestTiers(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(assign_risk_tier(0.0), "Low")
        self.assertEqual(assign_risk_tier(2.99), "Low")
        self.assertEqual(assign_risk_tier(3.0), "Medium")
        self.assertEqual(assign_risk_tier(9.0), "High")
        self.assertEqual(assign_risk_tier(20.0), "Critical")
        self.assertIn("action", get_tier_info("Critical"))

    def test_watch_flag(self):
        self.assertEqual(engine.watch_tier(7.0, 14.0), "High")      # Medium, range reaches High
        self.assertEqual(engine.watch_tier(12.0, 25.0), "Critical")  # High, range reaches Critical
        self.assertIsNone(engine.watch_tier(1.0, 5.0))               # only reaches Medium: no flag
        self.assertIsNone(engine.watch_tier(10.0, 15.0))             # already High


class TestTrainedEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for p in [FEATURES_PATH, PREDICTIONS_PATH, METRICS_PATH] + [model_path(h) for h in HORIZONS]:
            if not p.exists():
                raise unittest.SkipTest("Model files missing. Run `python train.py` first.")
        cls.models = load_artifacts()
        cls.feat = pd.read_parquet(FEATURES_PATH)
        cls.preds = pd.read_parquet(PREDICTIONS_PATH)
        cls.metrics = json.loads(METRICS_PATH.read_text())

    def test_chronological_split(self):
        for h in HORIZONS:
            parts = split_by_target_date(self.feat, h)
            t = {k: v[f"target_date_h{h}"] for k, v in parts.items()}
            self.assertLess(t["train"].max(), t["val"].min())
            self.assertLess(t["val"].max(), t["test"].min())
            self.assertTrue((t["test"].dt.year == 2025).all())

    def test_forecasts_valid_for_every_horizon(self):
        for h in HORIZONS:
            p, lo, hi = (self.preds[f"{k}_h{h}"].to_numpy() for k in ("pred", "low", "high"))
            self.assertTrue((p >= 0).all() and (lo >= 0).all())
            self.assertTrue((hi >= p).all())

    def test_beats_simple_baselines_on_2025(self):
        for h in HORIZONS:
            m = self.metrics["horizons"][f"h{h}"]
            self.assertLess(m["lightgbm"]["MAE"], m["same_week_last_year"]["MAE"])
            self.assertLess(m["lightgbm"]["MAE"], m["negative_binomial_glm"]["MAE"])

    def test_live_prediction_matches_precomputed(self):
        row = self.feat.iloc[[5000]]
        ward, date = row["ward_id"].iloc[0], row["date"].iloc[0]
        for h in HORIZONS:
            live = engine.predict_live(self.models[h], row, date)
            cached = engine.lookup(self.preds, self.feat, ward, date, h)
            self.assertAlmostEqual(live.cases, cached.cases, places=3)
            self.assertEqual(live.tier, cached.tier)

    def test_drivers_add_up_to_forecast(self):
        row = self.feat.iloc[[12000]]
        art = self.models[3]
        table, base = engine.drivers(art, row, top_k=len(FEATURES))
        pred = float(art["main"].predict(row[FEATURES])[0])
        self.assertAlmostEqual(base + table["effect_cases"].sum(), pred, places=3)

    def test_outlook_gives_three_horizons_from_same_day(self):
        row = self.feat.iloc[5000]
        fcs = engine.outlook(self.models, self.preds, self.feat, row["ward_id"], row["date"])
        self.assertEqual([f.horizon for f in fcs], list(HORIZONS))
        self.assertTrue(all(f.issue_date == row["date"] for f in fcs))
        live = engine.outlook(self.models, self.preds, self.feat, row["ward_id"], row["date"],
                              self.feat.iloc[[5000]])
        for a, b in zip(fcs, live):
            self.assertAlmostEqual(a.cases, b.cases, places=3)

    def test_all_wards_outlook_ranked(self):
        allw = engine.all_wards_outlook(self.preds, pd.Timestamp("2025-07-14"))
        self.assertEqual(len(allw), self.feat["ward_id"].nunique())
        self.assertTrue((allw["tier_rank"].diff().dropna() <= 0).all())

    def test_predict_table_matches_precomputed_and_single_ward_path(self):
        date = pd.Timestamp("2025-07-14")
        frows = self.feat[self.feat["date"] == date]
        live = engine.predict_table(self.models, frows).set_index("ward_id")
        saved = self.preds[self.preds["date"] == date].set_index("ward_id")
        for h in HORIZONS:
            np.testing.assert_allclose(live.loc[saved.index, f"pred_h{h}"], saved[f"pred_h{h}"], rtol=1e-4, atol=1e-4)
        # City-wide extra rain gives the same forecast as the single-ward what-if path
        ward = "DNCC_Ward_01"
        wet_all = engine.predict_table(self.models, frows, extra_rain_mm=100).set_index("ward_id")
        weeks = engine.add_recent_rain(engine.recent_weeks(self.feat, ward, date), 100)
        X = engine.build_row(self.feat, ward, date, weeks, engine.ward_traits(self.feat, ward))
        for f in engine.outlook(self.models, self.preds, self.feat, ward, date, X):
            self.assertAlmostEqual(float(wet_all.loc[ward, f"pred_h{f.horizon}"]), f.cases, places=2)

    def test_extra_rain_works_in_first_weeks_of_data(self):
        """Regression: one ward + Feb 2016 + extra rain crashed (fewer than 4 recorded weeks)."""
        ward = "CCC_Ward_03"
        date = self.feat["date"].min() + pd.Timedelta(weeks=1)
        frow = self.feat[(self.feat["ward_id"] == ward) & (self.feat["date"] == date)]
        pd.testing.assert_frame_equal(engine.add_rain_to_features(frow, 0), frow)
        fcs = engine.outlook(self.models, self.preds, self.feat, ward, date, engine.add_rain_to_features(frow, 100))
        self.assertEqual([f.horizon for f in fcs], list(HORIZONS))
        self.assertTrue(all(f.cases >= 0 for f in fcs))

    def test_extra_recent_rain_raises_later_forecast_in_monsoon(self):
        ward, date = "DNCC_Ward_01", pd.Timestamp("2025-07-14")
        weeks = engine.recent_weeks(self.feat, ward, date)
        traits = engine.ward_traits(self.feat, ward)
        wet = engine.add_recent_rain(weeks, 100)
        self.assertAlmostEqual(float(wet["rainfall_mm"].iloc[-1]), float(weeks["rainfall_mm"].iloc[-1]) + 100, places=3)
        self.assertEqual(wet["rainfall_mm"].iloc[:-1].tolist(), weeks["rainfall_mm"].iloc[:-1].tolist())
        base = engine.predict_live(self.models[3], engine.build_row(self.feat, ward, date, weeks, traits), date)
        more = engine.predict_live(self.models[3], engine.build_row(self.feat, ward, date, wet, traits), date)
        self.assertGreater(more.cases, base.cases)

    def test_what_if_more_rain_raises_risk_in_monsoon(self):
        ward = self.feat["ward_id"].iloc[0]
        date = pd.Timestamp("2025-07-14")
        weeks = engine.recent_weeks(self.feat, ward, date)
        traits = engine.ward_traits(self.feat, ward)
        base = engine.predict_live(self.models[3], engine.build_row(self.feat, ward, date, weeks, traits), date)
        wet = engine.apply_what_if(weeks, rain_pct=100, humid_delta=0, temp_delta=0)
        more = engine.predict_live(self.models[3], engine.build_row(self.feat, ward, date, wet, traits), date)
        self.assertGreater(more.cases, base.cases)


if __name__ == "__main__":
    unittest.main()
