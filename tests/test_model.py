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
