"""R3 simulator: model-driven counterfactual sanity (plan test 9 + model test 3)."""
import unittest

import numpy as np

from dengue.command import CommandParams, run_command_center
from dengue.config import FEATURES_PATH, PREDICTIONS_PATH
from dengue.simulator import MAX_CREWS_PER_WARD, counterfactual_rows, run_simulator


class SimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (FEATURES_PATH.exists() and PREDICTIONS_PATH.exists()):
            raise unittest.SkipTest("Model files missing. Run `python train.py` first.")
        cls.params = CommandParams(issue_date="2025-06-30", horizon=3, stock_pct=60, total_crews=8)
        cls.result = run_command_center(cls.params)
        cls.sim = run_simulator(cls.result.wards, cls.params)

    def test_zero_crews_prevent_exactly_zero(self):
        r = run_command_center(CommandParams(issue_date="2025-06-30", horizon=3, total_crews=0))
        self.assertEqual(float(r.wards["cases_prevented"].sum()), 0.0)
        for arr in self.sim["cum"].values():
            self.assertTrue(np.all(arr[0] == 0.0))

    def test_prevented_is_non_negative_and_grows_with_crews(self):
        for arr in self.sim["cum"].values():
            self.assertTrue(np.all(arr >= 0.0))
            self.assertTrue(np.all(np.diff(arr.sum(axis=1)) >= -1e-12))
        self.assertEqual(len(next(iter(self.sim["marginal_roi"].values()))), MAX_CREWS_PER_WARD)

    def test_crews_go_where_they_prevent_most(self):
        w = self.result.wards
        self.assertEqual(int(w["crews_assigned"].sum()), 8)
        self.assertGreater(float(w.loc[w["crews_assigned"] > 0, "cases_prevented"].sum()), 0.0)
        self.assertTrue((w.loc[w["crews_assigned"] == 0, "cases_prevented"] == 0).all())
        roi = w.loc[w["crews_assigned"] > 0, "roi_per_crew"]
        self.assertTrue(np.allclose(roi, w.loc[w["crews_assigned"] > 0, "cases_prevented"] / w.loc[w["crews_assigned"] > 0, "crews_assigned"]))

    def test_more_efficacy_prevents_more(self):
        lo = run_simulator(self.result.wards, CommandParams(issue_date="2025-06-30", efficacy=0.10))
        hi = run_simulator(self.result.wards, CommandParams(issue_date="2025-06-30", efficacy=0.40))
        tot = lambda s: sum(a[1].sum() for a in s["cum"].values())  # noqa: E731
        self.assertGreater(tot(hi), tot(lo))

    def test_counterfactual_only_touches_breeding_water(self):
        import pandas as pd
        rows = pd.read_parquet(FEATURES_PATH).query("date == '2025-06-30'").head(3)
        cf = counterfactual_rows(rows, 2, 0.3)
        for c in ("cases_now", "cases_mean_4w", "humid_mean_4w", "temp_mean_4w", "population_density_sqkm"):
            self.assertTrue(np.allclose(cf[c], rows[c]), c)
        self.assertTrue(np.allclose(cf["rain_lag_0w"], rows["rain_lag_0w"] * 0.49))

    def test_metadata_reports_model_source(self):
        self.assertIn("LightGBM", self.result.metadata["simulator"]["source"])


if __name__ == "__main__":
    unittest.main()
