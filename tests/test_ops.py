"""Unit tests for operational decision modules (allocator, backtest, guardrails, field flags)."""
import unittest

import pandas as pd

from dengue.config import FEATURES_PATH, PREDICTIONS_PATH
from dengue.ops import (
    WARD_COORDS,
    allocate_teams,
    attach_coordinates,
    backtest_allocation,
    build_sms_preview,
    check_weather_bounds,
    flag_field_report,
)


class TestOps(unittest.TestCase):
    def _sample_week(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"ward_id": "DNCC_Ward_01", "city": "Dhaka North", "pred_h2": 22.0, "population_density_sqkm": 60000, "water_logging_risk_index": 0.9},
                {"ward_id": "CCC_Ward_01", "city": "Chattogram", "pred_h2": 14.0, "population_density_sqkm": 50000, "water_logging_risk_index": 0.8},
                {"ward_id": "DSCC_Ward_01", "city": "Dhaka South", "pred_h2": 4.0, "population_density_sqkm": 75000, "water_logging_risk_index": 0.95},
                {"ward_id": "GCC_Ward_01", "city": "Gazipur", "pred_h2": 1.0, "population_density_sqkm": 20000, "water_logging_risk_index": 0.2},
            ]
        )

    def test_all_60_wards_have_coordinates(self):
        self.assertEqual(len(WARD_COORDS), 60)
        df = attach_coordinates(self._sample_week())
        self.assertTrue({"lat", "lon"}.issubset(df.columns))
        self.assertFalse(df["lat"].isna().any())

    def test_allocator_beats_even_split_and_respects_overrides(self):
        df = self._sample_week()
        res = allocate_teams(df, n_teams=2, pred_col="pred_h2")
        self.assertEqual(res["teams_assigned"], 2)
        self.assertGreater(res["coverage_pct"], res["even_split_pct"])
        self.assertEqual(list(res["selected"]["ward_id"]), ["DNCC_Ward_01", "CCC_Ward_01"])

        # Human-in-the-loop: exclude top ward & lock low ward
        res_override = allocate_teams(
            df, n_teams=2, pred_col="pred_h2",
            locked_wards=["GCC_Ward_01"],
            excluded_wards=["DNCC_Ward_01"],
        )
        picked = list(res_override["selected"]["ward_id"])
        self.assertNotIn("DNCC_Ward_01", picked)
        self.assertIn("GCC_Ward_01", picked)

    def test_allocator_edge_cases(self):
        df = self._sample_week()
        self.assertEqual(allocate_teams(df, n_teams=0, pred_col="pred_h2")["teams_assigned"], 0)
        self.assertEqual(allocate_teams(pd.DataFrame(), n_teams=5, pred_col="pred_h2")["teams_assigned"], 0)

    def test_field_report_keyword_flag(self):
        bn = flag_field_report("রাস্তায় জমা পানি ও ড্রেনে মশার লার্ভা দেখা গেছে")
        self.assertTrue(bn["flagged"])
        self.assertEqual(bn["level"], "Urgent Citizen Signal")
        clean = flag_field_report("All clear after inspection")
        self.assertFalse(clean["flagged"])

    def test_weather_guardrail_and_sms(self):
        self.assertEqual(len(check_weather_bounds(50.0, 75.0, 28.0)), 0)
        self.assertGreaterEqual(len(check_weather_bounds(600.0, 105.0, 42.0)), 3)
        res = allocate_teams(self._sample_week(), n_teams=2, pred_col="pred_h2")
        sms = build_sms_preview(res["selected"], horizon=2, pred_col="pred_h2", is_bn=True)
        self.assertIn("DNCC_Ward_01", sms)

    def test_backtest_beats_even_split_on_2025_data(self):
        if not (PREDICTIONS_PATH.exists() and FEATURES_PATH.exists()):
            self.skipTest("Processed parquet files missing")
        preds = pd.read_parquet(PREDICTIONS_PATH)
        feats = pd.read_parquet(FEATURES_PATH)
        bt = backtest_allocation(preds, feats, horizon=2, n_teams=10, test_year=2025)
        self.assertGreater(bt["weeks"], 20)
        self.assertGreater(bt["alertahead_pct"], bt["even_split_pct"])
        self.assertLessEqual(bt["alertahead_pct"], bt["oracle_pct"] + 0.1)


if __name__ == "__main__":
    unittest.main()