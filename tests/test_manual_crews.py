"""R3 "deploy crews to selected wards": manual crew assignment + SMS wording."""
import unittest

import pandas as pd

from dengue import notify
from dengue.command import CommandParams, run_command_center
from dengue.config import FEATURES_PATH, PREDICTIONS_PATH
from dengue.resources import manual_crew_assignment


class ManualCrewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (FEATURES_PATH.exists() and PREDICTIONS_PATH.exists()):
            raise unittest.SkipTest("Model files missing. Run `python train.py` first.")

    def _run(self, total, manual):
        return run_command_center(CommandParams(issue_date="2025-06-30", horizon=3, total_crews=total, manual_crews=manual)).wards

    def test_officer_selection_is_used(self):
        w = self._run(8, {"DNCC_Ward_10": 3, "CCC_Ward_01": 2}).set_index("ward_id")
        self.assertEqual(int(w.loc["DNCC_Ward_10", "crews_assigned"]), 3)
        self.assertEqual(int(w.loc["CCC_Ward_01", "crews_assigned"]), 2)
        self.assertEqual(int(w["crews_assigned"].sum()), 5)
        self.assertTrue((w.loc[w["crews_assigned"] == 0, "cases_prevented"] == 0).all())
        self.assertGreater(float(w.loc["DNCC_Ward_10", "cases_prevented"]), 0.0)

    def test_total_never_exceeds_slider(self):
        w = self._run(3, {"DNCC_Ward_10": 3, "CCC_Ward_01": 2})
        self.assertEqual(int(w["crews_assigned"].sum()), 3)
        self.assertEqual(manual_crew_assignment(["A", "B", "C"], {"B": 2, "C": 2}, 3).tolist(), [0, 2, 1])

    def test_empty_selection_sends_no_crews(self):
        w = self._run(8, {})
        self.assertEqual(int(w["crews_assigned"].sum()), 0)
        self.assertEqual(float(w["cases_prevented"].sum()), 0.0)

    def test_auto_is_unchanged_when_manual_is_none(self):
        self.assertEqual(int(self._run(8, None)["crews_assigned"].sum()), 8)

    def test_sms_says_one_kit(self):
        ctx = notify.NoticeContext(target=pd.Timestamp("2025-07-21"), horizon=3, stock_pct=60, total_crews=8)
        w = dict(ward_id="CCC_Ward_01", city="Chattogram", tier="High", pred=12, saline_req=45, saline_alloc=43,
                 kits_req=22, kits_alloc=21, beds_req=4, bed_cap=4, crews=1, unmet=0)
        self.assertIn("1 kit.", notify.sms_en(w, ctx))


if __name__ == "__main__":
    unittest.main()
