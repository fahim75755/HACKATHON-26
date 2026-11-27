"""R5 offline resilience: kill-switch (network blocked) and the fallback ladder (plan tests 1 and 3)."""
import os
import socket
import unittest
import urllib.request
from unittest import mock

import pandas as pd

from dengue import notify, ops
from dengue.command import CommandParams, run_command_center
from dengue.config import FEATURES_PATH, PREDICTIONS_PATH
from dengue.fallback import check_connectivity, get_forecast, load_ward_metadata, reset_connectivity_cache

PARAMS = CommandParams(issue_date="2025-06-30", horizon=3, stock_pct=60, total_crews=8, offline=True)


class NetworkBlocked:
    """Every socket connect / urlopen raises and is counted."""

    def __init__(self):
        self.attempts = 0

    def _fail(self, *a, **k):
        self.attempts += 1
        raise OSError("network disabled by test (kill-switch)")

    def __enter__(self):
        self._patches = [mock.patch.object(socket.socket, "connect", self._fail),
                         mock.patch.object(socket, "create_connection", self._fail),
                         mock.patch.object(urllib.request, "urlopen", self._fail)]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()


class OfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (FEATURES_PATH.exists() and PREDICTIONS_PATH.exists()):
            raise unittest.SkipTest("Model files missing. Run `python train.py` first.")

    def test_kill_switch_full_pipeline_zero_network_calls(self):
        """Forecast -> allocation -> routing -> simulator -> notices -> export, with the network blocked."""
        reset_connectivity_cache()
        with mock.patch.dict(os.environ, {"ALERTAHEAD_OFFLINE": "1"}), NetworkBlocked() as net:
            self.assertFalse(check_connectivity())
            r = run_command_center(PARAMS)
            from dengue.ui_command import adapt_result
            wards, transfers, cities, _, source = adapt_result(r)
            ctx = notify.NoticeContext(target=pd.Timestamp("2025-07-21"), horizon=3, stock_pct=60, total_crews=8, scenario="offline")
            files = notify.export_files(wards, cities, transfers, ctx)
        self.assertEqual(net.attempts, 0, "offline run tried to use the network")
        self.assertEqual(source, "parquet")
        self.assertEqual(len(wards), 60)
        self.assertGreater(float(wards["prevented"].sum()), 0.0)          # simulator ran locally
        self.assertGreater(len(files), 300)

    def test_live_weather_falls_back_without_crashing(self):
        with NetworkBlocked():
            w = ops.fetch_live_weather("Dhaka North", timeout=0.5)
        self.assertEqual(w["source"], "offline_cache")

    def test_models_missing_simulator_falls_back_to_heuristic(self):
        with mock.patch("dengue.simulator._models", side_effect=FileNotFoundError("models/ renamed")):
            r = run_command_center(PARAMS)
        self.assertIn("heuristic", r.metadata["simulator"]["source"])
        self.assertEqual(int(r.wards["crews_assigned"].sum()), 8)
        self.assertEqual(float(r.wards["cases_prevented"].sum()), 0.0)

    def test_parquet_missing_uses_live_lightgbm(self):
        fc, source = get_forecast(pd.DataFrame(), "2025-06-30", 3)
        self.assertEqual(source, "live_lightgbm")
        ref, _ = get_forecast(None, "2025-06-30", 3)
        merged = fc.merge(ref, on="ward_id", suffixes=("_live", "_parquet"))
        diff = (merged["pred_h3_live"] - merged["pred_h3_parquet"]).abs().max()
        self.assertLess(float(diff), 0.01)  # float32 parquet vs float64 live

    def test_parquet_and_models_missing_uses_last_year(self):
        with mock.patch("dengue.model.load_artifacts", side_effect=FileNotFoundError("models/ renamed")):
            fc, source = get_forecast(pd.DataFrame(), "2025-06-30", 3, raw_df=load_ward_metadata())
        self.assertEqual(source, "fallback: seasonal")
        self.assertEqual(len(fc), 60)


if __name__ == "__main__":
    unittest.main()
