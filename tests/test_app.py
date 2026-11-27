"""Smoke tests: the dashboard runs headless in English and Bangla, for all wards and for one ward."""
import datetime as dt
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from dengue.config import FEATURES_PATH, METRICS_PATH, PREDICTIONS_PATH

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _select(at, labels, value):
    box = next(s for s in at.selectbox if s.label in labels)
    box.set_value(value).run()


class TestDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all(p.exists() for p in (FEATURES_PATH, PREDICTIONS_PATH, METRICS_PATH)):
            raise unittest.SkipTest("Model files missing. Run `python train.py` first.")

    def _run(self):
        at = AppTest.from_file(APP, default_timeout=120).run()
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        return at

    def test_all_wards_has_2_3_4_week_tabs(self):
        at = self._run()
        labels = [t.label for t in at.tabs]
        self.assertEqual(labels[3], "📊 Accuracy (for reviewers)")
        self.assertNotIn("🚑 Command Center", labels)   # Phase 2 now opens from the sidebar "View" switch
        for h in (2, 3, 4):
            self.assertTrue(any(f"{h} weeks ahead" in x for x in labels), labels)
        self.assertTrue(any("of 60 wards" in m.value for m in at.markdown))

    def test_bangla_toggle(self):
        at = self._run()
        _select(at, ("Language / ভাষা",), "বাংলা")
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        self.assertTrue(any("সপ্তাহ" in t.label for t in at.tabs))

    def test_one_ward_shows_its_tier_in_each_tab(self):
        at = self._run()
        _select(at, ("Ward",), "DNCC_Ward_01")
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        week_tabs = [t.label for t in at.tabs][:3]
        self.assertTrue(all(any(tier in x for tier in ("Low", "Medium", "High", "Critical")) for x in week_tabs),
                        week_tabs)

    def test_one_ward_early_date_with_extra_rain(self):
        """Regression: one ward + Feb 2016 + extra rain used to crash the page."""
        at = self._run()
        _select(at, ("Ward",), "CCC_Ward_03")
        at.sidebar.date_input[0].set_value(dt.date(2016, 2, 1)).run()
        at.sidebar.slider[0].set_value(100).run()
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        self.assertEqual(len(at.tabs), 4)   # 2/3/4 weeks + accuracy

    def test_sidebar_view_switch_opens_command_center(self):
        """The Command Center sits in the sidebar, right below 'Forecast from week of'."""
        at = self._run()
        names = [getattr(w, "label", None) for w in at.sidebar.children.values()]
        self.assertLess(names.index("Forecast from week of"), names.index("View"))
        at.sidebar.radio(key="app_view").set_value("command").run()
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        labels = [t.label for t in at.tabs]
        self.assertTrue(any("Officer portal" in x for x in labels), labels)          # Command Center sub-tabs
        self.assertFalse(any("weeks ahead" in x for x in labels), labels)            # dashboard tabs hidden
        at.sidebar.radio(key="app_view").set_value("dashboard").run()
        self.assertTrue(any("weeks ahead" in t.label for t in at.tabs))

    def test_future_date_uses_typed_weather(self):
        at = self._run()
        _select(at, ("Ward",), "DNCC_Ward_01")
        at.date_input[0].set_value(dt.date(2026, 8, 3)).run()
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        self.assertTrue(any("No recorded weather" in i.value for i in at.info))


if __name__ == "__main__":
    unittest.main()
