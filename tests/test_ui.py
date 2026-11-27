"""UI smoke test for the Command Center tab only (small, so it stays well inside AppTest's timeout)."""
import unittest

from streamlit.testing.v1 import AppTest


def _command_center_page():
    import pandas as pd
    from dengue.ui_command import render_command_center
    render_command_center(pd.Timestamp("2025-06-09"), "All cities", False, pd.Timestamp("2025-12-29"))


def _command_center_page_bn_offline():
    import pandas as pd
    import streamlit as st
    from dengue.ui_command import render_command_center
    st.session_state["offline_mode"] = True
    render_command_center(pd.Timestamp("2025-06-09"), "Dhaka North", True, pd.Timestamp("2025-12-29"))


class CommandCenterUITests(unittest.TestCase):
    def test_renders_all_sub_tabs_without_errors(self):
        at = AppTest.from_function(_command_center_page, default_timeout=60).run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(len(at.tabs), 6)
        self.assertGreaterEqual(len(at.metric), 6)

    def test_bangla_and_offline(self):
        at = AppTest.from_function(_command_center_page_bn_offline, default_timeout=60).run()
        self.assertFalse(at.exception, at.exception)

    def test_stock_slider_changes_allocation(self):
        at = AppTest.from_function(_command_center_page, default_timeout=60).run()
        before = at.metric[1].value
        at.slider(key="cc_stock").set_value(100).run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(at.metric[1].value, "100%")
        self.assertNotEqual(before, at.metric[1].value)


if __name__ == "__main__":
    unittest.main()
