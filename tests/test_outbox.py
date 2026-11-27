"""Commissioner outbox (bonus on R4): alert routing, access codes, brute-force brake, offline, UI sign-in."""
import socket
import unittest
import urllib.parse
from unittest import mock

import pandas as pd
from streamlit.testing.v1 import AppTest

from dengue import access, notify
from dengue import outbox as ob

CTX = notify.NoticeContext(target=pd.Timestamp("2025-06-30"), horizon=3, stock_pct=60, total_crews=8, scenario="test01")


def ward(**kw):
    base = dict(ward_id="DNCC_Ward_12", city="Dhaka North", tier="Critical", pred=23, p90=29.1, uses_p90=True,
                saline_req=69, saline_alloc=60, kits_req=34, kits_alloc=30, beds_req=7, bed_cap=4,
                sent=0, received=0, unmet=1, crews=2, crew_ids="Crew-01,Crew-05", prevented=6.4,
                deficit=True, red=True, code_black=True)
    base.update(kw)
    return base


WARDS = pd.DataFrame([
    ward(),                                                                             # CODE BLACK
    ward(ward_id="DNCC_Ward_05", tier="High", pred=14, unmet=0, code_black=False),       # RED
    ward(ward_id="DNCC_Ward_01", tier="Low", pred=1, saline_req=3, saline_alloc=3, kits_req=2, kits_alloc=2,
         beds_req=0, unmet=0, crews=0, crew_ids="", deficit=False, red=False, code_black=False),     # ROUTINE
    ward(ward_id="GCC_Ward_02", city="Gazipur", tier="Medium", pred=5, saline_req=12, saline_alloc=9, kits_req=6,
         kits_alloc=6, beds_req=1, bed_cap=2, unmet=0, crews=0, crew_ids="", red=False, code_black=False),  # SHORTFALL
])
CITIES = pd.DataFrame([
    {"city": "Dhaka North", "demand": 8, "capacity": 6, "utilization": 8 / 6, "routed": 1, "unmet": 1, "exhausted": True},
    {"city": "Gazipur", "demand": 1, "capacity": 2, "utilization": 0.5, "routed": 0, "unmet": 0, "exhausted": False},
])
TRANSFERS = pd.DataFrame(columns=["city", "from_ward", "to_ward", "patients", "km"])


def row(box, rid, level=None, cc=False):
    m = box[(box["recipient_id"] == rid) & (box["cc"] == cc)]
    return m if level is None else m[m["level"] == level]


class RoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.box = ob.build_outbox(WARDS, CITIES, TRANSFERS, CTX)

    def test_every_ward_reaches_its_commissioner_and_health_centre(self):
        for wid in WARDS["ward_id"]:
            for role in ("COMM", "HCM"):
                self.assertEqual(len(row(self.box, f"{role}-{notify.short_code(wid)}")), 1, (role, wid))

    def test_levels_follow_the_flags(self):
        self.assertEqual(row(self.box, "COMM-DNCC-12")["level"].item(), "CODE BLACK")
        self.assertEqual(row(self.box, "COMM-DNCC-05")["level"].item(), "RED")
        self.assertEqual(row(self.box, "COMM-GCC-02")["level"].item(), "SHORTFALL")
        self.assertEqual(row(self.box, "COMM-DNCC-01")["level"].item(), "ROUTINE")

    def test_escalation_goes_to_civil_surgeon_with_copy_to_code_black_commissioner(self):
        cs = row(self.box, "CS-DHAKA-NORTH", "ESCALATION")
        self.assertEqual(len(cs), 1)
        self.assertEqual(len(row(self.box, "COMM-DNCC-12", "ESCALATION", cc=True)), 1)
        self.assertTrue(row(self.box, "CS-GAZIPUR").empty)              # calm city: no escalation
        self.assertTrue(row(self.box, "COMM-DNCC-05", "ESCALATION", cc=True).empty)   # red, not black: no copy

    def test_most_urgent_first_and_ids_unique(self):
        self.assertTrue(self.box["priority"].is_monotonic_increasing)
        self.assertEqual(self.box["priority"].iloc[0], 1)
        self.assertTrue(self.box["msg_id"].is_unique)

    def test_every_sms_is_one_gsm_segment(self):
        for text in self.box["sms_en"]:
            info = notify.sms_info(text)
            self.assertTrue(info["gsm"], text)
            self.assertLessEqual(info["chars"], 160, text)

    def test_escalation_sms_shortens_a_long_ward_list(self):
        many = pd.concat([WARDS.iloc[[0]].assign(ward_id=f"DNCC_Ward_{i:02d}") for i in range(10, 30)], ignore_index=True)
        text = ob.escalation_sms("Dhaka North", many, CITIES, CTX)
        self.assertLessEqual(len(text), 160)
        self.assertTrue(notify.sms_info(text)["gsm"])

    def test_deterministic_and_exportable(self):
        again = ob.build_outbox(WARDS, CITIES, TRANSFERS, CTX)
        pd.testing.assert_frame_equal(self.box, again)
        files = ob.outbox_files(self.box)
        self.assertEqual(files, ob.outbox_files(again))
        self.assertIn("outbox/outbox.csv", files)
        self.assertIn("outbox/by_recipient/CS-DHAKA-NORTH.txt", files)
        self.assertIn("outbox/by_recipient/COMM-DNCC-05.txt", files)
        self.assertNotIn("outbox/by_recipient/COMM-DNCC-01.txt", files)   # routine only: no urgent file

    def test_sms_link_carries_the_text(self):
        text = self.box["sms_en"].iloc[0]
        link = ob.sms_link(text)
        self.assertTrue(link.startswith("sms:?body="))
        self.assertEqual(urllib.parse.unquote(link.split("body=", 1)[1]), text)

    def test_builds_with_the_network_blocked(self):
        def fail(*a, **k):
            raise OSError("network disabled by test")
        with mock.patch.object(socket.socket, "connect", fail), mock.patch.object(socket, "create_connection", fail):
            box = ob.build_outbox(WARDS, CITIES, TRANSFERS, CTX)
        self.assertEqual(len(box), len(self.box))

    def test_directory_lists_everyone(self):
        d = ob.recipients(WARDS)
        self.assertEqual(len(d), 2 * len(WARDS) + WARDS["city"].nunique())
        self.assertTrue(set(self.box["recipient_id"]) <= set(d["recipient_id"]))


class AccessCodeTests(unittest.TestCase):
    KEYS = access.load_keys({"ACCESS_SECRET": "a" * 32, "ADMIN_SECRET": "b" * 32}, env={})
    IDS = ["COMM-DNCC-05", "HCM-DNCC-05", "CS-DHAKA-NORTH"]

    def test_code_shape_and_round_trip(self):
        code = access.code_for(self.KEYS, "COMM-DNCC-05")
        self.assertRegex(code, r"^[A-Z2-9]{5}-[A-Z2-9]{5}$")
        self.assertEqual(access.who_is(code, self.KEYS, self.IDS), "COMM-DNCC-05")
        self.assertEqual(access.who_is(code.lower().replace("-", " "), self.KEYS, self.IDS), "COMM-DNCC-05")

    def test_wrong_or_malformed_code_is_rejected(self):
        self.assertIsNone(access.who_is("AAAAA-AAAAA", self.KEYS, self.IDS))
        self.assertIsNone(access.who_is("", self.KEYS, self.IDS))
        self.assertIsNone(access.who_is("short", self.KEYS, self.IDS))

    def test_admin_uses_its_own_secret(self):
        admin = access.code_for(self.KEYS, access.ADMIN_ID)
        self.assertEqual(access.who_is(admin, self.KEYS, self.IDS), access.ADMIN_ID)
        self.assertNotEqual(admin, access.make_code(self.KEYS.access, access.ADMIN_ID))

    def test_version_bump_revokes_one_code(self):
        old = access.code_for(self.KEYS, "COMM-DNCC-05")
        bumped = access.load_keys({"ACCESS_SECRET": "a" * 32, "ADMIN_SECRET": "b" * 32,
                                   "code_versions": {"COMM-DNCC-05": 2}}, env={})
        self.assertIsNone(access.who_is(old, bumped, self.IDS))
        self.assertEqual(access.who_is(access.code_for(bumped, "COMM-DNCC-05"), bumped, self.IDS), "COMM-DNCC-05")
        other = access.code_for(self.KEYS, "HCM-DNCC-05")
        self.assertEqual(access.who_is(other, bumped, self.IDS), "HCM-DNCC-05")   # others unaffected

    def test_changing_the_secret_revokes_every_code(self):
        rotated = access.load_keys({"ACCESS_SECRET": "c" * 32, "ADMIN_SECRET": "b" * 32}, env={})
        for rid in self.IDS:
            self.assertIsNone(access.who_is(access.code_for(self.KEYS, rid), rotated, self.IDS))

    def test_demo_mode_without_secrets(self):
        demo = access.load_keys({}, env={})
        self.assertTrue(demo.demo)
        self.assertFalse(self.KEYS.demo)
        self.assertNotEqual(access.code_for(demo, "COMM-DNCC-05"), access.code_for(self.KEYS, "COMM-DNCC-05"))


class LimiterTests(unittest.TestCase):
    def test_five_wrong_codes_lock_that_client(self):
        t = [1000.0]
        lim = access.AttemptLimiter(max_fails=5, window_s=300, lock_s=300, clock=lambda: t[0])
        for _ in range(4):
            lim.record_failure("ip-1")
        self.assertEqual(lim.locked_for("ip-1"), 0)
        lim.record_failure("ip-1")
        self.assertGreater(lim.locked_for("ip-1"), 0)
        self.assertEqual(lim.locked_for("ip-2"), 0)          # other clients unaffected
        t[0] += 301
        self.assertEqual(lim.locked_for("ip-1"), 0)          # lock expires

    def test_old_failures_fall_out_of_the_window(self):
        t = [0.0]
        lim = access.AttemptLimiter(max_fails=5, window_s=60, lock_s=300, clock=lambda: t[0])
        for _ in range(4):
            lim.record_failure()
        t[0] += 61
        lim.record_failure()
        self.assertEqual(lim.locked_for(), 0)


def _outbox_page():
    import pandas as pd
    from dengue import notify
    from dengue.ui_outbox import render_outbox
    base = dict(city="Dhaka North", pred=14, saline_req=69, saline_alloc=60, kits_req=34, kits_alloc=30, beds_req=7, bed_cap=4,
                sent=0, received=0, crews=2, crew_ids="Crew-01", prevented=3.0, deficit=True)
    wards = pd.DataFrame([dict(base, ward_id="DNCC_Ward_12", tier="Critical", unmet=1, red=True, code_black=True),
                          dict(base, ward_id="DNCC_Ward_05", tier="High", unmet=0, red=True, code_black=False)])
    cities = pd.DataFrame([{"city": "Dhaka North", "demand": 14, "capacity": 8, "utilization": 1.75, "routed": 0, "unmet": 1, "exhausted": True}])
    transfers = pd.DataFrame(columns=["city", "from_ward", "to_ward", "patients", "km"])
    ctx = notify.NoticeContext(target=pd.Timestamp("2025-06-30"), horizon=3, stock_pct=60, total_crews=8, scenario="test01")
    render_outbox(wards, transfers, cities, ctx, False)


class OutboxUITests(unittest.TestCase):
    def test_demo_admin_and_commissioner_sign_in(self):
        at = AppTest.from_function(_outbox_page, default_timeout=60).run()
        self.assertFalse(at.exception, at.exception)
        self.assertTrue(any("Demo mode" in m.value for m in at.markdown))
        at.button(key="ob_demo_admin").click().run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(at.session_state["ob_user"], access.ADMIN_ID)
        self.assertGreaterEqual(len(at.metric), 4)
        at.button(key="ob_logout").click().run()
        code = access.code_for(access.load_keys({}, env={}), "COMM-DNCC-05")
        at.text_input(key="ob_code").input(code)
        at.button(key="ob_submit").click().run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(at.session_state["ob_user"], "COMM-DNCC-05")
        self.assertTrue(any("RED" in m.value for m in at.markdown))


if __name__ == "__main__":
    unittest.main()
