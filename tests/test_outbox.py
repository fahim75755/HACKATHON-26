"""Officer portal (bonus on R4): SMS queue, SQLite store, password login, admin panel, brute-force brake, offline."""
import os
import socket
import tempfile
import unittest
import urllib.parse
from pathlib import Path
from unittest import mock

import pandas as pd
from streamlit.testing.v1 import AppTest

from dengue import access, db, notify
from dengue import outbox as ob

CTX = notify.NoticeContext(target=pd.Timestamp("2025-06-30"), horizon=3, stock_pct=60, total_crews=8, scenario="test01")
FAST = 1_000   # PBKDF2 iterations in tests (the app uses access.ITERATIONS)


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
         beds_req=0, unmet=0, crews=0, crew_ids="", deficit=False, red=False, code_black=False),     # no shortfall
    ward(ward_id="GCC_Ward_02", city="Gazipur", tier="Medium", pred=5, saline_req=12, saline_alloc=9, kits_req=6,
         kits_alloc=6, beds_req=1, bed_cap=2, unmet=0, crews=0, crew_ids="", red=False, code_black=False),  # SHORTFALL
])
CITIES = pd.DataFrame([
    {"city": "Dhaka North", "demand": 8, "capacity": 6, "utilization": 8 / 6, "routed": 1, "unmet": 1, "exhausted": True},
    {"city": "Gazipur", "demand": 1, "capacity": 2, "utilization": 0.5, "routed": 0, "unmet": 0, "exhausted": False},
])
TRANSFERS = pd.DataFrame(columns=["city", "from_ward", "to_ward", "patients", "km"])
WARDS_MAP = dict(zip(WARDS["ward_id"], WARDS["city"]))


class QueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q = ob.build_queue(WARDS, CITIES, TRANSFERS, CTX)

    def item(self, to_id):
        return self.q[self.q["to_id"] == to_id]

    def test_only_wards_with_an_alert_are_queued(self):
        self.assertEqual(self.item("HC-DNCC-12")["level"].item(), "CODE BLACK")
        self.assertEqual(self.item("HC-DNCC-05")["level"].item(), "RED")
        self.assertEqual(self.item("HC-GCC-02")["level"].item(), "SHORTFALL")
        self.assertTrue(self.item("HC-DNCC-01").empty)            # no shortfall -> no SMS

    def test_sms_goes_to_the_wards_medical_centre(self):
        for m in self.q[self.q["to_role"] == "hc"].itertuples(index=False):
            self.assertEqual(m.to_id, ob.medical_centre_id(m.ward_id))
            self.assertIn(notify.short_code(m.ward_id), m.sms_en)

    def test_escalation_only_for_a_city_in_trouble(self):
        self.assertEqual(self.item("CS-DHAKA-NORTH")["level"].item(), "ESCALATION")
        self.assertTrue(self.item("CS-GAZIPUR").empty)

    def test_most_urgent_first_and_ids_unique(self):
        self.assertTrue(self.q["priority"].is_monotonic_increasing)
        self.assertEqual(self.q.iloc[0]["to_id"], "CS-DHAKA-NORTH")
        self.assertTrue(self.q["msg_id"].is_unique)

    def test_every_sms_is_one_gsm_segment(self):
        for text in self.q["sms_en"]:
            info = notify.sms_info(text)
            self.assertTrue(info["gsm"], text)
            self.assertLessEqual(info["chars"], 160, text)

    def test_escalation_sms_shortens_a_long_ward_list(self):
        many = pd.concat([WARDS.iloc[[0]].assign(ward_id=f"DNCC_Ward_{i:02d}") for i in range(10, 30)], ignore_index=True)
        text = ob.escalation_sms("Dhaka North", many, CITIES, CTX)
        self.assertLessEqual(len(text), 160)
        self.assertTrue(notify.sms_info(text)["gsm"])

    def test_each_officer_sees_their_own_part(self):
        self.assertEqual(len(ob.visible(self.q, {"role": "admin"})), len(self.q))
        dn = ob.visible(self.q, {"role": "city", "city": "Dhaka North"})
        self.assertEqual(set(dn["city"]), {"Dhaka North"})
        self.assertIn("CS-DHAKA-NORTH", set(dn["to_id"]))              # the city officer sends the escalation
        w5 = ob.visible(self.q, {"role": "ward", "ward_id": "DNCC_Ward_05"})
        self.assertEqual(list(w5["to_id"]), ["HC-DNCC-05"])
        self.assertTrue(ob.visible(self.q, {"role": "ward", "ward_id": "DNCC_Ward_01"}).empty)
        self.assertTrue(ob.visible(self.q, None).empty)

    def test_deterministic_and_exportable(self):
        again = ob.build_queue(WARDS, CITIES, TRANSFERS, CTX)
        pd.testing.assert_frame_equal(self.q, again)
        files = ob.queue_files(self.q)
        self.assertEqual(files, ob.queue_files(again))
        for path in ("outbox/queue.csv", "outbox/by_city/Dhaka_North.txt", "outbox/by_city/Gazipur.txt"):
            self.assertIn(path, files)

    def test_sms_link_carries_number_and_text(self):
        text = self.q["sms_en"].iloc[0]
        self.assertTrue(ob.sms_link(text).startswith("sms:?body="))
        link = ob.sms_link(text, "+8801700000000")
        self.assertTrue(link.startswith("sms:+8801700000000?body="))
        self.assertEqual(urllib.parse.unquote(link.split("body=", 1)[1]), text)

    def test_builds_with_the_network_blocked(self):
        def fail(*a, **k):
            raise OSError("network disabled by test")
        with mock.patch.object(socket.socket, "connect", fail), mock.patch.object(socket, "create_connection", fail):
            q = ob.build_queue(WARDS, CITIES, TRANSFERS, CTX)
        self.assertEqual(len(q), len(self.q))


class PasswordTests(unittest.TestCase):
    def test_hash_is_salted_and_verifies(self):
        a, b = access.hash_password("s3cret-pass", FAST), access.hash_password("s3cret-pass", FAST)
        self.assertNotEqual(a, b)                          # different salt each time
        self.assertNotIn("s3cret-pass", a)                 # never stored in clear
        self.assertTrue(access.verify_password("s3cret-pass", a))
        self.assertFalse(access.verify_password("wrong-pass", a))
        self.assertFalse(access.verify_password("x", "garbage"))

    def test_generated_passwords(self):
        p = access.new_password()
        self.assertEqual(len(p), 12)
        self.assertTrue(set(p) <= set(access.PASSWORD_ALPHABET))


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.path = Path(self.tmp.name) / "portal.db"
        self.store = db.Store(self.path, iterations=FAST)

    def tearDown(self):
        self.tmp.cleanup()

    def test_demo_seed_and_login(self):
        self.assertTrue(self.store.demo)
        for user, pw, _n, role, ward_id, city in db.DEMO_ACCOUNTS:
            o = self.store.authenticate(user, pw)
            self.assertIsNotNone(o, user)
            self.assertEqual((o["role"], o["ward_id"], o["city"]), (role, ward_id, city))
        self.assertIsNone(self.store.authenticate("admin", "wrong-password"))
        self.assertIsNone(self.store.authenticate("nobody", "admin-demo-2026"))
        self.assertIsNotNone(self.store.authenticate("  ADMIN ", "admin-demo-2026"))   # trims and lower-cases

    def test_private_admin_has_no_demo_accounts(self):
        s = db.Store(Path(self.tmp.name) / "private.db", admin_password="Very-Secret-1", admin_user="chief", iterations=FAST)
        self.assertFalse(s.demo)
        self.assertEqual(list(s.officers()["username"]), ["chief"])
        self.assertIsNotNone(s.authenticate("chief", "Very-Secret-1"))
        self.assertIsNone(s.authenticate("admin", "admin-demo-2026"))

    def test_demo_database_adopts_private_admin(self):
        """A database first started in demo mode, reopened with ADMIN_USER / ADMIN_PASSWORD set."""
        s = db.Store(self.path, admin_password="Chief-Pass-9", admin_user=" Chief ", iterations=FAST)
        self.assertFalse(s.demo)
        o = s.authenticate("chief", "Chief-Pass-9")
        self.assertEqual(o["role"], "admin")
        for user, pw, *_ in db.DEMO_ACCOUNTS:                      # public demo passwords stop working
            self.assertIsNone(s.authenticate(user, pw), user)
        self.assertEqual(set(s.officers().query("active == 0")["username"]), {u for u, *_ in db.DEMO_ACCOUNTS})
        s.set_password("chief", "changed-in-app", by="chief")      # reopening never resets a changed password
        again = db.Store(self.path, admin_password="Chief-Pass-9", admin_user="chief", iterations=FAST)
        self.assertIsNotNone(again.authenticate("chief", "changed-in-app"))
        self.assertEqual(int((again.officers()["username"] == "chief").sum()), 1)

    def test_add_officer_and_validation(self):
        self.store.add_officer("dncc05", "Ward 5 officer", "ward", "longpassword", ward_id="DNCC_Ward_05", wards=WARDS_MAP)
        o = self.store.authenticate("dncc05", "longpassword")
        self.assertEqual((o["role"], o["ward_id"], o["city"]), ("ward", "DNCC_Ward_05", "Dhaka North"))   # city filled from ward
        bad = [dict(username="dncc05", full_name="Dup", role="ward", password="longpassword", ward_id="DNCC_Ward_05"),
               dict(username="x", full_name="Short", role="ward", password="longpassword", ward_id="DNCC_Ward_05"),
               dict(username="ok.user", full_name="", role="ward", password="longpassword", ward_id="DNCC_Ward_05"),
               dict(username="ok.user", full_name="Pw", role="ward", password="short", ward_id="DNCC_Ward_05"),
               dict(username="ok.user", full_name="No ward", role="ward", password="longpassword"),
               dict(username="ok.user", full_name="Bad ward", role="ward", password="longpassword", ward_id="XX_Ward_99"),
               dict(username="ok.user", full_name="Bad city", role="city", password="longpassword", city="Atlantis"),
               dict(username="ok.user", full_name="Bad role", role="king", password="longpassword")]
        for kw in bad:
            with self.assertRaises(db.StoreError, msg=kw):
                self.store.add_officer(**kw, wards=WARDS_MAP)

    def test_disable_reset_and_last_admin_guard(self):
        self.store.update_officer("dncc10", "DNCC Ward 10", "ward", ward_id="DNCC_Ward_10", active=False)
        self.assertIsNone(self.store.authenticate("dncc10", "ward-demo-2026"))        # disabled cannot log in
        self.store.set_password("dhaka.north", "brand-new-pass", by="admin")
        self.assertIsNone(self.store.authenticate("dhaka.north", "city-demo-2026"))
        self.assertIsNotNone(self.store.authenticate("dhaka.north", "brand-new-pass"))
        with self.assertRaises(db.StoreError):
            self.store.update_officer("admin", "Control room", "admin", active=False)  # the only admin
        self.store.add_officer("second.admin", "Second admin", "admin", "longpassword")
        self.store.update_officer("admin", "Control room", "admin", active=False)       # now allowed

    def test_contacts_phones_and_validation(self):
        self.store.seed_contacts(ob.contacts_seed(WARDS))
        df = self.store.contacts()
        self.assertEqual(len(df), len(WARDS) + WARDS["city"].nunique())
        df.loc[df["contact_id"] == "HC-DNCC-05", "phone"] = "+8801700000000"
        self.assertEqual(self.store.save_contacts(df, by="admin"), 1)
        self.assertEqual(self.store.phones(), {"HC-DNCC-05": "+8801700000000"})
        self.store.seed_contacts(ob.contacts_seed(WARDS))                              # re-seed keeps the number
        self.assertEqual(self.store.phones()["HC-DNCC-05"], "+8801700000000")
        df.loc[df["contact_id"] == "HC-DNCC-05", "phone"] = "call me"
        with self.assertRaises(db.StoreError):
            self.store.save_contacts(df, by="admin")

    def test_sms_status_and_audit_persist(self):
        self.store.set_status("abc123", "sent", "dncc10")
        self.store.set_status("abc123", "ack", "dncc10")
        again = db.Store(self.path, iterations=FAST)                                   # reopen the same file
        self.assertEqual(again.statuses()["abc123"]["status"], "ack")
        self.assertIn("ack", set(again.audit()["action"]))
        with self.assertRaises(db.StoreError):
            again.set_status("abc123", "deleted", "x")

    def test_works_with_the_network_blocked(self):
        def fail(*a, **k):
            raise OSError("network disabled by test")
        with mock.patch.object(socket.socket, "connect", fail), mock.patch.object(socket, "create_connection", fail):
            s = db.Store(Path(self.tmp.name) / "offline.db", iterations=FAST)
            self.assertIsNotNone(s.authenticate("admin", "admin-demo-2026"))


class LimiterTests(unittest.TestCase):
    def test_five_wrong_logins_lock_that_client(self):
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


def _portal_page():
    import pandas as pd
    from dengue import notify
    from dengue.ui_outbox import render_outbox
    base = dict(city="Dhaka North", pred=14, saline_req=69, saline_alloc=60, kits_req=34, kits_alloc=30, beds_req=7, bed_cap=4,
                sent=0, received=0, crews=2, crew_ids="Crew-01", prevented=3.0, deficit=True)
    wards = pd.DataFrame([dict(base, ward_id="DNCC_Ward_10", tier="Critical", unmet=1, red=True, code_black=True),
                          dict(base, ward_id="DNCC_Ward_05", tier="High", unmet=0, red=True, code_black=False)])
    cities = pd.DataFrame([{"city": "Dhaka North", "demand": 14, "capacity": 8, "utilization": 1.75, "routed": 0, "unmet": 1, "exhausted": True}])
    transfers = pd.DataFrame(columns=["city", "from_ward", "to_ward", "patients", "km"])
    ctx = notify.NoticeContext(target=pd.Timestamp("2025-06-30"), horizon=3, stock_pct=60, total_crews=8, scenario="test01")
    render_outbox(wards, transfers, cities, ctx, False)


class PortalUITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.env = mock.patch.dict(os.environ, {"ALERTAHEAD_DB": str(Path(self.tmp.name) / "ui.db")})
        self.env.start()
        # Demo mode regardless of the developer's own .streamlit/secrets.toml (ADMIN_USER / ADMIN_PASSWORD)
        self.secrets = mock.patch("dengue.ui_outbox._secret", return_value=None)
        self.secrets.start()

    def tearDown(self):
        self.secrets.stop()
        self.env.stop()
        self.tmp.cleanup()

    @staticmethod
    def login(at, user, pw):
        at.text_input(key="op_user").input(user)
        at.text_input(key="op_pw").input(pw)
        at.button(key="op_submit").click().run()

    def test_admin_login_panel_and_add_officer(self):
        at = AppTest.from_function(_portal_page, default_timeout=60).run()
        self.assertFalse(at.exception, at.exception)
        shown = " ".join([m.value for m in at.markdown] + [str(d.value) for d in at.dataframe])
        self.assertFalse(any(pw in shown for _u, pw, *_ in db.DEMO_ACCOUNTS))   # the page never lists a password
        self.login(at, "admin", "admin-demo-2026")
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(at.session_state["op_username"], "admin")
        self.assertEqual(at.metric[0].value, "3")                       # 2 ward SMS + 1 escalation
        self.assertTrue(any("Admin panel" in m.value for m in at.markdown))
        at.text_input(key="op_add_user").input("dncc05")
        at.text_input(key="op_add_name").input("Ward 5 officer")
        at.selectbox(key="op_add_ward").set_value("DNCC_Ward_05")
        at.text_input(key="op_add_pw").input("longpassword")
        at.button(key="op_add_go").click().run()
        self.assertFalse(at.exception, at.exception)
        s = db.Store(os.environ["ALERTAHEAD_DB"], iterations=FAST)
        self.assertIsNotNone(s.authenticate("dncc05", "longpassword"))

    def test_ward_officer_sees_only_their_ward(self):
        at = AppTest.from_function(_portal_page, default_timeout=60).run()
        self.login(at, "dncc10", "ward-demo-2026")
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(at.metric[0].value, "1")
        self.assertTrue(any("HC-DNCC-10" in m.value for m in at.markdown))
        self.assertFalse(any("Admin panel" in m.value for m in at.markdown))
        at.button(key="op_logout").click().run()
        self.assertNotIn("op_username", at.session_state)

    def test_wrong_password_is_refused(self):
        at = AppTest.from_function(_portal_page, default_timeout=60).run()
        self.login(at, "admin", "not-the-password")
        self.assertFalse(at.exception, at.exception)
        self.assertNotIn("op_username", at.session_state)
        self.assertTrue(any("Wrong user name or password" in e.value for e in at.error))


if __name__ == "__main__":
    unittest.main()
