"""R4 notice templates: content, SMS limits, escalation rule, determinism."""
import unittest

import pandas as pd

from dengue import notify

CTX = notify.NoticeContext(target=pd.Timestamp("2025-06-30"), horizon=3, stock_pct=60, total_crews=8, scenario="test01")


def ward(**kw):
    base = dict(ward_id="DNCC_Ward_12", city="Dhaka North", tier="Critical", pred=23, p90=29.1, uses_p90=True,
                saline_req=69, saline_alloc=60, kits_req=34, kits_alloc=30, beds_req=7, bed_cap=4,
                sent=2, received=0, unmet=1, crews=2, crew_ids="Crew-01,Crew-05", prevented=6.4)
    base.update(kw)
    return base


TRANSFERS = pd.DataFrame([{"city": "Dhaka North", "from_ward": "DNCC_Ward_12", "to_ward": "DNCC_Ward_11", "patients": 2, "km": 1.9}])


class NoticeTests(unittest.TestCase):
    def test_sms_en_is_one_gsm_sms(self):
        for w in (ward(), ward(unmet=0), ward(saline_alloc=69, kits_alloc=34, unmet=0), ward(tier="Low", pred=1)):
            for aud in ("comm", "hcm"):
                s = notify.sms_en(w, CTX, TRANSFERS, aud)
                info = notify.sms_info(s)
                self.assertTrue(info["gsm"], s)
                self.assertLessEqual(info["chars"], 160, s)
                self.assertEqual(info["parts"], 1)

    def test_sms_bn_at_most_three_parts(self):
        s = notify.sms_bn(ward(), CTX)
        self.assertLessEqual(len(s), notify.SMS_BN_MAX)
        self.assertLessEqual(notify.sms_info(s)["parts"], 3)
        self.assertIn("কোড ব্ল্যাক", s)

    def test_every_notice_has_the_four_items(self):
        msg = notify.whatsapp(ward(), CTX, TRANSFERS, "comm", "en")
        self.assertIn("Expected cases: *23*", msg)            # predicted cases
        self.assertIn("Saline: 60 of 69 bags", msg)           # allocated vs required
        self.assertIn("CODE BLACK", msg)                      # deficit alert
        self.assertIn("Spraying crews: 2 (Crew-01, Crew-05)", msg)  # assigned teams
        brief = notify.brief_html(ward(), CTX, TRANSFERS, "comm", "both")
        for token in ("Expected cases", "Allocated", "CODE BLACK", "Crew-01", "আনুমানিক রোগী"):
            self.assertIn(token, brief)

    def test_alert_levels(self):
        ok = ward(saline_alloc=69, kits_alloc=34, unmet=0)
        self.assertIn("No shortfall", notify.whatsapp(ok, CTX, None))
        red = ward(unmet=0)
        self.assertIn("RED ALERT", notify.whatsapp(red, CTX, None))
        short = ward(tier="Medium", unmet=0)
        self.assertIn("Shortfall", notify.whatsapp(short, CTX, None))

    def test_escalation_only_when_needed(self):
        cities = pd.DataFrame([{"city": "Dhaka North", "demand": 70, "capacity": 60, "utilization": 70 / 60,
                                "routed": 9, "unmet": 1, "exhausted": True}])
        wards = pd.DataFrame([ward()])
        self.assertIsNotNone(notify.escalation("Dhaka North", wards, cities, CTX))
        calm_c = cities.assign(unmet=0, exhausted=False)
        calm_w = pd.DataFrame([ward(unmet=0)])
        self.assertIsNone(notify.escalation("Dhaka North", calm_w, calm_c, CTX))

    def test_exports_are_deterministic(self):
        cities = pd.DataFrame([{"city": "Dhaka North", "demand": 7, "capacity": 4, "utilization": 1.75,
                                "routed": 2, "unmet": 1, "exhausted": True}])
        wards = pd.DataFrame([ward(), ward(ward_id="DNCC_Ward_11", tier="Low", pred=1, saline_req=2, saline_alloc=2,
                                            kits_req=1, kits_alloc=1, beds_req=0, sent=0, received=2, unmet=0, crews=0, crew_ids="")])
        a = notify.export_files(wards, cities, TRANSFERS, CTX)
        b = notify.export_files(wards, cities, TRANSFERS, CTX)
        self.assertEqual(a, b)
        self.assertEqual(notify.zip_bytes(a), notify.zip_bytes(b))
        self.assertIn("escalation/Dhaka_North_civil_surgeon.txt", a)
        self.assertIn("briefs/DNCC_Ward_12_commissioner.html", a)


if __name__ == "__main__":
    unittest.main()
