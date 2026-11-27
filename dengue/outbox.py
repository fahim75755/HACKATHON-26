"""Alert routing + outbox: who receives which dispatch notice (a bonus on top of R4; fully offline).

Rules (one row per recipient and message):
  * every ward      -> its ward commissioner (COMM-...) and health-centre manager (HCM-...)
                       level = CODE BLACK / RED / SHORTFALL / ROUTINE from notify.flags()
  * a city with a CODE BLACK ward or an exhausted bed pool
                    -> the city's civil surgeon (CS-...) gets an ESCALATION,
                       and the commissioner of every CODE BLACK ward gets a copy (cc)
Priority: 1 = CODE BLACK / ESCALATION, 2 = RED, 3 = SHORTFALL, 4 = ROUTINE.

Nothing is sent from here. The officer / commissioner copies the text or taps the `sms:` link, which
opens the phone's own SMS app with the text filled in, so the app never needs a gateway, an API key or
the internet. Message text comes from dengue/notify.py, so it always matches the Dispatch tab and exports.
Pure functions, no clock: the same scenario gives the same outbox.
"""
from __future__ import annotations

import hashlib
import urllib.parse

import pandas as pd

from dengue import notify

PRIORITY = {"CODE BLACK": 1, "ESCALATION": 1, "RED": 2, "SHORTFALL": 3, "ROUTINE": 4}
ICON = {"CODE BLACK": "⚫", "ESCALATION": "⚫", "RED": "🔴", "SHORTFALL": "🟠", "ROUTINE": "🟢"}
ROLE_NAME = {"comm": ("Ward commissioner", "ওয়ার্ড কমিশনার"), "hcm": ("Health-centre manager", "স্বাস্থ্যকেন্দ্র ব্যবস্থাপক"),
             "cs": ("Civil surgeon", "সিভিল সার্জন")}
ROLE_ORDER = {"cs": 0, "comm": 1, "hcm": 2}
OUTBOX_COLS = ["msg_id", "priority", "level", "recipient_id", "role", "ward_id", "city", "cc", "sms_en", "sms_bn", "whatsapp"]


def recipient_id(role: str, ward_or_city: str) -> str:
    """COMM-DNCC-05, HCM-DNCC-05, CS-DHAKA-NORTH."""
    if role == "cs":
        return "CS-" + str(ward_or_city).upper().replace(" ", "-")
    return f"{role.upper()}-{notify.short_code(ward_or_city)}"


def recipients(wards: pd.DataFrame) -> pd.DataFrame:
    """Everyone who can receive a notice for these wards (the access-code directory)."""
    rows = []
    for _, w in wards.sort_values("ward_id").iterrows():
        for role in ("comm", "hcm"):
            rows.append({"recipient_id": recipient_id(role, w["ward_id"]), "role": role, "ward_id": w["ward_id"], "city": w["city"]})
    for city in sorted(wards["city"].unique()):
        rows.append({"recipient_id": recipient_id("cs", city), "role": "cs", "ward_id": "", "city": city})
    return pd.DataFrame(rows, columns=["recipient_id", "role", "ward_id", "city"])


def ward_level(w) -> str:
    f = notify.flags(w)
    return "CODE BLACK" if f["black"] else "RED" if f["red"] else "SHORTFALL" if f["deficit"] else "ROUTINE"


def escalation_sms(city: str, wards: pd.DataFrame, cities: pd.DataFrame, ctx: notify.NoticeContext) -> str | None:
    """One-segment English SMS for the civil surgeon (GSM-7, <= 160 chars), or None if no escalation."""
    if notify.escalation(city, wards, cities, ctx, "en") is None:
        return None
    ws = wards[wards["city"] == city]
    black = [notify.short_code(r["ward_id"]) for _, r in ws.iterrows() if notify.flags(r)["black"]]
    c = cities[cities["city"] == city].iloc[0] if not cities.empty and (cities["city"] == city).any() else {}
    unmet = int(notify._g(c, "unmet", 0))
    sal = int(sum(notify.shortfall(r)[0] for _, r in ws.iterrows()))
    d = notify.fmt_date(ctx.target, "en", True).replace(" ", "")
    for shown in range(len(black), -1, -1):
        listed = ",".join(black[:shown]) + (f"+{len(black) - shown}" if shown < len(black) else "")
        text = (f"AlertAhead ESCALATION {city} wk {d}: {unmet} without bed after routing. "
                f"CODE BLACK {listed or 'none'}. Saline short {sal}. Request surge beds.")
        if len(text) <= notify.SMS_EN_MAX:
            return text
    return text[:notify.SMS_EN_MAX]


def _msg_id(ctx: notify.NoticeContext, rid: str, level: str, where: str, cc: bool) -> str:
    return hashlib.sha1(f"{ctx.scenario}|{rid}|{level}|{where}|{cc}".encode()).hexdigest()[:10]


def build_outbox(wards: pd.DataFrame, cities: pd.DataFrame, transfers: pd.DataFrame | None,
                 ctx: notify.NoticeContext) -> pd.DataFrame:
    """Every message, sorted most urgent first."""
    rows = []
    for _, w in wards.sort_values("ward_id").iterrows():
        level = ward_level(w)
        for role in ("comm", "hcm"):
            rid = recipient_id(role, w["ward_id"])
            rows.append({"msg_id": _msg_id(ctx, rid, level, w["ward_id"], False), "priority": PRIORITY[level], "level": level,
                         "recipient_id": rid, "role": role, "ward_id": w["ward_id"], "city": w["city"], "cc": False,
                         "sms_en": notify.sms_en(w, ctx, transfers, role), "sms_bn": notify.sms_bn(w, ctx),
                         "whatsapp": notify.whatsapp(w, ctx, transfers, role, "both")})
    for city in sorted(wards["city"].unique()):
        sms = escalation_sms(city, wards, cities, ctx)
        if sms is None:
            continue
        long_text = notify.escalation(city, wards, cities, ctx, "both")
        rid = recipient_id("cs", city)
        rows.append({"msg_id": _msg_id(ctx, rid, "ESCALATION", city, False), "priority": 1, "level": "ESCALATION",
                     "recipient_id": rid, "role": "cs", "ward_id": "", "city": city, "cc": False,
                     "sms_en": sms, "sms_bn": "", "whatsapp": long_text})
        for _, w in wards[wards["city"] == city].sort_values("ward_id").iterrows():
            if notify.flags(w)["black"]:
                crid = recipient_id("comm", w["ward_id"])
                rows.append({"msg_id": _msg_id(ctx, crid, "ESCALATION", city, True), "priority": 1, "level": "ESCALATION",
                             "recipient_id": crid, "role": "comm", "ward_id": w["ward_id"], "city": city, "cc": True,
                             "sms_en": sms, "sms_bn": "", "whatsapp": long_text})
    box = pd.DataFrame(rows, columns=OUTBOX_COLS)
    if box.empty:
        return box
    box["_r"] = box["role"].map(ROLE_ORDER)
    box = box.sort_values(["priority", "city", "ward_id", "_r", "cc"]).drop(columns="_r").reset_index(drop=True)
    return box


def sms_link(text: str, phone: str = "") -> str:
    """`sms:` link: on a phone it opens the SMS app with the text filled in (works without mobile data)."""
    return f"sms:{phone}?body={urllib.parse.quote(text)}"


def outbox_files(box: pd.DataFrame) -> dict[str, str]:
    """outbox/outbox.csv (every message) + one text file per recipient who has an urgent message (priority <= 2)."""
    files = {"outbox/outbox.csv": box.drop(columns=["whatsapp"]).to_csv(index=False, lineterminator="\n")}
    urgent = box[box["priority"] <= 2]
    for rid in sorted(urgent["recipient_id"].unique()):
        mine = box[box["recipient_id"] == rid]
        parts = [f"To: {rid}", ""]
        for _, m in mine.iterrows():
            head = f"[{m['level']}{' · copy' if m['cc'] else ''}] {m['ward_id'] or m['city']}"
            parts += [head, m["sms_en"], ""]
        files[f"outbox/by_recipient/{rid}.txt"] = "\n".join(parts).rstrip() + "\n"
    return files
