"""Alert SMS queue: which wards need an SMS to their medical centre (a bonus on top of R4; fully offline).

The queue (one item per alert, most urgent first):
  * every ward with a shortfall  -> SMS to that ward's medical centre (HC-...),
                                    level CODE BLACK / RED / SHORTFALL from notify.flags()
  * a city with a CODE BLACK ward or an exhausted bed pool
                                 -> ESCALATION SMS to the city's civil surgeon (CS-...)
Wards with no shortfall are not queued. Priority: 1 = CODE BLACK / ESCALATION, 2 = RED, 3 = SHORTFALL.

Who works the queue: officers signed in to the portal (accounts in dengue/db.py). A ward officer sees their
ward, a city health officer sees their whole city including escalations, an admin (control room) sees everything.
The officer copies the SMS or taps the `sms:` link, which opens the phone's own SMS app with the
centre's number and the text filled in, then marks it sent. The app itself sends nothing: no gateway,
no API key, no internet. Texts come from dengue/notify.py (the health-centre version), so they always
match the Dispatch tab and the exports. Pure functions, no clock: the same scenario gives the same queue.
"""
from __future__ import annotations

import hashlib
import urllib.parse

import pandas as pd

from dengue import notify

PRIORITY = {"CODE BLACK": 1, "ESCALATION": 1, "RED": 2, "SHORTFALL": 3}
ICON = {"CODE BLACK": "⚫", "ESCALATION": "⚫", "RED": "🔴", "SHORTFALL": "🟠"}
RECIPIENT_NAME = {"hc": ("Medical centre", "মেডিকেল সেন্টার"), "cs": ("Civil surgeon", "সিভিল সার্জন")}
ROLE_NAME = {"admin": ("Control room (admin)", "কন্ট্রোল রুম (অ্যাডমিন)"), "city": ("City health officer", "শহর স্বাস্থ্য কর্মকর্তা"),
             "ward": ("Ward officer", "ওয়ার্ড কর্মকর্তা")}
QUEUE_COLS = ["msg_id", "priority", "level", "ward_id", "city", "to_id", "to_role", "sms_en", "sms_bn", "whatsapp"]


def city_slug(city: str) -> str:
    return str(city).upper().replace(" ", "-")


def medical_centre_id(ward_id: str) -> str:
    return f"HC-{notify.short_code(ward_id)}"       # HC-DNCC-05


def civil_surgeon_id(city: str) -> str:
    return f"CS-{city_slug(city)}"                  # CS-DHAKA-NORTH


def contacts_seed(wards: pd.DataFrame) -> list[tuple[str, str]]:
    """Every SMS recipient as (contact_id, name): one medical centre per ward, one civil surgeon per city."""
    rows = [(medical_centre_id(w), f"Medical centre · {w}") for w in sorted(wards["ward_id"].unique())]
    rows += [(civil_surgeon_id(c), f"Civil surgeon · {c}") for c in sorted(wards["city"].unique())]
    return rows


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


def _msg_id(ctx: notify.NoticeContext, to_id: str, level: str, where: str) -> str:
    return hashlib.sha1(f"{ctx.scenario}|{to_id}|{level}|{where}".encode()).hexdigest()[:10]


def build_queue(wards: pd.DataFrame, cities: pd.DataFrame, transfers: pd.DataFrame | None,
                ctx: notify.NoticeContext) -> pd.DataFrame:
    """Every SMS that should go out for this scenario, most urgent first."""
    rows = []
    for _, w in wards.sort_values("ward_id").iterrows():
        level = ward_level(w)
        if level == "ROUTINE":
            continue
        to = medical_centre_id(w["ward_id"])
        rows.append({"msg_id": _msg_id(ctx, to, level, w["ward_id"]), "priority": PRIORITY[level], "level": level,
                     "ward_id": w["ward_id"], "city": w["city"], "to_id": to, "to_role": "hc",
                     "sms_en": notify.sms_en(w, ctx, transfers, "hcm"), "sms_bn": notify.sms_bn(w, ctx),
                     "whatsapp": notify.whatsapp(w, ctx, transfers, "hcm", "both")})
    for city in sorted(wards["city"].unique()):
        sms = escalation_sms(city, wards, cities, ctx)
        if sms is None:
            continue
        to = civil_surgeon_id(city)
        rows.append({"msg_id": _msg_id(ctx, to, "ESCALATION", city), "priority": 1, "level": "ESCALATION",
                     "ward_id": "", "city": city, "to_id": to, "to_role": "cs",
                     "sms_en": sms, "sms_bn": "", "whatsapp": notify.escalation(city, wards, cities, ctx, "both")})
    q = pd.DataFrame(rows, columns=QUEUE_COLS)
    if q.empty:
        return q
    q["_cs"] = (q["to_role"] != "cs").astype(int)          # within a city, the civil-surgeon escalation first
    return q.sort_values(["priority", "city", "_cs", "ward_id"]).drop(columns="_cs").reset_index(drop=True)


def visible(queue: pd.DataFrame, officer: dict | None) -> pd.DataFrame:
    """The part of the queue an officer works: admin = all, city officer = their city, ward officer = their ward."""
    role = (officer or {}).get("role")
    if role == "admin":
        return queue
    if role == "city":
        return queue[queue["city"] == officer.get("city")]
    if role == "ward":
        return queue[queue["ward_id"] == officer.get("ward_id")]
    return queue.iloc[0:0]


def sms_link(text: str, phone: str = "") -> str:
    """`sms:` link: on a phone it opens the SMS app with the number and text filled in (no mobile data needed)."""
    return f"sms:{phone}?body={urllib.parse.quote(text)}"


def queue_files(queue: pd.DataFrame) -> dict[str, str]:
    """outbox/queue.csv (every queued SMS) + one ready-to-send list per city for its health officer."""
    files = {"outbox/queue.csv": queue.drop(columns=["whatsapp"]).to_csv(index=False, lineterminator="\n")}
    if queue.empty:
        return files
    for city in sorted(queue["city"].unique()):
        mine = queue[queue["city"] == city]
        parts = [f"AlertAhead SMS queue · {city} · {len(mine)} message(s), most urgent first", ""]
        for _, m in mine.iterrows():
            parts += [f"[{m['level']}] {m['ward_id'] or city} -> {m['to_id']}", m["sms_en"], ""]
        files[f"outbox/by_city/{city.replace(' ', '_')}.txt"] = "\n".join(parts).rstrip() + "\n"
    return files
