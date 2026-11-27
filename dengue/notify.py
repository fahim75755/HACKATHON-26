"""R4 - multi-channel dispatch notices (SMS, WhatsApp, printable brief, city summary, escalation).

The one template engine for the app AND make_exports.py, so both always show the same numbers.
Pure functions, no network, no LLM, no clock: the same inputs always give byte-identical text.

A ward record `w` is a dict (or pandas row) with the CommandResult ward columns. Missing optional
columns fall back to safe defaults, so notices keep working while the backend is still growing.
"""
from __future__ import annotations

import html
import io
import math
import re
import zipfile
from dataclasses import dataclass

import pandas as pd

MONTHS_EN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS_BN = ["জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন", "জুলাই", "আগস্ট", "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর"]
BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
TIER_BN = {"Low": "কম", "Medium": "মাঝারি", "High": "উচ্চ", "Critical": "চরম"}
CITY_BN = {"Dhaka North": "ঢাকা উত্তর", "Dhaka South": "ঢাকা দক্ষিণ", "Chattogram": "চট্টগ্রাম", "Gazipur": "গাজীপুর", "Narayanganj": "নারায়ণগঞ্জ"}
ACTION = {
    "en": {
        "Low": "Routine larval surveillance and standard waste collection.",
        "Medium": "Inspect water-logged hotspots and issue community source-reduction advisories.",
        "High": "Deploy larvicide and fogging teams to this ward and alert local clinics.",
        "Critical": "Activate outbreak response: intensive vector control, hospital surge beds, saline and test-kit stock.",
    },
    "bn": {
        "Low": "নিয়মিত পরিচ্ছন্নতা বজায় রাখুন এবং বাড়ির আশপাশে জমে থাকা পানি সপ্তাহে একবার পরিষ্কার করুন।",
        "Medium": "বাড়ির টব, ছাদ ও ড্রেনে জমে থাকা পানি সরান, মশারি ব্যবহার করুন এবং লার্ভা পরীক্ষা চালু রাখুন।",
        "High": "দ্রুত ফগিং ও লার্ভিসাইড স্প্রে শুরু করুন এবং এলাকার হাসপাতাল ও ক্লিনিকগুলোকে সতর্ক করুন।",
        "Critical": "জরুরি ভিত্তিতে মশক নিধন অভিযান চালান এবং হাসপাতালে শয্যা, স্যালাইন ও টেস্ট কিট প্রস্তুত রাখুন।",
    },
}
SMS_EN_MAX, SMS_BN_MAX = 160, 201
_GSM_BASIC = re.compile(r"^[A-Za-z0-9 @$!\"#%&'()*+,\-./:;<=>?\n]*$")


@dataclass(frozen=True)
class NoticeContext:
    """Scenario facts printed on every notice (no wall-clock time -> deterministic)."""
    target: pd.Timestamp
    horizon: int
    stock_pct: int
    total_crews: int
    scenario: str = "-"
    source: str = "parquet"


# ------------------------------------------------------------------ small helpers
def _g(w, key, default=0):
    v = w.get(key, default) if hasattr(w, "get") else default
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    return v


def _i(w, key) -> int:
    return int(round(float(_g(w, key, 0))))


def num(x, lang: str = "en", d: int = 0) -> str:
    s = f"{x:,.{d}f}" if d else f"{int(round(float(x))):,}"
    return s.translate(BN_DIGITS) if lang == "bn" else s


def fmt_date(ts, lang: str = "en", short: bool = False) -> str:
    ts = pd.Timestamp(ts)
    if lang == "bn":
        out = f"{num(ts.day, 'bn')} {MONTHS_BN[ts.month - 1]}"
        return out if short else f"{out} {str(ts.year).translate(BN_DIGITS)}"
    out = f"{ts.day} {MONTHS_EN[ts.month - 1]}"
    return out if short else f"{out} {ts.year}"


def tier_name(tier: str, lang: str) -> str:
    return TIER_BN.get(tier, tier) if lang == "bn" else tier


def city_name(city: str, lang: str) -> str:
    return CITY_BN.get(city, city) if lang == "bn" else city


def short_code(ward_id: str) -> str:
    """DNCC_Ward_05 -> DNCC-05 (keeps SMS short and GSM-7 safe)."""
    return re.sub(r"^([A-Z]+)_Ward_(\d+)$", r"\1-\2", str(ward_id))


def crew_ids(w) -> list[str]:
    v = _g(w, "crew_ids", [])
    if isinstance(v, str):
        return [x for x in v.split(",") if x]
    return [str(x) for x in list(v)]


def shortfall(w) -> tuple[int, int]:
    return max(0, _i(w, "saline_req") - _i(w, "saline_alloc")), max(0, _i(w, "kits_req") - _i(w, "kits_alloc"))


def flags(w) -> dict:
    s, k = shortfall(w)
    deficit = bool(_g(w, "deficit", s > 0 or k > 0))
    red = bool(_g(w, "red", deficit and _g(w, "tier", "Low") in ("High", "Critical")))
    black = bool(_g(w, "code_black", s > 0 and _i(w, "unmet") > 0))
    return {"deficit": deficit, "red": red, "black": black}


def _kits(n: int) -> str:
    return "kit" if int(n) == 1 else "kits"


def sms_info(text: str) -> dict:
    """Length, encoding and number of SMS parts (GSM-7: 160/153, Unicode: 70/67)."""
    gsm = bool(_GSM_BASIC.match(text))
    n = len(text)
    parts = (1 if n <= 160 else math.ceil(n / 153)) if gsm else (1 if n <= 70 else math.ceil(n / 67))
    return {"chars": n, "gsm": gsm, "parts": parts}


def _transfers_for(transfers: pd.DataFrame | None, ward_id: str):
    if transfers is None or transfers.empty:
        return [], []
    out = transfers[transfers["from_ward"] == ward_id]
    inc = transfers[transfers["to_ward"] == ward_id]
    return list(out.itertuples(index=False)), list(inc.itertuples(index=False))


# ------------------------------------------------------------------ SMS
def sms_en(w, ctx: NoticeContext, transfers: pd.DataFrame | None = None, audience: str = "comm") -> str:
    """English SMS, GSM-7 characters only, at most 160 characters (one SMS)."""
    f, (s, k) = flags(w), shortfall(w)
    code, d = short_code(_g(w, "ward_id", "?")), fmt_date(ctx.target, "en", True).replace(" ", "")
    tier = str(_g(w, "tier", "Low")).upper()
    sent, _ = _transfers_for(transfers, _g(w, "ward_id", ""))
    if f["black"]:
        alert = f"CODE BLACK: no bed for {_i(w, 'unmet')}, saline short {s}."
    elif f["red"]:
        alert = f"RED: request {s} saline, {k} {_kits(k)}."
    elif f["deficit"]:
        alert = f"Short {s} saline, {k} {_kits(k)}."
    else:
        alert = "No shortfall."
    beds = f"Beds {_i(w, 'beds_req')}/{_i(w, 'bed_cap')}"
    beds_full = beds + (f" ({sum(t.patients for t in sent)} to {'+'.join(short_code(t.to_ward) for t in sent)})" if sent else "") \
        + (f" (+{_i(w, 'received')} in)" if _i(w, "received") else "")
    head = f"AlertAhead {'HCM ' if audience == 'hcm' else ''}{code} wk {d}: exp {_i(w, 'pred')} cases {tier}" + (" (watch)" if _g(w, "watch", False) else "") + "."
    supplies = f"Saline {_i(w, 'saline_alloc')}/{_i(w, 'saline_req')} Kits {_i(w, 'kits_alloc')}/{_i(w, 'kits_req')}."
    crews = f"Crews {_i(w, 'crews')}."
    for bed_txt in (beds_full, beds):
        parts = [head, bed_txt + ".", supplies, alert] if audience == "hcm" else [head, supplies, bed_txt + ".", crews, alert]
        text = " ".join(parts)
        if len(text) <= SMS_EN_MAX:
            return text
    return text.replace("AlertAhead ", "AA ")[:SMS_EN_MAX]


def sms_bn(w, ctx: NoticeContext) -> str:
    """Bangla SMS (Unicode), at most 3 parts = 201 characters."""
    f, (s, k) = flags(w), shortfall(w)
    b = lambda x: num(x, "bn")  # noqa: E731
    if f["black"]:
        alert = "কোড ব্ল্যাক: শয্যা ও স্যালাইন দুটোরই ঘাটতি"
    elif f["red"]:
        alert = f"লাল সতর্কতা: {b(s)} স্যালাইন, {b(k)} কিট চাই"
    elif f["deficit"]:
        alert = f"ঘাটতি: {b(s)} স্যালাইন, {b(k)} কিট"
    else:
        alert = "ঘাটতি নেই"
    wid, tier = _g(w, "ward_id", "?"), tier_name(_g(w, "tier", "Low"), "bn")
    core = (f"স্যালাইন {b(_i(w, 'saline_alloc'))}/{b(_i(w, 'saline_req'))}, কিট {b(_i(w, 'kits_alloc'))}/{b(_i(w, 'kits_req'))}, "
            f"শয্যা {b(_i(w, 'beds_req'))}/{b(_i(w, 'bed_cap'))}।")
    text = f"অ্যালার্টঅ্যাহেড {wid}: {fmt_date(ctx.target, 'bn', True)} সপ্তাহে আনুমানিক {b(_i(w, 'pred'))} রোগী ({tier})। {core} টিম {b(_i(w, 'crews'))}। {alert}।"
    if len(text) > SMS_BN_MAX:
        text = f"{wid}: {b(_i(w, 'pred'))} রোগী ({tier})। {core} {alert}।"
    return text[:SMS_BN_MAX]


# ------------------------------------------------------------------ WhatsApp
def whatsapp(w, ctx: NoticeContext, transfers: pd.DataFrame | None = None, audience: str = "comm", lang: str = "en") -> str:
    if lang == "both":
        return whatsapp(w, ctx, transfers, audience, "bn") + "\n\n— English —\n" + whatsapp(w, ctx, transfers, audience, "en")
    en, L = lang == "en", lang
    n = lambda x, d=0: num(x, L, d)  # noqa: E731
    f, (s, k) = flags(w), shortfall(w)
    wid, tier = _g(w, "ward_id", "?"), _g(w, "tier", "Low")
    sent, inc = _transfers_for(transfers, wid)
    route = ""
    if sent:
        route += ("\n   ↪ send " if en else "\n   ↪ পাঠান ") + ", ".join(f"{n(t.patients)} → {t.to_ward} ({n(t.km, 1)} km)" for t in sent)
    if inc:
        route += ("\n   ↩ receive " if en else "\n   ↩ গ্রহণ ") + ", ".join(f"{n(t.patients)} ← {t.from_ward}" for t in inc)
    if _i(w, "unmet"):
        route += f"\n   ⚠ {n(_i(w, 'unmet'))} patient(s) still without a bed" if en else f"\n   ⚠ {n(_i(w, 'unmet'))} জন রোগী এখনো শয্যা পাননি"
    if f["black"]:
        alert = (f"⬛ *CODE BLACK* — no bed for {n(_i(w, 'unmet'))} and saline short by {n(s)}. Escalated to the civil surgeon." if en
                 else f"⬛ *কোড ব্ল্যাক* — {n(_i(w, 'unmet'))} জনের শয্যা নেই, স্যালাইন ঘাটতি {n(s)}। সিভিল সার্জনকে জানানো হয়েছে।")
    elif f["red"]:
        alert = (f"🔴 *RED ALERT* — request {n(s)} saline + {n(k)} {_kits(k)} from the regional store today." if en
                 else f"🔴 *লাল সতর্কতা* — আজই আঞ্চলিক মজুদ থেকে {n(s)} স্যালাইন ও {n(k)} কিট চান।")
    elif f["deficit"]:
        alert = (f"🟠 *Shortfall* — {n(s)} saline, {n(k)} {_kits(k)} below need." if en
                 else f"🟠 *ঘাটতি* — প্রয়োজনের চেয়ে {n(s)} স্যালাইন, {n(k)} কিট কম।")
    else:
        alert = "🟢 No shortfall at this stock level." if en else "🟢 এই মজুদে কোনো ঘাটতি নেই।"
    ids = crew_ids(w)
    prev = float(_g(w, "prevented", 0.0))
    crews = (f"🚿 Spraying crews: {n(_i(w, 'crews'))}" if en else f"🚿 স্প্রে টিম: {n(_i(w, 'crews'))}") \
        + (f" ({', '.join(ids)})" if ids else "") \
        + ((f" · ~{n(prev, 1)} cases prevented" if en else f" · ~{n(prev, 1)} জন রোগী প্রতিরোধ") if _i(w, "crews") and prev > 0 else "")
    if en:
        head = [f"🦟 *AlertAhead · {'Health-centre notice' if audience == 'hcm' else 'Dispatch notice'}*",
                f"📍 *{wid}* · {_g(w, 'city', '')}", f"🗓 Week of {fmt_date(ctx.target)} (+{ctx.horizon} weeks)",
                f"📈 Expected cases: *{n(_i(w, 'pred'))}* · {tier}" + (" · watch" if _g(w, "watch", False) else "")]
        beds = f"🛏 Beds: {n(_i(w, 'beds_req'))} needed · {n(_i(w, 'bed_cap'))} available{route}"
        supply = [f"💧 Saline: {n(_i(w, 'saline_alloc'))} of {n(_i(w, 'saline_req'))} bags" + (f" (short {n(s)})" if s else ""),
                  f"🧪 Test kits: {n(_i(w, 'kits_alloc'))} of {n(_i(w, 'kits_req'))}" + (f" (short {n(k)})" if k else "")]
        foot = f"_Scenario {ctx.scenario} · stock {ctx.stock_pct}% · generated offline_"
    else:
        head = [f"🦟 *অ্যালার্টঅ্যাহেড · {'স্বাস্থ্যকেন্দ্র বার্তা' if audience == 'hcm' else 'নির্দেশনা বার্তা'}*",
                f"📍 *{wid}* · {city_name(_g(w, 'city', ''), 'bn')}", f"🗓 {fmt_date(ctx.target, 'bn')} সপ্তাহ (+{n(ctx.horizon)} সপ্তাহ)",
                f"📈 আনুমানিক রোগী: *{n(_i(w, 'pred'))}* · {tier_name(tier, 'bn')}" + (" · নজরে" if _g(w, "watch", False) else "")]
        beds = f"🛏 শয্যা: {n(_i(w, 'beds_req'))} প্রয়োজন · {n(_i(w, 'bed_cap'))} আছে{route}"
        supply = [f"💧 স্যালাইন: {n(_i(w, 'saline_req'))}টির মধ্যে {n(_i(w, 'saline_alloc'))} ব্যাগ" + (f" (ঘাটতি {n(s)})" if s else ""),
                  f"🧪 টেস্ট কিট: {n(_i(w, 'kits_req'))}টির মধ্যে {n(_i(w, 'kits_alloc'))}" + (f" (ঘাটতি {n(k)})" if k else "")]
        foot = f"_পরিস্থিতি {ctx.scenario} · মজুদ {n(ctx.stock_pct)}% · অফলাইনে তৈরি_"
    body = [beds] + supply if audience == "hcm" else supply + [beds, crews]
    return "\n".join(head + body + [alert, "✅ " + ACTION[L].get(tier, ACTION[L]["Low"]), foot])


# ------------------------------------------------------------------ printable brief (HTML -> browser Print -> PDF)
BRIEF_CSS = """@page{size:A4;margin:16mm}body{font-family:"BnSubset","Segoe UI","Nirmala UI","Noto Sans Bengali",system-ui,sans-serif;color:#16323B;font-size:12.5pt;margin:0;padding:18px}
h1{font-size:18pt;color:#0F4C5C;margin:0 0 4px}h2{font-size:13pt;color:#0F4C5C;margin:16px 0 6px;border-bottom:2px solid #246A73;padding-bottom:2px}
.meta{color:#4E646B;margin:0 0 10px}table{border-collapse:collapse;width:100%;margin:6px 0}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #DCE5E3;vertical-align:top}
table.grid th{background:#246A73;color:#fff}.alert{padding:8px 12px;border-radius:6px;margin:8px 0;font-weight:600}.alert.red{background:#FCE4E3;color:#6f1717}
.alert.black{background:#1f1f1f;color:#fff}.alert.warn{background:#FFF4D6;color:#6b4d00}.alert.ok{background:#E6F5E6;color:#0b4a0b}
.sign{display:flex;gap:40px;margin-top:28px}section+section{page-break-before:always;margin-top:30px}
.foot{margin-top:26px;font-size:9.5pt;color:#7E8C90;border-top:1px solid #DCE5E3;padding-top:6px}"""


def brief_html(w, ctx: NoticeContext, transfers: pd.DataFrame | None = None, audience: str = "comm", lang: str = "en",
               font_face_css: str = "") -> str:
    """Printable A4 brief. `font_face_css` can carry an @font-face for the Bangla subset font."""
    langs = ["bn", "en"] if lang == "both" else [lang]
    e = html.escape
    wid, tier = _g(w, "ward_id", "?"), _g(w, "tier", "Low")
    f, (s, k) = flags(w), shortfall(w)
    sent, inc = _transfers_for(transfers, wid)

    def section(L: str) -> str:
        en = L == "en"
        n = lambda x, d=0: num(x, L, d)  # noqa: E731
        aud = ("Health-centre manager" if en else "স্বাস্থ্যকেন্দ্র ব্যবস্থাপক") if audience == "hcm" else ("Ward commissioner" if en else "ওয়ার্ড কমিশনার")
        if f["black"]:
            alert, cls = ("CODE BLACK — no bed for {} patient(s) and saline short. Escalated to the civil surgeon.".format(n(_i(w, "unmet"))) if en
                          else "কোড ব্ল্যাক — {} জনের শয্যা নেই এবং স্যালাইন ঘাটতি। সিভিল সার্জনকে জানানো হয়েছে।".format(n(_i(w, "unmet")))), "black"
        elif f["red"]:
            alert, cls = ("RED ALERT — request the shortfall from the regional store today." if en else "লাল সতর্কতা — আজই আঞ্চলিক মজুদ থেকে ঘাটতি চান।"), "red"
        elif f["deficit"]:
            alert, cls = ("Shortfall — stock below need at this level." if en else "ঘাটতি — এই মজুদে প্রয়োজনের চেয়ে কম।"), "warn"
        else:
            alert, cls = ("No shortfall." if en else "কোনো ঘাটতি নেই।"), "ok"
        routing = [(f"Send {n(t.patients)} → {t.to_ward} ({n(t.km, 1)} km)" if en else f"পাঠান {n(t.patients)} → {t.to_ward} ({n(t.km, 1)} কিমি)") for t in sent] \
            + [(f"Receive {n(t.patients)} ← {t.from_ward}" if en else f"গ্রহণ {n(t.patients)} ← {t.from_ward}") for t in inc]
        ids = crew_ids(w)
        rows = [("Saline" if en else "স্যালাইন", _i(w, "saline_req"), _i(w, "saline_alloc"), s),
                ("Test kits" if en else "টেস্ট কিট", _i(w, "kits_req"), _i(w, "kits_alloc"), k)]
        plan_note = ""
        if _g(w, "uses_p90", False):
            plan_note = (" · planned on P90 = " if en else " · P90 ধরে পরিকল্পনা = ") + n(float(_g(w, "p90", 0.0)), 1)
        beds_cell = f"{n(_i(w, 'beds_req'))} {'needed' if en else 'প্রয়োজন'} · {n(_i(w, 'bed_cap'))} {'available' if en else 'আছে'}"
        if routing:
            beds_cell += "<br>" + "<br>".join(e(r) for r in routing)
        if _i(w, "unmet"):
            beds_cell += f"<br><b>{'Still without a bed: ' if en else 'এখনো শয্যা নেই: '}{n(_i(w, 'unmet'))}</b>"
        prev = float(_g(w, "prevented", 0.0))
        crew_cell = n(_i(w, "crews")) + (f" ({e(', '.join(ids))})" if ids else "") \
            + ((f" · ~{n(prev, 1)} cases prevented over 2–4 weeks" if en else f" · ~{n(prev, 1)} জন রোগী প্রতিরোধ (২–৪ সপ্তাহ)") if prev > 0 else "")
        return f"""<section lang="{L}">
<h1>{'AlertAhead · Commissioner brief' if en else 'অ্যালার্টঅ্যাহেড · কমিশনার ব্রিফ'}</h1>
<p class="meta"><b>{e(wid)}</b> · {e(city_name(_g(w, 'city', ''), L))} · {e(aud)} · {'Week of' if en else 'সপ্তাহ'} {e(fmt_date(ctx.target, L))} (+{n(ctx.horizon)} {'weeks' if en else 'সপ্তাহ'})</p>
<div class="alert {cls}">{e(alert)}</div>
<table><tr><th>{'Expected cases' if en else 'আনুমানিক রোগী'}</th><td><b>{n(_i(w, 'pred'))}</b> · {e(tier_name(tier, L))}{e(plan_note)}</td></tr>
<tr><th>{'Hospital beds' if en else 'হাসপাতাল শয্যা'}</th><td>{beds_cell}</td></tr>
<tr><th>{'Spraying crews' if en else 'স্প্রে টিম'}</th><td>{crew_cell}</td></tr></table>
<h2>{'Supplies' if en else 'সরবরাহ'}</h2>
<table class="grid"><tr><th>{'Item' if en else 'সামগ্রী'}</th><th>{'Required' if en else 'প্রয়োজন'}</th><th>{'Allocated' if en else 'বরাদ্দ'}</th><th>{'Shortfall' if en else 'ঘাটতি'}</th></tr>
{''.join(f'<tr><td>{e(r[0])}</td><td>{n(r[1])}</td><td>{n(r[2])}</td><td>{"<b>" + n(r[3]) + "</b>" if r[3] else n(0)}</td></tr>' for r in rows)}</table>
<h2>{'Recommended action' if en else 'সুপারিশকৃত পদক্ষেপ'}</h2><p>{e(ACTION[L].get(tier, ACTION[L]['Low']))}</p>
<div class="sign"><div>{'Prepared by' if en else 'প্রস্তুতকারী'}: ____________________</div><div>{'Approved by' if en else 'অনুমোদনকারী'}: ____________________</div></div>
</section>"""

    return (f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>{e(wid)} brief</title><style>{font_face_css}{BRIEF_CSS}</style></head><body>\n'
            + "\n".join(section(L) for L in langs)
            + f'\n<div class="foot">AlertAhead · scenario {e(ctx.scenario)} · stock {ctx.stock_pct}% · crews {ctx.total_crews} · source {e(ctx.source)} · generated offline from templates (no LLM)</div>\n</body></html>')


# ------------------------------------------------------------------ city summary & escalation
def _city_rows(wards: pd.DataFrame, city: str) -> pd.DataFrame:
    return wards[wards["city"] == city]


def city_summary(city: str, wards: pd.DataFrame, cities: pd.DataFrame, transfers: pd.DataFrame | None,
                 ctx: NoticeContext, lang: str = "en") -> str:
    if lang == "both":
        return city_summary(city, wards, cities, transfers, ctx, "bn") + "\n\n— English —\n" + city_summary(city, wards, cities, transfers, ctx, "en")
    en, n = lang == "en", (lambda x, d=0: num(x, lang, d))
    ws = _city_rows(wards, city)
    c = cities[cities["city"] == city].iloc[0] if not cities.empty and (cities["city"] == city).any() else {}
    fl = [flags(r) for _, r in ws.iterrows()]
    red = [r["ward_id"] for (_, r), f in zip(ws.iterrows(), fl) if f["red"]]
    black = [r["ward_id"] for (_, r), f in zip(ws.iterrows(), fl) if f["black"]]
    tr = transfers[transfers["city"] == city] if transfers is not None and not transfers.empty else pd.DataFrame()
    hc = int(ws["tier"].isin(["High", "Critical"]).sum()) if "tier" in ws else 0
    util = float(_g(c, "utilization", 0.0))
    if en:
        lines = [f"AlertAhead · City summary · {city}",
                 f"Week of {fmt_date(ctx.target)} (+{ctx.horizon} weeks) · stock {ctx.stock_pct}% · crews {ctx.total_crews}", "",
                 f"Expected cases: {n(ws['pred'].sum())} · High/Critical wards: {n(hc)}",
                 f"Saline: {n(ws['saline_alloc'].sum())} of {n(ws['saline_req'].sum())} bags · Kits: {n(ws['kits_alloc'].sum())} of {n(ws['kits_req'].sum())}",
                 f"Beds: {n(_g(c, 'demand', 0))} needed / {n(_g(c, 'capacity', 0))} capacity ({util * 100:.0f}%) · routed {n(_g(c, 'routed', 0))} · unmet {n(_g(c, 'unmet', 0))}",
                 f"Red alert wards: {', '.join(red) if red else 'none'}", f"CODE BLACK wards: {', '.join(black) if black else 'none'}", "Transfers:"]
        lines += [f"  {t.from_ward} -> {t.to_ward}: {t.patients} patient(s), {t.km:.1f} km" for t in tr.itertuples(index=False)] or ["  none"]
    else:
        lines = [f"অ্যালার্টঅ্যাহেড · শহরের সারসংক্ষেপ · {city_name(city, 'bn')}",
                 f"{fmt_date(ctx.target, 'bn')} সপ্তাহ (+{n(ctx.horizon)} সপ্তাহ) · মজুদ {n(ctx.stock_pct)}% · টিম {n(ctx.total_crews)}", "",
                 f"আনুমানিক রোগী: {n(ws['pred'].sum())} · উচ্চ/চরম ওয়ার্ড: {n(hc)}",
                 f"স্যালাইন: {n(ws['saline_req'].sum())}টির মধ্যে {n(ws['saline_alloc'].sum())} · কিট: {n(ws['kits_req'].sum())}টির মধ্যে {n(ws['kits_alloc'].sum())}",
                 f"শয্যা: {n(_g(c, 'demand', 0))} প্রয়োজন / {n(_g(c, 'capacity', 0))} সক্ষমতা ({n(util * 100)}%) · স্থানান্তর {n(_g(c, 'routed', 0))} · বাকি {n(_g(c, 'unmet', 0))}",
                 f"লাল সতর্কতার ওয়ার্ড: {', '.join(red) if red else 'নেই'}", f"কোড ব্ল্যাক ওয়ার্ড: {', '.join(black) if black else 'নেই'}", "স্থানান্তর:"]
        lines += [f"  {t.from_ward} -> {t.to_ward}: {n(t.patients)} জন, {n(t.km, 1)} কিমি" for t in tr.itertuples(index=False)] or ["  নেই"]
    return "\n".join(lines)


def escalation(city: str, wards: pd.DataFrame, cities: pd.DataFrame, ctx: NoticeContext, lang: str = "en") -> str | None:
    """Civil-surgeon escalation - only when a ward is CODE BLACK or the city bed pool is exhausted."""
    if lang == "both":
        a = escalation(city, wards, cities, ctx, "bn")
        return None if a is None else a + "\n\n— English —\n" + escalation(city, wards, cities, ctx, "en")
    ws = _city_rows(wards, city)
    c = cities[cities["city"] == city].iloc[0] if not cities.empty and (cities["city"] == city).any() else {}
    black = [r["ward_id"] for _, r in ws.iterrows() if flags(r)["black"]]
    unmet = int(_g(c, "unmet", 0))
    if not black and not bool(_g(c, "exhausted", unmet > 0)):
        return None
    sal = int(sum(shortfall(r)[0] for _, r in ws.iterrows()))
    n = lambda x: num(x, lang)  # noqa: E731
    if lang == "en":
        return "\n".join([f"ESCALATION · Civil Surgeon · {city}", f"Week of {fmt_date(ctx.target)} (+{ctx.horizon} weeks)", "",
                          f"Patients without a bed after intra-city routing: {n(unmet)}",
                          f"CODE BLACK wards (no bed + saline short): {', '.join(black) if black else 'none'}",
                          f"Saline shortfall in the city pool: {n(sal)} bags", "",
                          f"Request: {n(unmet)} surge beds or transfer to a district/medical-college hospital; emergency saline {n(sal)} bags."])
    return "\n".join([f"জরুরি বার্তা · সিভিল সার্জন · {city_name(city, 'bn')}", f"{fmt_date(ctx.target, 'bn')} সপ্তাহ (+{n(ctx.horizon)} সপ্তাহ)", "",
                      f"শহরের ভেতরে স্থানান্তরের পরও শয্যা পাননি: {n(unmet)} জন",
                      f"কোড ব্ল্যাক ওয়ার্ড (শয্যা নেই + স্যালাইন ঘাটতি): {', '.join(black) if black else 'নেই'}",
                      f"শহরের পুলে স্যালাইন ঘাটতি: {n(sal)} ব্যাগ", "",
                      f"অনুরোধ: {n(unmet)}টি অতিরিক্ত শয্যা বা জেলা/মেডিকেল কলেজ হাসপাতালে স্থানান্তর; জরুরি স্যালাইন {n(sal)} ব্যাগ।"])


# ------------------------------------------------------------------ export bundle (shared by the app ZIP and make_exports.py)
def export_files(wards: pd.DataFrame, cities: pd.DataFrame, transfers: pd.DataFrame | None, ctx: NoticeContext,
                 font_rel_url: str | None = None) -> dict[str, str]:
    """All dispatch files for these wards, as {relative path: text}. Deterministic (sorted, no clock)."""
    font_css = (f'@font-face{{font-family:"BnSubset";src:url("{font_rel_url}") format("woff2")}}' if font_rel_url else "")
    files: dict[str, str] = {}
    for _, w in wards.sort_values("ward_id").iterrows():
        wid = w["ward_id"]
        files[f"sms/{wid}_en.txt"] = sms_en(w, ctx, transfers) + "\n"
        files[f"sms/{wid}_bn.txt"] = sms_bn(w, ctx) + "\n"
        files[f"whatsapp/{wid}.txt"] = whatsapp(w, ctx, transfers, "comm", "both") + "\n"
        files[f"briefs/{wid}_commissioner.html"] = brief_html(w, ctx, transfers, "comm", "both", font_css)
        files[f"briefs/{wid}_health_centre.html"] = brief_html(w, ctx, transfers, "hcm", "both", font_css)
    for city in sorted(wards["city"].unique()):
        slug = city.replace(" ", "_")
        files[f"city_summary/{slug}.txt"] = city_summary(city, wards, cities, transfers, ctx, "both") + "\n"
        esc = escalation(city, wards, cities, ctx, "both")
        if esc:
            files[f"escalation/{slug}_civil_surgeon.txt"] = esc + "\n"
    alloc_cols = [c for c in ["ward_id", "city", "tier", "pred", "saline_req", "saline_alloc", "kits_req", "kits_alloc",
                              "beds_req", "bed_cap", "sent", "received", "unmet", "crews", "prevented"] if c in wards.columns]
    files["allocation.csv"] = wards.sort_values("ward_id")[alloc_cols].to_csv(index=False, lineterminator="\n")
    tr = transfers if transfers is not None else pd.DataFrame(columns=["city", "from_ward", "to_ward", "patients", "km"])
    files["routing.csv"] = tr.to_csv(index=False, lineterminator="\n", float_format="%.2f")
    return files


def zip_bytes(files: dict[str, str]) -> bytes:
    """Store files in a ZIP with a fixed timestamp, so the same scenario gives the same bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):
            info = zipfile.ZipInfo(path, date_time=(2025, 6, 9, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, files[path].encode("utf-8"))
    return buf.getvalue()
