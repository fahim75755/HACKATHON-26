"""⑥ Commissioner outbox (a bonus on top of R4): access-code sign-in, a per-recipient inbox, SMS-app links,
Mark sent / Acknowledge, and an admin overview with the code directory.

The app never sends anything. A commissioner (or the control room) copies the SMS or taps
"Open in SMS app", which opens the phone's own SMS app with the text filled in, then marks it sent.
No gateway, no API key, no internet: it keeps working with the kill-switch on.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from dengue import access, notify
from dengue import outbox as ob

STATUS = {"queued": ("🟡", "Queued", "অপেক্ষায়"), "sent": ("✅", "Sent", "পাঠানো হয়েছে"), "ack": ("👁", "Acknowledged", "দেখেছেন")}


def _t(is_bn: bool, en: str, bn: str) -> str:
    return bn if is_bn else en


@st.cache_resource
def _shared() -> dict:
    """Status and audit log shared by every visitor of this server process (in memory: a restart clears it)."""
    return {"lock": threading.Lock(), "status": {}, "audit": [], "limiter": access.AttemptLimiter()}


def _keys() -> access.Keys:
    try:
        src = {k: st.secrets[k] for k in st.secrets}
    except Exception:   # no secrets.toml / Cloud secrets -> demo mode
        src = {}
    return access.load_keys(src)


def _client_key() -> str:
    try:
        ip = st.context.ip_address
    except Exception:
        ip = None
    return ip or "global"


def _now() -> str:
    return datetime.now().strftime("%d %b %H:%M:%S")


def _log(state: dict, user: str, action: str, msg_id: str = "") -> None:
    with state["lock"]:
        state["audit"].append({"time": _now(), "user": user, "action": action, "message": msg_id})


def _status(state: dict, msg_id: str) -> str:
    return state["status"].get(msg_id, {}).get("status", "queued")


def _set_status(state: dict, msg_id: str, status: str, user: str) -> None:
    with state["lock"]:
        state["status"][msg_id] = {"status": status, "by": user, "at": _now()}
    _log(state, user, status, msg_id)


def _who(rid: str, directory: pd.DataFrame, is_bn: bool) -> str:
    if rid == access.ADMIN_ID:
        return _t(is_bn, "Control room (admin)", "কন্ট্রোল রুম (অ্যাডমিন)")
    row = directory[directory["recipient_id"] == rid]
    if row.empty:
        return rid
    r = row.iloc[0]
    role = ob.ROLE_NAME[r["role"]][1 if is_bn else 0]
    return f"{role} · {r['ward_id'] or notify.city_name(r['city'], 'bn' if is_bn else 'en')}"


# ------------------------------------------------------------------ sign-in
def _demo_box(box: pd.DataFrame, keys: access.Keys, directory: pd.DataFrame, is_bn: bool) -> None:
    urgent = box[(box["priority"] <= 2) & ~box["cc"]]
    picks = [access.ADMIN_ID] + list(urgent[urgent["role"] == "comm"]["recipient_id"].drop_duplicates().head(2)) \
        + list(urgent[urgent["role"] == "cs"]["recipient_id"].head(1))
    if len(picks) == 1:   # calm scenario: still show one commissioner
        picks += list(directory[directory["role"] == "comm"]["recipient_id"].head(1))
    with st.container(border=True):
        st.markdown("🧪 **" + _t(is_bn, "Demo mode", "ডেমো মোড") + "** — " + _t(
            is_bn, "no secrets are set, so these demo codes work (for judges and testing). Set ACCESS_SECRET and ADMIN_SECRET "
                   "in Streamlit secrets to issue private codes.",
            "কোনো secret সেট নেই, তাই এই ডেমো কোডগুলো কাজ করবে (বিচারক ও পরীক্ষার জন্য)। নিজস্ব কোডের জন্য Streamlit secrets-এ "
            "ACCESS_SECRET ও ADMIN_SECRET দিন।"))
        st.dataframe(pd.DataFrame({_t(is_bn, "Who", "কে"): [_who(r, directory, is_bn) for r in picks],
                                   _t(is_bn, "Recipient ID", "প্রাপক আইডি"): picks,
                                   _t(is_bn, "Demo code", "ডেমো কোড"): [access.code_for(keys, r) for r in picks]}),
                     hide_index=True, width="stretch")


def _sign_in(box: pd.DataFrame, directory: pd.DataFrame, keys: access.Keys, state: dict, is_bn: bool) -> None:
    lim, ck = state["limiter"], _client_key()
    wait = lim.locked_for(ck)
    if wait > 0:
        st.error(_t(is_bn, f"Too many wrong codes. Try again in about {int(wait // 60) + 1} min.",
                    f"অনেকবার ভুল কোড। প্রায় {notify.num(int(wait // 60) + 1, 'bn')} মিনিট পরে আবার চেষ্টা করুন।"), icon="🔒")
    with st.form("ob_login", clear_on_submit=True):
        st.markdown("**" + _t(is_bn, "Sign in with your access code", "আপনার অ্যাক্সেস কোড দিয়ে ঢুকুন") + "**")
        code = st.text_input(_t(is_bn, "Access code", "অ্যাক্সেস কোড"), type="password", placeholder="XXXXX-XXXXX", key="ob_code")
        submitted = st.form_submit_button(_t(is_bn, "Sign in", "ঢুকুন"), disabled=wait > 0, key="ob_submit")
    if submitted and wait <= 0:
        rid = access.who_is(code, keys, directory["recipient_id"])
        if rid is None:
            lim.record_failure(ck)
            time.sleep(1.0)   # slows guessing
            st.error(_t(is_bn, "That code is not valid.", "এই কোডটি সঠিক নয়।"))
        else:
            st.session_state["ob_user"] = rid
            _log(state, rid, "sign-in")
            st.rerun()
    if keys.demo:
        _demo_box(box, keys, directory, is_bn)
        if st.button(_t(is_bn, "Sign in as demo admin", "ডেমো অ্যাডমিন হিসেবে ঢুকুন"), key="ob_demo_admin"):
            st.session_state["ob_user"] = access.ADMIN_ID
            _log(state, access.ADMIN_ID, "sign-in (demo)")
            st.rerun()


# ------------------------------------------------------------------ one message
def _card(m, state: dict, user: str, is_bn: bool, directory: pd.DataFrame) -> None:
    s = _status(state, m.msg_id)
    icon, en, bn = STATUS[s]
    with st.container(border=True):
        where = m.ward_id or notify.city_name(m.city, "bn" if is_bn else "en")
        cc = _t(is_bn, " · copy", " · কপি") if m.cc else ""
        to = f" · {_t(is_bn, 'to', 'প্রাপক')} {_who(m.recipient_id, directory, is_bn)}" if user == access.ADMIN_ID else ""
        st.markdown(f"{ob.ICON[m.level]} **{m.level}**{cc} · {where}{to} · {icon} {_t(is_bn, en, bn)}")
        st.code(m.sms_en, language=None, wrap_lines=True)
        info = notify.sms_info(m.sms_en)
        st.caption(f"SMS · {info['chars']} {_t(is_bn, 'chars', 'অক্ষর')} · {info['parts']} SMS")
        with st.expander(_t(is_bn, "Bangla SMS and WhatsApp text", "বাংলা এসএমএস ও হোয়াটসঅ্যাপ লেখা")):
            if m.sms_bn:
                st.code(m.sms_bn, language=None, wrap_lines=True)
            st.code(m.whatsapp, language=None, wrap_lines=True)
        c1, c2, c3 = st.columns(3)
        c1.link_button("📱 " + _t(is_bn, "Open in SMS app", "এসএমএস অ্যাপে খুলুন"), ob.sms_link(m.sms_en), key=f"ob_sms_{m.msg_id}")
        if c2.button("✅ " + _t(is_bn, "Mark sent", "পাঠানো হয়েছে"), key=f"ob_sent_{m.msg_id}", disabled=s != "queued"):
            _set_status(state, m.msg_id, "sent", user)
            st.rerun()
        if c3.button("👁 " + _t(is_bn, "Acknowledge", "দেখেছি"), key=f"ob_ack_{m.msg_id}", disabled=s == "ack"):
            _set_status(state, m.msg_id, "ack", user)
            st.rerun()


# ------------------------------------------------------------------ views
def _inbox(box: pd.DataFrame, rid: str, state: dict, is_bn: bool, directory: pd.DataFrame) -> None:
    mine = box[box["recipient_id"] == rid]
    if mine.empty:
        st.info(_t(is_bn, "No messages for you in this scenario. If your ward is outside the sidebar city filter, choose All cities.",
                   "এই পরিস্থিতিতে আপনার কোনো বার্তা নেই। আপনার ওয়ার্ড সাইডবারের শহর-ফিল্টারের বাইরে হলে All cities বেছে নিন।"))
        return
    urgent = int((mine["priority"] <= 2).sum())
    st.caption(_t(is_bn, f"{len(mine)} message(s), {urgent} urgent. Copy the SMS or open it in your SMS app, then mark it sent.",
                  f"{notify.num(len(mine), 'bn')}টি বার্তা, {notify.num(urgent, 'bn')}টি জরুরি। এসএমএস কপি করুন বা এসএমএস অ্যাপে খুলুন, তারপর 'পাঠানো হয়েছে' চাপুন।"))
    for m in mine.itertuples(index=False):
        _card(m, state, rid, is_bn, directory)


def _admin(box: pd.DataFrame, directory: pd.DataFrame, keys: access.Keys, state: dict, is_bn: bool) -> None:
    status = box["msg_id"].map(lambda i: _status(state, i))
    k = st.columns(4)
    k[0].metric(_t(is_bn, "Messages", "বার্তা"), len(box))
    k[1].metric("⚫ " + _t(is_bn, "CODE BLACK / escalation", "কোড ব্ল্যাক / জরুরি"), int((box["priority"] == 1).sum()))
    k[2].metric("🔴 " + _t(is_bn, "Red alerts", "লাল সতর্কতা"), int((box["level"] == "RED").sum()))
    k[3].metric(_t(is_bn, "Sent / acknowledged", "পাঠানো / দেখেছেন"), f"{int((status == 'sent').sum())} / {int((status == 'ack').sum())}")

    f1, f2, f3 = st.columns([1.4, 1, 1.4])
    levels = f1.multiselect(_t(is_bn, "Alert level", "সতর্কতার স্তর"), list(ob.PRIORITY), default=["CODE BLACK", "ESCALATION", "RED"], key="ob_f_level")
    city = f2.selectbox(_t(is_bn, "City", "শহর"), ["All"] + sorted(box["city"].unique()), key="ob_f_city")
    rid = f3.selectbox(_t(is_bn, "Recipient", "প্রাপক"), ["All"] + sorted(box["recipient_id"].unique()), key="ob_f_rid",
                       format_func=lambda r: r if r == "All" else f"{r} — {_who(r, directory, is_bn)}")
    view = box[box["level"].isin(levels)] if levels else box
    if city != "All":
        view = view[view["city"] == city]
    if rid != "All":
        view = view[view["recipient_id"] == rid]
    st.dataframe(pd.DataFrame({
        "": view["level"].map(ob.ICON), _t(is_bn, "Level", "স্তর"): view["level"],
        _t(is_bn, "Recipient", "প্রাপক"): view["recipient_id"], _t(is_bn, "Role", "পদ"): view["role"].map(lambda r: ob.ROLE_NAME[r][1 if is_bn else 0]),
        _t(is_bn, "Ward / city", "ওয়ার্ড / শহর"): [w or c for w, c in zip(view["ward_id"], view["city"])],
        _t(is_bn, "Copy", "কপি"): view["cc"].map({True: "cc", False: ""}),
        _t(is_bn, "Status", "অবস্থা"): view["msg_id"].map(lambda i: " ".join(STATUS[_status(state, i)][:2]))}),
        hide_index=True, width="stretch")
    shown = view.head(12)
    if len(view) > len(shown):
        st.caption(_t(is_bn, f"Showing the {len(shown)} most urgent of {len(view)} messages — pick a recipient to see theirs.",
                      f"{notify.num(len(view), 'bn')}টির মধ্যে সবচেয়ে জরুরি {notify.num(len(shown), 'bn')}টি দেখানো হচ্ছে — নির্দিষ্ট প্রাপক বেছে নিন।"))
    for m in shown.itertuples(index=False):
        _card(m, state, access.ADMIN_ID, is_bn, directory)

    st.download_button("⬇ " + _t(is_bn, "Download outbox (CSV)", "আউটবক্স ডাউনলোড (CSV)"), ob.outbox_files(box)["outbox/outbox.csv"].encode("utf-8"),
                       "outbox.csv", "text/csv", key="ob_dl")
    with st.expander(_t(is_bn, "Access-code directory (admin only)", "অ্যাক্সেস কোডের তালিকা (শুধু অ্যাডমিন)")):
        who = st.selectbox(_t(is_bn, "Recipient", "প্রাপক"), list(directory["recipient_id"]), key="ob_code_for",
                           format_func=lambda r: f"{r} — {_who(r, directory, is_bn)}")
        st.code(access.code_for(keys, who), language=None)
        st.caption(_t(is_bn,
                      f"Version {keys.versions.get(who, 1)}. Hand this code to the officer in person. To revoke it, raise "
                      f"`{who} = {keys.versions.get(who, 1) + 1}` under [code_versions] in the Streamlit secrets; to revoke every code, change ACCESS_SECRET.",
                      f"সংস্করণ {notify.num(keys.versions.get(who, 1), 'bn')}। কোডটি কর্মকর্তাকে সরাসরি দিন। বাতিল করতে Streamlit secrets-এ [code_versions]-এর নিচে "
                      f"`{who} = {keys.versions.get(who, 1) + 1}` দিন; সব কোড বাতিল করতে ACCESS_SECRET বদলান।"))
    with st.expander(_t(is_bn, "Audit log", "অডিট লগ")):
        log = pd.DataFrame(state["audit"][-100:], columns=["time", "user", "action", "message"])
        st.dataframe(log.iloc[::-1], hide_index=True, width="stretch")
        st.caption(_t(is_bn, "Kept in memory on this server; a restart clears it.", "এই সার্ভারের মেমোরিতে থাকে; রিস্টার্টে মুছে যায়।"))


def render_outbox(wards: pd.DataFrame, transfers: pd.DataFrame, cities: pd.DataFrame, ctx: notify.NoticeContext, is_bn: bool) -> None:
    st.caption(_t(is_bn,
                  "Who gets which notice: ⚫ CODE BLACK / escalation → civil surgeon (+ copy to the ward commissioner), 🔴 red alert and every "
                  "routine notice → ward commissioner and health-centre manager. The app sends nothing: sign in, copy the SMS or open it in "
                  "your phone's SMS app, then mark it sent. Works offline.",
                  "কে কোন বার্তা পাবেন: ⚫ কোড ব্ল্যাক / জরুরি → সিভিল সার্জন (+ ওয়ার্ড কমিশনারকে কপি), 🔴 লাল সতর্কতা ও সাধারণ বার্তা → ওয়ার্ড কমিশনার ও "
                  "স্বাস্থ্যকেন্দ্র ব্যবস্থাপক। অ্যাপ নিজে কিছু পাঠায় না: কোড দিয়ে ঢুকুন, এসএমএস কপি করুন বা ফোনের এসএমএস অ্যাপে খুলুন, তারপর "
                  "'পাঠানো হয়েছে' চাপুন। অফলাইনেও চলে।"))
    box = ob.build_outbox(wards, cities, transfers, ctx)
    directory = ob.recipients(wards)
    keys, state = _keys(), _shared()
    user = st.session_state.get("ob_user")
    if not user:
        _sign_in(box, directory, keys, state, is_bn)
        return
    c1, c2 = st.columns([4, 1])
    c1.markdown("👤 " + _t(is_bn, "Signed in as", "প্রবেশ করেছেন") + f" **{_who(user, directory, is_bn)}** (`{user}`)"
                + (" · 🧪 " + _t(is_bn, "demo mode", "ডেমো মোড") if keys.demo else ""))
    if c2.button(_t(is_bn, "Sign out", "বের হন"), key="ob_logout"):
        _log(state, user, "sign-out")
        st.session_state.pop("ob_user", None)
        st.rerun()
    if user == access.ADMIN_ID:
        _admin(box, directory, keys, state, is_bn)
    else:
        _inbox(box, user, state, is_bn, directory)
