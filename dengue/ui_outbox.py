"""⑥ Officer portal (a bonus on top of R4): officer login, the alert SMS queue, and an admin panel.

* Login: user name + password, checked against the small SQLite store (dengue/db.py; salted PBKDF2 hashes).
  5 wrong logins lock that client for 5 minutes. Accounts are never listed on the page: the admin comes from
  ADMIN_USER / ADMIN_PASSWORD in the Streamlit secrets, and the admin adds every other officer.
* SMS queue: which wards need an SMS to their medical centre (and which cities need a civil-surgeon escalation).
  The officer opens each SMS in their phone's own SMS app (number + text filled in), sends it, and marks it
  sent / centre confirmed. The app itself sends nothing: no gateway, no API key, no internet.
* Admin panel: officer accounts (add, edit, disable, reset password), medical-centre phone numbers,
  SMS log and audit log.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import streamlit as st

from dengue import access, db, notify
from dengue import outbox as ob

STATUS = {"queued": ("🟡", "Queued", "অপেক্ষায়"), "sent": ("✅", "Sent", "পাঠানো হয়েছে"),
          "ack": ("📩", "Centre confirmed", "সেন্টার নিশ্চিত করেছে")}
CITIES = ["Dhaka North", "Dhaka South", "Chattogram", "Gazipur", "Narayanganj"]


def _t(is_bn: bool, en: str, bn: str) -> str:
    return bn if is_bn else en


def _secret(name: str) -> str | None:
    try:
        v = st.secrets.get(name)
    except Exception:   # no secrets.toml -> demo mode
        v = None
    return v or os.environ.get(f"ALERTAHEAD_{name}")


@st.cache_resource
def _store(path: str, admin_password: str | None, admin_user: str) -> db.Store:
    return db.Store(path, admin_password=admin_password, admin_user=admin_user)


@st.cache_resource
def _limiter() -> access.AttemptLimiter:
    return access.AttemptLimiter()


def get_store() -> db.Store:
    return _store(str(db.default_path()), _secret("ADMIN_PASSWORD"), _secret("ADMIN_USER") or "admin")


@st.cache_data
def _all_wards() -> dict:
    """ward_id -> city for all 60 wards (the admin assigns officers to any ward, whatever the sidebar filter)."""
    from dengue.config import PREDICTIONS_PATH
    try:
        df = pd.read_parquet(PREDICTIONS_PATH, columns=["ward_id", "city"]).drop_duplicates("ward_id")
        return dict(zip(df["ward_id"], df["city"]))
    except Exception:
        return {}


def _client_key() -> str:
    try:
        ip = st.context.ip_address
    except Exception:
        ip = None
    return ip or "global"


def _who(o: dict, is_bn: bool) -> str:
    role = ob.ROLE_NAME[o["role"]][1 if is_bn else 0]
    area = o.get("ward_id") or (notify.city_name(o["city"], "bn" if is_bn else "en") if o.get("city") else "")
    return f"{o.get('full_name', '')} · {role}" + (f" · {area}" if area else "")


# ------------------------------------------------------------------ login
def _login(store: db.Store, is_bn: bool) -> None:
    lim, ck = _limiter(), _client_key()
    wait = lim.locked_for(ck)
    if wait > 0:
        st.error(_t(is_bn, f"Too many wrong logins. Try again in about {int(wait // 60) + 1} min.",
                    f"অনেকবার ভুল লগইন। প্রায় {notify.num(int(wait // 60) + 1, 'bn')} মিনিট পরে আবার চেষ্টা করুন।"), icon="🔒")
    with st.form("op_login", clear_on_submit=False):
        st.markdown("🔐 **" + _t(is_bn, "Officer login", "কর্মকর্তার লগইন") + "**")
        user = st.text_input(_t(is_bn, "User name", "ইউজার নেম"), key="op_user")
        pw = st.text_input(_t(is_bn, "Password", "পাসওয়ার্ড"), type="password", key="op_pw")
        submitted = st.form_submit_button(_t(is_bn, "Log in", "লগইন"), disabled=wait > 0, key="op_submit", type="primary")
    if submitted and wait <= 0:
        o = store.authenticate(user, pw)
        if o is None:
            lim.record_failure(ck)
            store.log((user or "?").strip().lower()[:32], "login failed")
            time.sleep(1.0)   # slows guessing
            st.error(_t(is_bn, "Wrong user name or password, or the account is disabled.",
                        "ইউজার নেম বা পাসওয়ার্ড ভুল, অথবা অ্যাকাউন্ট বন্ধ।"))
        else:
            st.session_state["op_username"] = o["username"]
            st.session_state.pop("op_pw", None)
            store.log(o["username"], "login")
            st.rerun()


def _change_my_password(store: db.Store, me: dict, is_bn: bool) -> None:
    with st.expander("🔑 " + _t(is_bn, "Change my password", "আমার পাসওয়ার্ড বদলান")):
        with st.form("op_pw_change", clear_on_submit=True):
            old = st.text_input(_t(is_bn, "Current password", "বর্তমান পাসওয়ার্ড"), type="password", key="op_pw_old")
            new = st.text_input(_t(is_bn, "New password (min 8)", "নতুন পাসওয়ার্ড (কমপক্ষে ৮)"), type="password", key="op_pw_new")
            if st.form_submit_button(_t(is_bn, "Change password", "পাসওয়ার্ড বদলান"), key="op_pw_go"):
                if store.authenticate(me["username"], old) is None:
                    st.error(_t(is_bn, "Current password is wrong.", "বর্তমান পাসওয়ার্ড ভুল।"))
                else:
                    try:
                        store.set_password(me["username"], new, by=me["username"])
                        st.success(_t(is_bn, "Password changed.", "পাসওয়ার্ড বদলানো হয়েছে।"))
                    except db.StoreError as e:
                        st.error(str(e))


# ------------------------------------------------------------------ SMS queue
def _card(m, store: db.Store, statuses: dict, me: dict, phones: dict, is_bn: bool) -> None:
    s = statuses.get(m.msg_id, {}).get("status", "queued")
    icon, en, bn = STATUS[s]
    phone = phones.get(m.to_id, "")
    to = f"{ob.RECIPIENT_NAME[m.to_role][1 if is_bn else 0]} · {m.ward_id or notify.city_name(m.city, 'bn' if is_bn else 'en')}"
    with st.container(border=True):
        st.markdown(f"{ob.ICON[m.level]} **{m.level}** · {m.ward_id or notify.city_name(m.city, 'bn' if is_bn else 'en')} → "
                    f"**{to}** (`{m.to_id}`{' · ' + phone if phone else ''}) · {icon} {_t(is_bn, en, bn)}")
        st.code(m.sms_en, language=None, wrap_lines=True)
        info = notify.sms_info(m.sms_en)
        st.caption(f"SMS · {info['chars']} {_t(is_bn, 'chars', 'অক্ষর')} · {info['parts']} SMS"
                   + ("" if phone else " · " + _t(is_bn, "no phone saved: the SMS app opens without a number",
                                                   "ফোন নম্বর সেভ নেই: এসএমএস অ্যাপ নম্বর ছাড়া খুলবে")))
        with st.expander(_t(is_bn, "Bangla SMS and WhatsApp text", "বাংলা এসএমএস ও হোয়াটসঅ্যাপ লেখা")):
            if m.sms_bn:
                st.code(m.sms_bn, language=None, wrap_lines=True)
            st.code(m.whatsapp, language=None, wrap_lines=True)
        c1, c2, c3 = st.columns(3)
        c1.link_button("📱 " + _t(is_bn, "Open in SMS app", "এসএমএস অ্যাপে খুলুন"), ob.sms_link(m.sms_en, phone), key=f"op_sms_{m.msg_id}",
                       width="stretch")
        if c2.button("✅ " + _t(is_bn, "Mark sent", "পাঠানো হয়েছে"), key=f"op_sent_{m.msg_id}", disabled=s != "queued",
                      type="primary", width="stretch"):
            store.set_status(m.msg_id, "sent", me["username"])
            st.rerun()
        if c3.button("📩 " + _t(is_bn, "Centre confirmed", "সেন্টার নিশ্চিত করেছে"), key=f"op_ack_{m.msg_id}", disabled=s == "ack",
                      width="stretch"):
            store.set_status(m.msg_id, "ack", me["username"])
            st.rerun()


def _queue_view(queue: pd.DataFrame, store: db.Store, me: dict, is_bn: bool) -> None:
    mine = ob.visible(queue, me)
    statuses, phones = store.statuses(), store.phones()
    status = mine["msg_id"].map(lambda i: statuses.get(i, {}).get("status", "queued"))
    k = st.columns(4)
    k[0].metric(_t(is_bn, "SMS in queue", "কিউতে এসএমএস"), len(mine))
    k[1].metric("⚫ " + _t(is_bn, "CODE BLACK / escalation", "কোড ব্ল্যাক / জরুরি"), int((mine["priority"] == 1).sum()))
    k[2].metric("🔴 " + _t(is_bn, "Red alerts", "লাল সতর্কতা"), int((mine["level"] == "RED").sum()))
    k[3].metric(_t(is_bn, "Still to send", "পাঠানো বাকি"), int((status == "queued").sum()))
    if mine.empty:
        st.success(_t(is_bn, "No ward in your area needs an SMS in this scenario. If your ward is outside the sidebar city filter, choose All cities.",
                      "এই পরিস্থিতিতে আপনার এলাকার কোনো ওয়ার্ডে এসএমএস লাগবে না। আপনার ওয়ার্ড সাইডবারের শহর-ফিল্টারের বাইরে হলে All cities বেছে নিন।"))
        return
    f1, f2, f3 = st.columns([1.4, 1, 1])
    levels = f1.multiselect(_t(is_bn, "Alert level", "সতর্কতার স্তর"), list(ob.PRIORITY), default=["CODE BLACK", "ESCALATION", "RED"],
                            key="op_f_level", help=_t(is_bn, "Add 🟠 SHORTFALL to see every short ward (at low stock that is most wards).",
                                                      "প্রতিটি ঘাটতির ওয়ার্ড দেখতে 🟠 SHORTFALL যোগ করুন (কম মজুদে প্রায় সব ওয়ার্ড)।"))
    city = f2.selectbox(_t(is_bn, "City", "শহর"), ["All"] + sorted(mine["city"].unique()), key="op_f_city", filter_mode=None)
    only_open = f3.toggle(_t(is_bn, "Only not yet sent", "শুধু পাঠানো বাকি"), key="op_f_open")
    view = mine[mine["level"].isin(levels)] if levels else mine
    if city != "All":
        view = view[view["city"] == city]
    if only_open:
        view = view[view["msg_id"].map(lambda i: i not in statuses)]
    st.dataframe(pd.DataFrame({
        "": view["level"].map(ob.ICON), _t(is_bn, "Level", "স্তর"): view["level"],
        _t(is_bn, "Ward / city", "ওয়ার্ড / শহর"): [w or c for w, c in zip(view["ward_id"], view["city"])],
        _t(is_bn, "Send to", "যাকে পাঠাবেন"): view["to_id"] + view["to_id"].map(lambda t: f" · {phones[t]}" if t in phones else ""),
        _t(is_bn, "Status", "অবস্থা"): view["msg_id"].map(lambda i: " ".join(STATUS[statuses.get(i, {}).get("status", "queued")][:2]))}),
        hide_index=True, width="stretch")
    shown = view.head(12)
    if len(view) > len(shown):
        st.caption(_t(is_bn, f"Showing the {len(shown)} most urgent of {len(view)} — filter by city or level to see the rest.",
                      f"{notify.num(len(view), 'bn')}টির মধ্যে সবচেয়ে জরুরি {notify.num(len(shown), 'bn')}টি — বাকিগুলো দেখতে শহর বা স্তর বাছুন।"))
    for m in shown.itertuples(index=False):
        _card(m, store, statuses, me, phones, is_bn)


# ------------------------------------------------------------------ admin panel
def _officers_admin(store: db.Store, me: dict, wards_map: dict, is_bn: bool) -> None:
    df = store.officers()
    st.dataframe(df.assign(active=df["active"].map({1: "✅", 0: "⛔"})), hide_index=True, width="stretch")
    ward_ids = sorted(wards_map)
    a, b = st.columns(2)
    with a.form("op_add", clear_on_submit=True):
        st.markdown("➕ **" + _t(is_bn, "Add officer", "কর্মকর্তা যোগ করুন") + "**")
        username = st.text_input(_t(is_bn, "User name (e.g. dncc05)", "ইউজার নেম (যেমন dncc05)"), key="op_add_user")
        name = st.text_input(_t(is_bn, "Full name", "পুরো নাম"), key="op_add_name")
        role = st.selectbox(_t(is_bn, "Role", "ভূমিকা"), list(db.ROLES), index=2, key="op_add_role",
                            format_func=lambda r: ob.ROLE_NAME[r][1 if is_bn else 0], filter_mode=None)
        ward = st.selectbox(_t(is_bn, "Ward (ward officer)", "ওয়ার্ড (ওয়ার্ড কর্মকর্তা)"), [""] + ward_ids, key="op_add_ward", filter_mode=None)
        city = st.selectbox(_t(is_bn, "City (city officer)", "শহর (শহর কর্মকর্তা)"), [""] + CITIES, key="op_add_city", filter_mode=None)
        pw = st.text_input(_t(is_bn, "Password (empty = generate one)", "পাসওয়ার্ড (খালি রাখলে তৈরি হবে)"), type="password", key="op_add_pw")
        if st.form_submit_button(_t(is_bn, "Add", "যোগ করুন"), key="op_add_go", type="primary"):
            pw = pw or access.new_password()
            try:
                store.add_officer(username, name, role, pw, ward_id=ward if role == "ward" else "", city=city,
                                  by=me["username"], wards=wards_map)
                st.success(_t(is_bn, f"Added {username.strip().lower()}. Password (shown once): {pw}",
                              f"{username.strip().lower()} যোগ হয়েছে। পাসওয়ার্ড (একবারই দেখানো হবে): {pw}"))
            except db.StoreError as e:
                st.error(str(e))
    with b.container(border=True):
        st.markdown("✏️ **" + _t(is_bn, "Edit, disable or reset", "সম্পাদনা, বন্ধ বা রিসেট") + "**")
        who = st.selectbox(_t(is_bn, "Officer", "কর্মকর্তা"), list(df["username"]), key="op_edit_who", filter_mode=None)
        cur = store.officer(who)
        if cur:
            with st.form("op_edit", clear_on_submit=False):
                name = st.text_input(_t(is_bn, "Full name", "পুরো নাম"), value=cur["full_name"], key=f"op_edit_name_{who}")
                role = st.selectbox(_t(is_bn, "Role", "ভূমিকা"), list(db.ROLES), index=list(db.ROLES).index(cur["role"]),
                                    key=f"op_edit_role_{who}", format_func=lambda r: ob.ROLE_NAME[r][1 if is_bn else 0], filter_mode=None)
                wopts = [""] + ward_ids
                ward = st.selectbox(_t(is_bn, "Ward", "ওয়ার্ড"), wopts, index=wopts.index(cur["ward_id"]) if cur["ward_id"] in wopts else 0,
                                    key=f"op_edit_ward_{who}", filter_mode=None)
                copts = [""] + CITIES
                city = st.selectbox(_t(is_bn, "City", "শহর"), copts, index=copts.index(cur["city"]) if cur["city"] in copts else 0,
                                    key=f"op_edit_city_{who}", filter_mode=None)
                active = st.checkbox(_t(is_bn, "Active (can log in)", "সক্রিয় (লগইন করতে পারবেন)"), value=bool(cur["active"]),
                                     key=f"op_edit_active_{who}")
                if st.form_submit_button(_t(is_bn, "Save", "সংরক্ষণ"), key="op_edit_go"):
                    try:
                        store.update_officer(who, name, role, ward_id=ward, city=city, active=active, by=me["username"], wards=wards_map)
                        st.success(_t(is_bn, "Saved.", "সংরক্ষিত।"))
                    except db.StoreError as e:
                        st.error(str(e))
            if st.button("🔄 " + _t(is_bn, "Reset password", "পাসওয়ার্ড রিসেট"), key="op_reset_go"):
                pw = access.new_password()
                store.set_password(who, pw, by=me["username"])
                st.success(_t(is_bn, f"New password for {who} (shown once): {pw}", f"{who}-এর নতুন পাসওয়ার্ড (একবারই দেখানো হবে): {pw}"))


def _contacts_admin(store: db.Store, me: dict, is_bn: bool) -> None:
    st.caption(_t(is_bn, "Phone numbers the SMS app opens with. Stored only in the local database (never in the repo).",
                  "এসএমএস অ্যাপ যে নম্বর নিয়ে খোলে। শুধু লোকাল ডেটাবেসে থাকে (রিপোতে কখনো নয়)।"))
    df = store.contacts()
    edited = st.data_editor(df, key="op_contacts", hide_index=True, width="stretch", disabled=["contact_id", "name"],
                            column_config={"phone": st.column_config.TextColumn(_t(is_bn, "Phone", "ফোন"), help="+8801XXXXXXXXX")})
    if st.button("💾 " + _t(is_bn, "Save phone numbers", "ফোন নম্বর সংরক্ষণ"), key="op_contacts_go"):
        try:
            n = store.save_contacts(edited, by=me["username"])
            st.success(_t(is_bn, f"Saved {n} change(s).", f"{notify.num(n, 'bn')}টি পরিবর্তন সংরক্ষিত।"))
        except db.StoreError as e:
            st.error(str(e))


def _admin_panel(queue: pd.DataFrame, store: db.Store, me: dict, wards_map: dict, is_bn: bool) -> None:
    st.markdown("### 🛠 " + _t(is_bn, "Admin panel", "অ্যাডমিন প্যানেল"))
    t1, t2, t3, t4 = st.tabs([_t(is_bn, "Officers", "কর্মকর্তা"), _t(is_bn, "Medical-centre phones", "মেডিকেল সেন্টারের ফোন"),
                              _t(is_bn, "SMS log", "এসএমএস লগ"), _t(is_bn, "Audit log", "অডিট লগ")])
    with t1:
        _officers_admin(store, me, wards_map, is_bn)
    with t2:
        _contacts_admin(store, me, is_bn)
    with t3:
        log = store.sms_log()
        st.dataframe(log, hide_index=True, width="stretch")
        st.download_button(_t(is_bn, "Download SMS queue (CSV)", "এসএমএস কিউ ডাউনলোড (CSV)"),
                           ob.queue_files(queue)["outbox/queue.csv"].encode("utf-8"), "sms_queue.csv", "text/csv", key="op_dl",
                           icon=":material/download:")
    with t4:
        st.dataframe(store.audit(), hide_index=True, width="stretch")
    st.caption(_t(is_bn, f"Database: {store.path.name} (SQLite, local, offline). On Streamlit Cloud it resets when the app restarts.",
                  f"ডেটাবেস: {store.path.name} (SQLite, লোকাল, অফলাইন)। Streamlit Cloud-এ অ্যাপ রিস্টার্টে নতুন করে শুরু হয়।"))


# ------------------------------------------------------------------ entry point
def render_outbox(wards: pd.DataFrame, transfers: pd.DataFrame, cities: pd.DataFrame, ctx: notify.NoticeContext, is_bn: bool) -> None:
    st.caption(_t(is_bn,
                  "Officers log in here. The SMS queue lists every ward that needs an SMS to its medical centre (⚫ CODE BLACK, 🔴 red, "
                  "🟠 shortfall) and every city whose civil surgeon needs an escalation. Open each SMS in your phone's SMS app, send it, "
                  "mark it sent. The app sends nothing itself, so it works offline.",
                  "কর্মকর্তারা এখানে লগইন করেন। এসএমএস কিউতে আছে কোন ওয়ার্ডের মেডিকেল সেন্টারে এসএমএস লাগবে (⚫ কোড ব্ল্যাক, 🔴 লাল, 🟠 ঘাটতি) "
                  "আর কোন শহরের সিভিল সার্জনকে জরুরি বার্তা দিতে হবে। প্রতিটি এসএমএস ফোনের এসএমএস অ্যাপে খুলে পাঠান, তারপর 'পাঠানো হয়েছে' চাপুন। "
                  "অ্যাপ নিজে কিছু পাঠায় না, তাই অফলাইনেও চলে।"))
    store = get_store()
    store.seed_contacts(ob.contacts_seed(wards))
    queue = ob.build_queue(wards, cities, transfers, ctx)
    wards_map = {**dict(zip(wards["ward_id"], wards["city"])), **_all_wards()}
    me = store.officer(st.session_state.get("op_username", "")) if st.session_state.get("op_username") else None
    if me is None or not me["active"]:
        st.session_state.pop("op_username", None)
        _login(store, is_bn)
        return
    c1, c2 = st.columns([4, 1])
    c1.markdown("👤 " + _t(is_bn, "Logged in as", "লগইন করেছেন") + f" **{_who(me, is_bn)}** (`{me['username']}`)"
                + (" · 🧪 " + _t(is_bn, "demo mode", "ডেমো মোড") if store.demo else ""))
    if c2.button(_t(is_bn, "Log out", "লগআউট"), key="op_logout"):
        store.log(me["username"], "logout")
        st.session_state.pop("op_username", None)
        st.rerun()
    _change_my_password(store, me, is_bn)
    _queue_view(queue, store, me, is_bn)
    if me["role"] == "admin":
        _admin_panel(queue, store, me, wards_map, is_bn)
