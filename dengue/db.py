"""Small SQLite store for the officer portal: officer accounts, medical-centre phone numbers, SMS status, audit log.

* One local file (`data/alertahead.db`, git-ignored; override with ALERTAHEAD_DB). sqlite3 ships with Python,
  so it needs no server and no internet: it keeps working with the kill-switch on.
* Passwords are stored only as salted PBKDF2 hashes (dengue/access.py).
* First start: if the officers table is empty, an admin account is created. With ADMIN_USER / ADMIN_PASSWORD in
  the Streamlit secrets (or env ALERTAHEAD_ADMIN_USER / ALERTAHEAD_ADMIN_PASSWORD) that is the admin and no demo
  accounts are made; without them the store runs in demo mode with the demo accounts in DEMO_ACCOUNTS.
  A database first started in demo mode and later opened with ADMIN_PASSWORD gets that admin added and its
  demo accounts disabled. The page never lists any account or password.
* On Streamlit Community Cloud the app folder is temporary: the file survives while the app runs but a reboot or
  a new deploy starts it again from the seed. Real deployment would point ALERTAHEAD_DB at a persistent disk.
"""
from __future__ import annotations

import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

from dengue import access

ROOT = Path(__file__).resolve().parent.parent
ROLES = ("admin", "city", "ward")
USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,31}$")
PHONE_RE = re.compile(r"^\+?[0-9][0-9 -]{5,19}$")
DEMO_ACCOUNTS = [   # (username, password, full name, role, ward_id, city) - public on purpose, demo mode only
    ("admin", "admin-demo-2026", "Control room (demo)", "admin", "", ""),
    ("dhaka.north", "city-demo-2026", "Dhaka North health officer (demo)", "city", "", "Dhaka North"),
    ("dncc10", "ward-demo-2026", "DNCC Ward 10 commissioner (demo)", "ward", "DNCC_Ward_10", "Dhaka North"),
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS officers (
    username   TEXT PRIMARY KEY,
    full_name  TEXT NOT NULL,
    role       TEXT NOT NULL CHECK (role IN ('admin', 'city', 'ward')),
    ward_id    TEXT NOT NULL DEFAULT '',
    city       TEXT NOT NULL DEFAULT '',
    pw_hash    TEXT NOT NULL,
    active     INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS contacts (
    contact_id TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    phone      TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS sms_status (
    msg_id  TEXT PRIMARY KEY,
    status  TEXT NOT NULL CHECK (status IN ('sent', 'ack')),
    by_user TEXT NOT NULL,
    at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    at     TEXT NOT NULL,
    user   TEXT NOT NULL,
    action TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT ''
);
"""


def default_path() -> Path:
    return Path(os.environ.get("ALERTAHEAD_DB") or ROOT / "data" / "alertahead.db")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class StoreError(ValueError):
    """A request the admin panel should show as a plain message (bad input, duplicate, last admin...)."""


class Store:
    def __init__(self, path: str | Path, admin_password: str | None = None, admin_user: str = "admin",
                 iterations: int = access.ITERATIONS):
        self.path = Path(path)
        self.iterations = iterations
        self.demo = not admin_password
        admin_user = (admin_user or "admin").strip().lower()
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)
            if c.execute("SELECT COUNT(*) FROM officers").fetchone()[0] == 0:
                seed = DEMO_ACCOUNTS if self.demo else [(admin_user, admin_password, "Control room", "admin", "", "")]
                for user, pw, name, role, ward, city in seed:
                    c.execute("INSERT INTO officers VALUES (?,?,?,?,?,?,1,?)",
                              (user, name, role, ward, city, access.hash_password(pw, iterations), _now()))
                c.execute("INSERT INTO audit (at, user, action, detail) VALUES (?,?,?,?)",
                          (_now(), "system", "seed", "demo accounts" if self.demo else f"admin '{admin_user}'"))
            elif not self.demo:
                self._adopt_private_admin(c, admin_user, admin_password, iterations)

    @staticmethod
    def _adopt_private_admin(c: sqlite3.Connection, admin_user: str, admin_password: str, iterations: int) -> None:
        """A database first started in demo mode, now opened with ADMIN_PASSWORD set: add the configured admin
        if it is missing, and disable (not delete) every demo account that still has its public password.
        An existing admin account keeps its password, so a password changed inside the app is never reset."""
        if c.execute("SELECT 1 FROM officers WHERE username = ?", (admin_user,)).fetchone() is None:
            c.execute("INSERT INTO officers VALUES (?,?,?,?,?,?,1,?)",
                      (admin_user, "Control room", "admin", "", "", access.hash_password(admin_password, iterations), _now()))
            c.execute("INSERT INTO audit (at, user, action, detail) VALUES (?,?,?,?)", (_now(), "system", "seed", f"admin '{admin_user}'"))
        for user, pw, *_ in DEMO_ACCOUNTS:
            r = c.execute("SELECT pw_hash, active FROM officers WHERE username = ?", (user,)).fetchone()
            if user != admin_user and r and r["active"] and access.verify_password(pw, r["pw_hash"]):
                c.execute("UPDATE officers SET active = 0 WHERE username = ?", (user,))
                c.execute("INSERT INTO audit (at, user, action, detail) VALUES (?,?,?,?)",
                          (_now(), "system", "disable demo account", user))

    @contextmanager
    def _conn(self):
        """One short-lived connection per call: commits on success and always closes (no locked files on Windows)."""
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def _write(self, sql: str, args: tuple = ()) -> int:
        with self._lock, self._conn() as c:
            return c.execute(sql, args).rowcount

    # ------------------------------------------------------------------ audit
    def log(self, user: str, action: str, detail: str = "") -> None:
        self._write("INSERT INTO audit (at, user, action, detail) VALUES (?,?,?,?)", (_now(), user, action, detail))

    def audit(self, limit: int = 200) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT at, user, action, detail FROM audit ORDER BY id DESC LIMIT ?", c, params=(limit,))

    # ------------------------------------------------------------------ officers
    def officer(self, username: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT username, full_name, role, ward_id, city, active FROM officers WHERE username = ?",
                          (username,)).fetchone()
        return dict(r) if r else None

    def officers(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT username, full_name, role, ward_id, city, active, created_at FROM officers "
                                     "ORDER BY role, city, ward_id, username", c)

    def authenticate(self, username: str, password: str) -> dict | None:
        """The active officer for these credentials, or None. Same work whether or not the user exists."""
        username = (username or "").strip().lower()
        with self._conn() as c:
            r = c.execute("SELECT * FROM officers WHERE username = ?", (username,)).fetchone()
        ok = access.verify_password(password or "", r["pw_hash"] if r else access.DUMMY_HASH)
        if not (r and ok and r["active"]):
            return None
        return {k: r[k] for k in ("username", "full_name", "role", "ward_id", "city")}

    @staticmethod
    def _check(username: str, full_name: str, role: str, ward_id: str, city: str, wards: dict | None) -> None:
        if not USERNAME_RE.match(username):
            raise StoreError("User name: 3-32 characters, lower-case letters, digits, dot, dash or underscore.")
        if not full_name.strip():
            raise StoreError("Full name is required.")
        if role not in ROLES:
            raise StoreError(f"Role must be one of {', '.join(ROLES)}.")
        if role == "ward" and not ward_id:
            raise StoreError("A ward officer needs a ward.")
        if role == "city" and not city:
            raise StoreError("A city officer needs a city.")
        if wards is not None and role == "ward" and wards.get(ward_id) is None:
            raise StoreError(f"Unknown ward {ward_id}.")
        if wards is not None and role == "city" and city not in set(wards.values()):
            raise StoreError(f"Unknown city {city}.")

    def add_officer(self, username: str, full_name: str, role: str, password: str, ward_id: str = "", city: str = "",
                    by: str = "admin", wards: dict | None = None) -> None:
        username = username.strip().lower()
        if role == "ward" and wards:
            city = wards.get(ward_id, city)
        self._check(username, full_name, role, ward_id, city, wards)
        if len(password) < access.MIN_PASSWORD:
            raise StoreError(f"Password must be at least {access.MIN_PASSWORD} characters.")
        ward_id, city = (ward_id, city) if role == "ward" else ("", city) if role == "city" else ("", "")
        try:
            self._write("INSERT INTO officers VALUES (?,?,?,?,?,?,1,?)",
                        (username, full_name.strip(), role, ward_id, city, access.hash_password(password, self.iterations), _now()))
        except sqlite3.IntegrityError:
            raise StoreError(f"User name '{username}' already exists.") from None
        self.log(by, "add officer", f"{username} ({role} {ward_id or city})".strip())

    def _active_admins(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM officers WHERE role = 'admin' AND active = 1").fetchone()[0]

    def update_officer(self, username: str, full_name: str, role: str, ward_id: str = "", city: str = "",
                       active: bool = True, by: str = "admin", wards: dict | None = None) -> None:
        cur = self.officer(username)
        if cur is None:
            raise StoreError(f"No officer '{username}'.")
        if role == "ward" and wards:
            city = wards.get(ward_id, city)
        self._check(username, full_name, role, ward_id, city, wards)
        was_admin = cur["role"] == "admin" and cur["active"]
        if was_admin and (role != "admin" or not active) and self._active_admins() <= 1:
            raise StoreError("This is the last active admin; add another admin first.")
        ward_id, city = (ward_id, city) if role == "ward" else ("", city) if role == "city" else ("", "")
        self._write("UPDATE officers SET full_name=?, role=?, ward_id=?, city=?, active=? WHERE username=?",
                    (full_name.strip(), role, ward_id, city, int(bool(active)), username))
        self.log(by, "update officer", f"{username}: {role} {ward_id or city} {'active' if active else 'disabled'}".strip())

    def set_password(self, username: str, password: str, by: str) -> None:
        if len(password) < access.MIN_PASSWORD:
            raise StoreError(f"Password must be at least {access.MIN_PASSWORD} characters.")
        if self._write("UPDATE officers SET pw_hash=? WHERE username=?",
                       (access.hash_password(password, self.iterations), username)) == 0:
            raise StoreError(f"No officer '{username}'.")
        self.log(by, "set password", username)

    # ------------------------------------------------------------------ medical-centre / civil-surgeon contacts
    def seed_contacts(self, rows: list[tuple[str, str]]) -> None:
        """Add (contact_id, name) rows that are missing; never overwrites a saved phone number."""
        with self._lock, self._conn() as c:
            c.executemany("INSERT OR IGNORE INTO contacts (contact_id, name, phone) VALUES (?,?,'')", rows)

    def contacts(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT contact_id, name, phone FROM contacts ORDER BY contact_id", c)

    def phones(self) -> dict[str, str]:
        df = self.contacts()
        return {r.contact_id: r.phone for r in df.itertuples(index=False) if r.phone}

    def save_contacts(self, df: pd.DataFrame, by: str) -> int:
        """Save edited phone numbers. Rejects anything that does not look like a phone number."""
        known = set(self.contacts()["contact_id"])
        changes = []
        for r in df.itertuples(index=False):
            phone = str(r.phone or "").strip()
            if r.contact_id not in known:
                raise StoreError(f"Unknown contact {r.contact_id}.")
            if phone and not PHONE_RE.match(phone):
                raise StoreError(f"{r.contact_id}: '{phone}' is not a phone number.")
            changes.append((phone, r.contact_id))
        with self._lock, self._conn() as c:
            before = dict(c.execute("SELECT contact_id, phone FROM contacts").fetchall())
            changed = [(p, cid) for p, cid in changes if before.get(cid, "") != p]
            c.executemany("UPDATE contacts SET phone=? WHERE contact_id=?", changed)
        if changed:
            self.log(by, "save contacts", ", ".join(cid for _, cid in changed)[:500])
        return len(changed)

    # ------------------------------------------------------------------ SMS status
    def statuses(self) -> dict[str, dict]:
        with self._conn() as c:
            return {r["msg_id"]: dict(r) for r in c.execute("SELECT * FROM sms_status")}

    def set_status(self, msg_id: str, status: str, user: str) -> None:
        if status not in ("sent", "ack"):
            raise StoreError(f"Bad status {status}.")
        self._write("INSERT INTO sms_status VALUES (?,?,?,?) ON CONFLICT(msg_id) DO UPDATE SET status=excluded.status, "
                    "by_user=excluded.by_user, at=excluded.at", (msg_id, status, user, _now()))
        self.log(user, status, msg_id)

    def sms_log(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT msg_id, status, by_user, at FROM sms_status ORDER BY at DESC", c)
