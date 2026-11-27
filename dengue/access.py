"""Access codes for the commissioner outbox (a bonus on top of R4; fully offline).

A code is *derived*, never stored:

    code = HMAC-SHA256(secret, f"{recipient_id}:{version}")  -> 10 letters, shown as XXXXX-XXXXX

* `ACCESS_SECRET` signs commissioner / health-centre / civil-surgeon codes, `ADMIN_SECRET` signs the
  admin code, so leaking one does not leak the other.
* Both live in `.streamlit/secrets.toml` (git-ignored) locally and in Streamlit Cloud -> App settings ->
  Secrets online. Nothing secret is in the repo, and nothing has to be written to disk, so codes
  survive Streamlit Cloud restarts.
* Revoke one code: raise its number under `[code_versions]` in the secrets (e.g. `COMM-DNCC-05 = 2`).
  Revoke all codes: change `ACCESS_SECRET`.
* No secrets configured (tests, a fresh laptop, judging): fixed demo secrets are used and the UI says
  "Demo mode" and prints the demo codes, so nobody is ever locked out of a feature.

Pure Python (hmac, hashlib): no network, deterministic.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"   # no 0/O, 1/I/L: easy to read out over the phone
CODE_LEN = 10
ADMIN_ID = "ADMIN"
DEMO_ACCESS_SECRET = "alertahead-demo-access-secret"
DEMO_ADMIN_SECRET = "alertahead-demo-admin-secret"


@dataclass(frozen=True)
class Keys:
    access: bytes
    admin: bytes
    versions: dict = field(default_factory=dict)   # recipient_id -> int (missing = 1)
    demo: bool = True


def load_keys(source: Mapping | None = None, env: Mapping | None = None) -> Keys:
    """Read the secrets from `source` (e.g. st.secrets) or the environment; fall back to demo keys."""
    src = dict(source or {})
    env = os.environ if env is None else env
    access = src.get("ACCESS_SECRET") or env.get("ALERTAHEAD_ACCESS_SECRET")
    admin = src.get("ADMIN_SECRET") or env.get("ALERTAHEAD_ADMIN_SECRET")
    versions = {str(k): int(v) for k, v in dict(src.get("code_versions") or {}).items()}
    return Keys(access=str(access or DEMO_ACCESS_SECRET).encode(), admin=str(admin or DEMO_ADMIN_SECRET).encode(),
                versions=versions, demo=not (access and admin))


def make_code(secret: bytes, recipient_id: str, version: int = 1) -> str:
    """The code for one recipient: 10 characters from ALPHABET (~8 x 10^14 possibilities)."""
    mac = hmac.new(secret, f"{recipient_id}:{int(version)}".encode(), hashlib.sha256).digest()
    n, out = int.from_bytes(mac[:8], "big"), []
    for _ in range(CODE_LEN):
        n, r = divmod(n, len(ALPHABET))
        out.append(ALPHABET[r])
    s = "".join(out)
    return f"{s[:5]}-{s[5:]}"


def normalise(code: str) -> str:
    """Accept lower case, spaces and a missing dash: ' 7kq4m 9txrb ' -> '7KQ4M-9TXRB'."""
    s = "".join(ch for ch in str(code).upper() if ch.isalnum())
    return f"{s[:5]}-{s[5:]}" if len(s) == CODE_LEN else s


def code_for(keys: Keys, recipient_id: str) -> str:
    secret = keys.admin if recipient_id == ADMIN_ID else keys.access
    return make_code(secret, recipient_id, keys.versions.get(recipient_id, 1))


def who_is(code: str, keys: Keys, recipient_ids: Iterable[str]) -> str | None:
    """Return the recipient this code belongs to, or None. Compares every candidate in constant time."""
    c = normalise(code)
    if len(c) != CODE_LEN + 1:
        return None
    found = None
    for rid in [ADMIN_ID, *sorted(set(recipient_ids))]:
        if hmac.compare_digest(code_for(keys, rid), c):
            found = rid
    return found


class AttemptLimiter:
    """Brute-force brake shared by every visitor of one server process.

    `max_fails` wrong codes from one client key (the visitor's IP when Streamlit knows it, else one
    global key) inside `window_s` seconds lock that key for `lock_s` seconds.
    """

    def __init__(self, max_fails: int = 5, window_s: float = 300.0, lock_s: float = 300.0,
                 clock: Callable[[], float] = time.monotonic):
        self.max_fails, self.window_s, self.lock_s, self.clock = max_fails, window_s, lock_s, clock
        self._lock = threading.Lock()
        self._fails: dict[str, list[float]] = {}
        self._locked_until: dict[str, float] = {}

    def locked_for(self, key: str = "global") -> float:
        with self._lock:
            return max(0.0, self._locked_until.get(key, 0.0) - self.clock())

    def record_failure(self, key: str = "global") -> None:
        with self._lock:
            now = self.clock()
            fails = [t for t in self._fails.get(key, []) if now - t < self.window_s] + [now]
            if len(fails) >= self.max_fails:
                self._locked_until[key] = now + self.lock_s
                fails = []
            self._fails[key] = fails
