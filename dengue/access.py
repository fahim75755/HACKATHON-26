"""Login helpers for the officer portal: salted password hashes and a brute-force brake. Pure Python, offline.

Passwords are never stored: each one is kept as
    pbkdf2_sha256$<iterations>$<salt hex>$<hash hex>
(PBKDF2-HMAC-SHA256, 16-byte random salt), and checked with hmac.compare_digest.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from collections.abc import Callable

ITERATIONS = 200_000
MIN_PASSWORD = 8
PASSWORD_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789"


def hash_password(password: str, iterations: int = ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_hex, hash_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(digest.hex(), hash_hex)
    except (ValueError, AttributeError):
        return False


# A fixed hash checked when the username does not exist, so a wrong user name takes as long as a wrong password.
DUMMY_HASH = hash_password("not-a-real-account")


def new_password(length: int = 12) -> str:
    """A random password that is easy to read out (no 0/O, 1/l/I)."""
    return "".join(secrets.choice(PASSWORD_ALPHABET) for _ in range(length))


class AttemptLimiter:
    """Brute-force brake shared by every visitor of one server process.

    `max_fails` wrong logins from one client key (the visitor's IP when Streamlit knows it, else one
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
