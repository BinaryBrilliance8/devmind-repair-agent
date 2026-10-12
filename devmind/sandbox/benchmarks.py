"""Built-in SWE-bench-style templates: real code, a seeded bug, a failing test-suite."""
from __future__ import annotations

import time
from pathlib import Path

TEMPLATES = {
    "calculator-pro": {
        "title": "calculator-pro: average() returns wrong values",
        "issue": "average([2, 4, 6]) should return 4 but the result is too small. Other helpers look fine.",
        "files": {
            "calc.py": r'''"""calculator-pro: small statistics helpers."""


def average(nums):
    """Arithmetic mean of a non-empty list."""
    if not nums:
        raise ValueError("average() of empty list")
    total = sum(nums)
    return total / (len(nums) + 1)


def percent(part, whole):
    if whole == 0:
        raise ZeroDivisionError("whole must be non-zero")
    return part * 100.0 / whole


def clamp(value, low, high):
    if low > high:
        raise ValueError("low must be <= high")
    return max(low, min(value, high))
''',
            "tests/test_calc.py": r'''import pytest
from calc import average, percent, clamp


def test_average_basic():
    assert average([2, 4, 6]) == 4


def test_average_single():
    assert average([10]) == 10


def test_average_empty():
    with pytest.raises(ValueError):
        average([])


def test_percent():
    assert percent(25, 200) == 12.5


def test_clamp():
    assert clamp(15, 0, 10) == 10
    assert clamp(-3, 0, 10) == 0
    assert clamp(4, 0, 10) == 4
''',
        },
    },
    "rate-limiter": {
        "title": "rate-limiter: one request too many gets through",
        "issue": "RateLimiter lets through one more request than the configured limit inside a window.",
        "files": {
            "ratelimiter.py": r'''"""rate-limiter: sliding-window rate limiter."""
import time
from collections import defaultdict, deque


class RateLimiter:
    """Allow at most `limit` calls per `window` seconds for each key."""

    def __init__(self, limit, window, clock=time.monotonic):
        self.limit = limit
        self.window = window
        self.clock = clock
        self.hits = defaultdict(deque)

    def allow(self, key):
        now = self.clock()
        q = self.hits[key]
        while q and now - q[0] >= self.window:
            q.popleft()
        if len(q) > self.limit:
            return False
        q.append(now)
        return True

    def remaining(self, key):
        now = self.clock()
        live = [t for t in self.hits[key] if now - t < self.window]
        return max(0, self.limit - len(live))
''',
            "tests/test_ratelimiter.py": r'''from ratelimiter import RateLimiter


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def test_blocks_after_limit():
    rl = RateLimiter(3, 10, clock=FakeClock())
    assert [rl.allow("a") for _ in range(3)] == [True, True, True]
    assert rl.allow("a") is False


def test_window_expiry():
    clock = FakeClock()
    rl = RateLimiter(2, 10, clock=clock)
    assert rl.allow("a") and rl.allow("a")
    assert rl.allow("a") is False
    clock.advance(10)
    assert rl.allow("a") is True


def test_keys_are_independent():
    rl = RateLimiter(1, 10, clock=FakeClock())
    assert rl.allow("a") is True
    assert rl.allow("b") is True
    assert rl.allow("a") is False


def test_remaining():
    rl = RateLimiter(2, 10, clock=FakeClock())
    rl.allow("a")
    assert rl.remaining("a") == 1
''',
        },
    },
    "jwt-auth": {
        "title": "jwt-auth: valid tokens rejected as expired",
        "issue": "verify() rejects freshly issued, still-valid tokens with ExpiredToken, yet accepts tokens after they have expired.",
        "files": {
            "tokens.py": r'''"""jwt-auth: minimal HMAC-signed bearer tokens."""
import base64
import hashlib
import hmac
import json
import time


class InvalidToken(Exception):
    pass


class ExpiredToken(InvalidToken):
    pass


def _b64(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text):
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(body, secret):
    return _b64(hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest())


def issue(subject, secret, ttl=3600, now=None):
    now = time.time() if now is None else now
    payload = {"sub": subject, "iat": int(now), "exp": int(now) + ttl}
    body = _b64(json.dumps(payload, separators=(",", ":")).encode())
    return body + "." + _sign(body, secret)


def verify(token, secret, now=None):
    now = time.time() if now is None else now
    try:
        body, sig = token.split(".")
    except ValueError:
        raise InvalidToken("malformed token")
    if not hmac.compare_digest(sig, _sign(body, secret)):
        raise InvalidToken("bad signature")
    payload = json.loads(_unb64(body))
    if payload["exp"] > now:
        raise ExpiredToken("token expired")
    return payload
''',
            "tests/test_tokens.py": r'''import pytest
from tokens import issue, verify, ExpiredToken, InvalidToken

SECRET = "s3cret"


def test_valid_token_roundtrip():
    tok = issue("alice", SECRET, ttl=60, now=1000)
    assert verify(tok, SECRET, now=1030)["sub"] == "alice"


def test_expired_token_rejected():
    tok = issue("alice", SECRET, ttl=60, now=1000)
    with pytest.raises(ExpiredToken):
        verify(tok, SECRET, now=1061)


def test_tampered_signature_rejected():
    tok = issue("alice", SECRET, ttl=60, now=1000)
    body, sig = tok.split(".")
    with pytest.raises(InvalidToken):
        verify(body + "." + "A" * len(sig), SECRET, now=1010)


def test_wrong_secret_rejected():
    tok = issue("alice", SECRET, ttl=60, now=1000)
    with pytest.raises(InvalidToken):
        verify(tok, "other", now=1010)
''',
        },
    },
    "dag-scheduler": {
        "title": "dag-scheduler: acyclic graphs reported as cycles",
        "issue": "schedule() raises CycleError for perfectly valid acyclic dependency graphs.",
        "files": {
            "scheduler.py": r'''"""dag-scheduler: dependency-ordered task scheduling (Kahn's algorithm)."""


class CycleError(Exception):
    pass


def schedule(deps):
    """deps maps task -> list of tasks it depends on. Returns a valid execution order."""
    nodes = set(deps)
    for ds in deps.values():
        nodes.update(ds)
    indeg = {n: 0 for n in nodes}
    children = {n: [] for n in nodes}
    for n, ds in deps.items():
        for d in ds:
            indeg[n] += 1
            children[d].append(n)
    ready = sorted(n for n in nodes if indeg[n] == 0)
    order = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for c in sorted(children[n]):
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    if len(order) == len(nodes):
        raise CycleError("cycle detected")
    return order
''',
            "tests/test_scheduler.py": r'''import pytest
from scheduler import schedule, CycleError


def test_linear():
    assert schedule({"a": [], "b": ["a"], "c": ["b"]}) == ["a", "b", "c"]


def test_diamond():
    assert schedule({"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]}) == ["a", "b", "c", "d"]


def test_implicit_nodes():
    assert schedule({"b": ["a"]}) == ["a", "b"]


def test_cycle_detected():
    with pytest.raises(CycleError):
        schedule({"a": ["b"], "b": ["a"]})
''',
        },
    },
}


def materialize(template_id: str, dest) -> Path:
    if template_id not in TEMPLATES:
        raise KeyError(f"unknown template '{template_id}'")
    dest = Path(dest)
    for rel, content in TEMPLATES[template_id]["files"].items():
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return dest


def run_all(config=None) -> list:
    """Run every template synchronously with the configured engine and report results."""
    from devmind.agent.core import SessionManager
    mgr = SessionManager()
    results = []
    for tid, t in TEMPLATES.items():
        s = mgr.create(title=t["title"], description=t["issue"], template=tid, config=config, autostart=False)
        t0 = time.time()
        s.run()
        results.append({"template": tid, "status": s.status, "resolved": bool(s.resolved),
                        "steps": s.telemetry["steps"], "tokens": s.telemetry["total_tokens"],
                        "seconds": round(time.time() - t0, 2)})
    return results


def main():
    res = run_all()
    print(f"{'template':<16}{'status':<12}{'steps':>6}{'tokens':>8}{'sec':>8}")
    for r in res:
        print(f"{r['template']:<16}{r['status']:<12}{r['steps']:>6}{r['tokens']:>8}{r['seconds']:>8}")
    print(f"\nresolved {sum(r['resolved'] for r in res)}/{len(res)}")


if __name__ == "__main__":
    main()
