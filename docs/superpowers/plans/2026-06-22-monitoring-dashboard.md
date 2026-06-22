# Monitoring Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local auto-refreshing web dashboard that pulls live state (Metaculus, GitHub Actions, OpenRouter) to monitor how the bot is doing.

**Architecture:** Python stdlib `http.server` bound to `127.0.0.1`. A background thread refreshes an in-memory `state` dict every ~60s by calling collector functions (each wrapped so one failing source doesn't break the page). The browser page polls `/api/status` (JSON) every ~10s and re-renders. Collectors split pure parsers (unit-tested against fixtures) from network fetchers. Secrets never reach the browser.

**Tech Stack:** Python 3.12, stdlib `http.server`/`urllib`/`subprocess`/`threading`, `httpx` (already a dep) for API calls, `gh` CLI for GitHub, pytest + monkeypatch.

---

## Conventions (match these)
- Tests: flat `tests/test_<module>.py`; `asyncio_mode=auto` (bare `async def`, no marker). Stub network via `monkeypatch`. Run one: `.venv/bin/python -m pytest tests/test_dashboard_collectors.py -v`.
- Run full suite: `.venv/bin/python -m pytest -q` (must stay green; baseline 509 passed / 1 skipped).
- Entrypoints: module docstring, `load_dotenv()` before bot imports, `logging.basicConfig(INFO)`, `argparse`, `def main(argv=None) -> int`, `sys.exit(main())`.
- Secrets from `.env`: `OPENROUTER_API_KEY`, `METACULUS_TOKEN`. mopsus user id = `303527`. Tournaments: `33022` and slug `minibench`.

## File structure
- Create `bot/dashboard/__init__.py` — package marker.
- Create `bot/dashboard/collectors.py` — pure parsers + network fetchers.
- Create `bot/dashboard/state.py` — `DashboardState` (TTL cache + background refresh).
- Create `bot/dashboard/server.py` — `route()` pure function + `ThreadingHTTPServer` handler + `serve()`.
- Create `bot/dashboard/page.py` — `PAGE_HTML` template string.
- Create `scripts/dashboard.py` — entrypoint.
- Create `deploy/com.aaron.metaculus-bot-dashboard.plist` — optional launchd KeepAlive.
- Create `tests/test_dashboard_collectors.py`, `tests/test_dashboard_server.py`.

---

### Task 1: Package marker + OpenRouter budget parser (pure)

**Files:**
- Create: `bot/dashboard/__init__.py`
- Create: `bot/dashboard/collectors.py`
- Test: `tests/test_dashboard_collectors.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dashboard_collectors.py
from bot.dashboard import collectors


AUTH_KEY_OK = {
    "data": {
        "label": "sk-or-v1-d4f...4ca",
        "limit": 100,
        "limit_reset": "weekly",
        "limit_remaining": 98.05203175,
        "usage": 142.79757155,
        "usage_daily": 1.94796825,
        "usage_weekly": 1.94796825,
        "usage_monthly": 142.79757155,
        "creator_user_id": "user_secret",
    }
}


def test_parse_openrouter_budget_ok():
    out = collectors.parse_openrouter_budget(AUTH_KEY_OK)
    assert out["available"] is True
    assert out["limit"] == 100
    assert out["limit_reset"] == "weekly"
    assert out["remaining"] == pytestapprox(98.05203175)
    assert out["window_used"] == pytestapprox(1.94796825)  # paired to limit_reset=weekly
    assert out["today"] == pytestapprox(1.94796825)
    assert out["month"] == pytestapprox(142.79757155)
    assert out["lifetime"] == pytestapprox(142.79757155)
    assert out["pct_used"] == pytestapprox(1.948, abs=0.01)
    # secrets must NOT pass through
    assert "label" not in out
    assert "creator_user_id" not in out


def test_parse_openrouter_budget_unavailable():
    assert collectors.parse_openrouter_budget({"error": {"code": 401}})["available"] is False
    assert collectors.parse_openrouter_budget(None)["available"] is False
```

Add at top of file: `from pytest import approx as pytestapprox`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dashboard_collectors.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bot.dashboard'`.

- [ ] **Step 3: Write minimal implementation**

```python
# bot/dashboard/__init__.py
"""Live monitoring dashboard for the forecasting bot."""
```

```python
# bot/dashboard/collectors.py
"""Collectors for the live dashboard.

Pure parsers (``parse_*``) transform already-fetched JSON and are unit-tested
against fixtures. Network fetchers (``fetch_*``) do I/O and return the parser
output, or an ``{"available": False, ...}`` sentinel on any failure so one dead
source never breaks the page. Collectors return ONLY derived numbers — never
secrets (API keys, account ids).
"""
from __future__ import annotations

from typing import Any


def parse_openrouter_budget(payload: Any) -> dict:
    """Derive the cost panel from a GET /api/v1/auth/key response body."""
    data = (payload or {}).get("data") if isinstance(payload, dict) else None
    if not data:
        return {"available": False}
    reset = data.get("limit_reset")
    window_used = data.get(f"usage_{reset}") if reset else None
    limit = data.get("limit")
    remaining = data.get("limit_remaining")
    pct_used = None
    if isinstance(limit, (int, float)) and limit and isinstance(remaining, (int, float)):
        pct_used = (limit - remaining) / limit * 100.0
    return {
        "available": True,
        "limit": limit,
        "limit_reset": reset,
        "remaining": remaining,
        "window_used": window_used,
        "today": data.get("usage_daily"),
        "week": data.get("usage_weekly"),
        "month": data.get("usage_monthly"),
        "lifetime": data.get("usage"),
        "pct_used": pct_used,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dashboard_collectors.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add bot/dashboard/__init__.py bot/dashboard/collectors.py tests/test_dashboard_collectors.py
git commit -m "feat(dashboard): OpenRouter budget parser"
```

---

### Task 2: OpenRouter budget fetcher (network)

**Files:**
- Modify: `bot/dashboard/collectors.py`
- Test: `tests/test_dashboard_collectors.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dashboard_collectors.py
class _FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
    def json(self):
        return self._payload


def test_fetch_openrouter_budget_ok(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        assert url == "https://openrouter.ai/api/v1/auth/key"
        assert headers["Authorization"] == "Bearer KEY123"
        return _FakeResp(200, AUTH_KEY_OK)
    monkeypatch.setattr(collectors.httpx, "get", fake_get)
    out = collectors.fetch_openrouter_budget("KEY123")
    assert out["available"] is True and out["limit"] == 100


def test_fetch_openrouter_budget_handles_401(monkeypatch):
    monkeypatch.setattr(collectors.httpx, "get",
                        lambda *a, **k: _FakeResp(401, {"error": {"code": 401}}))
    assert collectors.fetch_openrouter_budget("bad")["available"] is False


def test_fetch_openrouter_budget_handles_exception(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(collectors.httpx, "get", boom)
    out = collectors.fetch_openrouter_budget("KEY123")
    assert out["available"] is False and "error" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dashboard_collectors.py -k openrouter_budget -v`
Expected: FAIL — `AttributeError: module 'bot.dashboard.collectors' has no attribute 'httpx'` / `fetch_openrouter_budget`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to imports at top of bot/dashboard/collectors.py
import httpx

# add function
def fetch_openrouter_budget(api_key: str, timeout: float = 8.0) -> dict:
    """GET the OpenRouter key usage; never raise — return a sentinel on failure."""
    if not api_key:
        return {"available": False, "error": "no api key"}
    try:
        resp = httpx.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return {"available": False, "error": f"http {resp.status_code}"}
        return parse_openrouter_budget(resp.json())
    except Exception as exc:  # noqa: BLE001 - one dead source must not break the page
        return {"available": False, "error": str(exc)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dashboard_collectors.py -k openrouter -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bot/dashboard/collectors.py tests/test_dashboard_collectors.py
git commit -m "feat(dashboard): OpenRouter budget fetcher with failure sentinel"
```

---

### Task 3: Own-forecast extractor (pure, all question types)

**Files:**
- Modify: `bot/dashboard/collectors.py`
- Test: `tests/test_dashboard_collectors.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dashboard_collectors.py
BINARY_POST = {
    "id": 43928, "status": "resolved",
    "question": {
        "id": 43954, "type": "binary", "status": "resolved", "resolution": "yes",
        "scaling": {"range_min": None, "range_max": None, "zero_point": None},
        "options": None,
        "my_forecasts": {
            "latest": {"start_time": 1781860108.943803,
                       "forecast_values": [0.1782, 0.8217],
                       "centers": None, "interval_lower_bounds": None,
                       "interval_upper_bounds": None},
            "score_data": {"peer_score": 47.17, "baseline_score": 68.30, "coverage": 0.95},
        },
    },
}
NUMERIC_POST = {
    "id": 44123, "status": "open",
    "question": {
        "id": 44145, "type": "numeric", "status": "open", "resolution": None,
        "scaling": {"range_min": 70000, "range_max": 110000, "zero_point": None},
        "options": None,
        "my_forecasts": {"latest": {"start_time": 1781000000.0,
                                    "forecast_values": [0.0] * 201,
                                    "centers": [0.6119061363500163],
                                    "interval_lower_bounds": [0.47264666340257455],
                                    "interval_upper_bounds": [0.7379228153543191]},
                         "score_data": None},
    },
}
MC_POST = {
    "id": 44119, "status": "open",
    "question": {
        "id": 44200, "type": "multiple_choice", "status": "open", "resolution": None,
        "scaling": {"range_min": None, "range_max": None, "zero_point": None},
        "options": ["Zero or One", "Two or Three", "Four or more"],
        "my_forecasts": {"latest": {"start_time": 1781000000.0,
                                    "forecast_values": [0.3278, 0.4911, 0.1811],
                                    "centers": None, "interval_lower_bounds": None,
                                    "interval_upper_bounds": None},
                         "score_data": None},
    },
}


def test_extract_own_forecast_binary():
    out = collectors.extract_own_forecast(BINARY_POST)
    assert out["type"] == "binary"
    assert out["summary"] == pytestapprox(0.8217)  # P(yes)
    assert out["resolution"] == "yes"
    assert out["start_time"] == 1781860108.943803
    assert out["score"]["peer_score"] == pytestapprox(47.17)


def test_extract_own_forecast_numeric_unscaled():
    out = collectors.extract_own_forecast(NUMERIC_POST)
    assert out["type"] == "numeric"
    # median unscaled into [70000, 110000]
    assert out["median"] == pytestapprox(94476, abs=1)
    assert out["q1"] == pytestapprox(88906, abs=1)
    assert out["q3"] == pytestapprox(99517, abs=1)


def test_extract_own_forecast_multiple_choice():
    out = collectors.extract_own_forecast(MC_POST)
    assert out["type"] == "multiple_choice"
    assert out["top_option"] == "Two or Three"
    assert dict(out["options"])["Two or Three"] == pytestapprox(0.4911)


def test_extract_own_forecast_no_forecast():
    out = collectors.extract_own_forecast({"id": 1, "question": {"id": 2, "type": "binary",
                                          "my_forecasts": {"latest": None}}})
    assert out["summary"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dashboard_collectors.py -k extract_own_forecast -v`
Expected: FAIL — no attribute `extract_own_forecast`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to bot/dashboard/collectors.py
def _unscale(scaled: float, scaling: dict) -> float:
    """Map a scaled [0,1] value to real units (linear; log if zero_point set)."""
    lo = scaling.get("range_min")
    hi = scaling.get("range_max")
    zero = scaling.get("zero_point")
    if lo is None or hi is None:
        return scaled
    if zero is None:
        return lo + scaled * (hi - lo)
    # log scaling: value = lo + (hi-lo) * ((deriv**scaled) - 1) / (deriv - 1)
    deriv = (hi - zero) / (lo - zero)
    return lo + (hi - lo) * ((deriv ** scaled) - 1.0) / (deriv - 1.0)


def extract_own_forecast(post: dict) -> dict:
    """Read mopsus's own latest forecast + score from a post JSON (all types)."""
    q = post.get("question") or {}
    qtype = q.get("type")
    mf = q.get("my_forecasts") or {}
    latest = mf.get("latest") or {}
    out = {
        "post_id": post.get("id"),
        "question_id": q.get("id"),
        "type": qtype,
        "status": post.get("status"),
        "resolution": q.get("resolution"),
        "start_time": latest.get("start_time"),
        "score": mf.get("score_data"),
        "summary": None,
    }
    fv = latest.get("forecast_values")
    if not fv:
        return out
    scaling = q.get("scaling") or {}
    if qtype == "binary":
        out["summary"] = fv[1]
    elif qtype in ("numeric", "date"):
        centers = latest.get("centers") or []
        lows = latest.get("interval_lower_bounds") or []
        highs = latest.get("interval_upper_bounds") or []
        if centers:
            out["median"] = _unscale(centers[0], scaling)
            out["summary"] = out["median"]
        if lows:
            out["q1"] = _unscale(lows[0], scaling)
        if highs:
            out["q3"] = _unscale(highs[0], scaling)
    elif qtype == "multiple_choice":
        options = q.get("options") or []
        pairs = list(zip(options, fv))
        out["options"] = pairs
        if pairs:
            top = max(pairs, key=lambda p: p[1])
            out["top_option"] = top[0]
            out["summary"] = top[0]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dashboard_collectors.py -k extract_own_forecast -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add bot/dashboard/collectors.py tests/test_dashboard_collectors.py
git commit -m "feat(dashboard): own-forecast extractor with numeric/date unscaling"
```

---

### Task 4: Coverage parser (pure)

**Files:**
- Modify: `bot/dashboard/collectors.py`
- Test: `tests/test_dashboard_collectors.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dashboard_collectors.py
def test_parse_coverage_computes_gap():
    open_posts = [
        {"id": 1, "title": "Q1", "question": {"id": 11, "type": "binary"}},
        {"id": 2, "title": "Q2", "question": {"id": 22, "type": "numeric"}},
        {"id": 3, "title": "Notebook", "question": None, "notebook": {}},  # excluded
    ]
    forecasted_ids = {1}
    out = collectors.parse_coverage(open_posts, forecasted_ids)
    assert out["open_count"] == 2          # notebook excluded
    assert out["forecasted_count"] == 1
    assert out["gap_count"] == 1
    assert out["gap"] == [{"post_id": 2, "title": "Q2", "type": "numeric"}]


def test_parse_coverage_no_gap():
    open_posts = [{"id": 1, "title": "Q1", "question": {"id": 11, "type": "binary"}}]
    out = collectors.parse_coverage(open_posts, {1})
    assert out["gap_count"] == 0 and out["gap"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dashboard_collectors.py -k parse_coverage -v`
Expected: FAIL — no attribute `parse_coverage`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to bot/dashboard/collectors.py
def _is_question_post(post: dict) -> bool:
    return post.get("question") is not None


def parse_coverage(open_posts: list[dict], forecasted_ids: set) -> dict:
    """open questions minus the ones mopsus has forecast = the coverage gap."""
    questions = [p for p in open_posts if _is_question_post(p)]
    gap = [
        {"post_id": p["id"], "title": p.get("title"),
         "type": (p.get("question") or {}).get("type")}
        for p in questions if p["id"] not in forecasted_ids
    ]
    return {
        "open_count": len(questions),
        "forecasted_count": sum(1 for p in questions if p["id"] in forecasted_ids),
        "gap_count": len(gap),
        "gap": gap,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dashboard_collectors.py -k parse_coverage -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bot/dashboard/collectors.py tests/test_dashboard_collectors.py
git commit -m "feat(dashboard): coverage gap parser (excludes notebook posts)"
```

---

### Task 5: Metaculus fetchers (paginated, network)

**Files:**
- Modify: `bot/dashboard/collectors.py`
- Test: `tests/test_dashboard_collectors.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dashboard_collectors.py
def test_fetch_posts_paginates(monkeypatch):
    # two pages then empty
    pages = [
        _FakeResp(200, {"results": [{"id": 1, "title": "a", "question": {"id": 11, "type": "binary"}}], "next": "x"}),
        _FakeResp(200, {"results": [{"id": 2, "title": "b", "question": {"id": 22, "type": "binary"}}], "next": "y"}),
        _FakeResp(200, {"results": [], "next": None}),
    ]
    calls = []
    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(params["offset"])
        return pages[len(calls) - 1]
    monkeypatch.setattr(collectors.httpx, "get", fake_get)
    posts = collectors.fetch_posts("TOK", tournament="33022", forecaster_id=None)
    assert [p["id"] for p in posts] == [1, 2]
    assert calls == [0, 100, 200]  # paginated by limit=100 until empty


def test_fetch_metaculus_coverage_builds_gap(monkeypatch):
    open_posts = [{"id": 1, "title": "a", "question": {"id": 11, "type": "binary"}},
                  {"id": 2, "title": "b", "question": {"id": 22, "type": "binary"}}]
    forecast_posts = [{"id": 1, "title": "a", "question": {"id": 11, "type": "binary"}}]

    def fake_fetch_posts(token, tournament, forecaster_id, **kw):
        return forecast_posts if forecaster_id else open_posts
    monkeypatch.setattr(collectors, "fetch_posts", fake_fetch_posts)
    out = collectors.fetch_metaculus_coverage("TOK", ["33022"])
    assert out["available"] is True
    assert out["tournaments"]["33022"]["gap_count"] == 1


def test_fetch_recent_resolved_maps_extractor(monkeypatch):
    # lists resolved forecaster posts (ids), then fetches detail + extracts
    def fake_fetch_posts(token, tournament, forecaster_id, statuses="open", **kw):
        assert statuses == "resolved" and forecaster_id == 303527
        return [{"id": 43928}] if tournament == "33022" else []
    def fake_fetch_post_detail(token, post_id, **kw):
        return BINARY_POST  # the resolved binary fixture from Task 3
    monkeypatch.setattr(collectors, "fetch_posts", fake_fetch_posts)
    monkeypatch.setattr(collectors, "fetch_post_detail", fake_fetch_post_detail)
    out = collectors.fetch_recent_resolved("TOK", ["33022", "minibench"], limit=10)
    assert out["available"] is True
    assert out["resolved"][0]["resolution"] == "yes"
    assert out["resolved"][0]["summary"] == pytestapprox(0.8217)  # via extract_own_forecast
    assert out["resolved"][0]["score"]["peer_score"] == pytestapprox(47.17)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dashboard_collectors.py -k "paginates or metaculus_coverage" -v`
Expected: FAIL — no attribute `fetch_posts`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to bot/dashboard/collectors.py
_METACULUS_POSTS = "https://www.metaculus.com/api/posts/"


def fetch_posts(token: str, tournament: str, forecaster_id: int | None,
                statuses: str = "open", page_size: int = 100,
                max_pages: int = 50, timeout: float = 15.0) -> list[dict]:
    """Paginate /api/posts/ for one tournament (count is always null → follow offset)."""
    results: list[dict] = []
    for page in range(max_pages):
        params = {"tournaments": tournament, "statuses": statuses,
                  "limit": page_size, "offset": page * page_size}
        if forecaster_id:
            params["forecaster_id"] = forecaster_id
        resp = httpx.get(_METACULUS_POSTS,
                         headers={"Authorization": f"Token {token}"},
                         params=params, timeout=timeout)
        if resp.status_code != 200:
            break
        batch = resp.json().get("results") or []
        results.extend(batch)
        if len(batch) < page_size:
            break
    return results


def fetch_metaculus_coverage(token: str, tournaments: list[str]) -> dict:
    """Per-tournament open-vs-forecasted coverage + the gap list."""
    if not token:
        return {"available": False, "error": "no token"}
    try:
        per: dict[str, dict] = {}
        for t in tournaments:
            open_posts = fetch_posts(token, t, forecaster_id=None)
            mine = fetch_posts(token, t, forecaster_id=303527)
            per[t] = parse_coverage(open_posts, {p["id"] for p in mine})
        return {"available": True, "tournaments": per}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}


def fetch_post_detail(token: str, post_id: int, timeout: float = 15.0) -> dict:
    """GET one post (detail endpoint carries my_forecasts + resolution + score)."""
    resp = httpx.get(f"{_METACULUS_POSTS}{post_id}/",
                     headers={"Authorization": f"Token {token}"}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_recent_resolved(token: str, tournaments: list[str], limit: int = 10) -> dict:
    """Resolved questions mopsus forecast: our value, the outcome, and our score.
    The closest readable signal of 'how is the bot doing' (community CP is locked,
    but question.my_forecasts.score_data is not)."""
    if not token:
        return {"available": False, "error": "no token"}
    try:
        ids: list[int] = []
        for t in tournaments:
            ids += [p["id"] for p in
                    fetch_posts(token, t, forecaster_id=303527, statuses="resolved")]
        resolved = [extract_own_forecast(fetch_post_detail(token, pid))
                    for pid in ids[:limit]]
        return {"available": True, "resolved": resolved, "total": len(ids)}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dashboard_collectors.py -k "paginates or metaculus_coverage or recent_resolved" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bot/dashboard/collectors.py tests/test_dashboard_collectors.py
git commit -m "feat(dashboard): paginated Metaculus coverage + recent-resolved fetchers"
```

---

### Task 6: GitHub run-health + armed-state fetchers (network via `gh`)

**Files:**
- Modify: `bot/dashboard/collectors.py`
- Test: `tests/test_dashboard_collectors.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dashboard_collectors.py
import json as _json

GH_RUNS_JSON = _json.dumps([
    {"databaseId": 1, "status": "completed", "conclusion": "success",
     "createdAt": "2026-06-22T16:08:35Z", "startedAt": "2026-06-22T16:08:35Z",
     "updatedAt": "2026-06-22T16:09:57Z", "event": "schedule"},
    {"databaseId": 2, "status": "completed", "conclusion": "failure",
     "createdAt": "2026-06-22T11:22:31Z", "startedAt": "2026-06-22T11:22:31Z",
     "updatedAt": "2026-06-22T11:25:36Z", "event": "schedule"},
])


def test_parse_github_runs():
    out = collectors.parse_github_runs(GH_RUNS_JSON, now_iso="2026-06-22T16:20:00Z")
    assert out["available"] is True
    assert out["runs"][0]["conclusion"] == "success"
    assert out["runs"][0]["duration_s"] == 82      # 16:09:57 - 16:08:35
    assert out["last_success_minutes_ago"] == 10    # floor((16:20:00 - 16:09:57)/60)
    assert out["recent_failures"] == 1


def test_fetch_github_runs_shells_gh(monkeypatch):
    class _CP:
        returncode = 0
        stdout = GH_RUNS_JSON
    monkeypatch.setattr(collectors.subprocess, "run", lambda *a, **k: _CP())
    out = collectors.fetch_github_runs(now_iso="2026-06-22T16:20:00Z")
    assert out["available"] is True and len(out["runs"]) == 2


def test_fetch_github_runs_handles_gh_failure(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("gh not installed")
    monkeypatch.setattr(collectors.subprocess, "run", boom)
    assert collectors.fetch_github_runs()["available"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dashboard_collectors.py -k github -v`
Expected: FAIL — no attribute `parse_github_runs`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to imports
import subprocess
from datetime import datetime, timezone

_REPO = "aaronbao1/metaculus-forecasting-bot"


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def parse_github_runs(stdout: str, now_iso: str | None = None) -> dict:
    """Shape `gh run list --json ...` output into the run-health panel."""
    runs_raw = (__import__("json").loads(stdout) if stdout.strip() else [])
    now = _parse_iso(now_iso) if now_iso else _now()
    runs = []
    last_success_dt = None
    failures = 0
    for r in runs_raw:
        started = _parse_iso(r["startedAt"]) if r.get("startedAt") else None
        updated = _parse_iso(r["updatedAt"]) if r.get("updatedAt") else None
        dur = int((updated - started).total_seconds()) if started and updated else None
        if r.get("conclusion") == "failure":
            failures += 1
        if r.get("conclusion") == "success" and last_success_dt is None:
            last_success_dt = updated
        runs.append({
            "id": r.get("databaseId"), "event": r.get("event"),
            "conclusion": r.get("conclusion"), "status": r.get("status"),
            "created_at": r.get("createdAt"), "duration_s": dur,
        })
    last_min = (int((now - last_success_dt).total_seconds() // 60)
                if last_success_dt else None)
    return {
        "available": True, "runs": runs,
        "recent_failures": failures,
        "last_success_minutes_ago": last_min,
        "cadence_stale": (last_min is not None and last_min > 40),
    }


def fetch_github_runs(limit: int = 15, now_iso: str | None = None) -> dict:
    """Run `gh run list` for the run-bot workflow; sentinel on any failure."""
    try:
        cp = subprocess.run(
            ["gh", "run", "list", "--workflow", "run-bot", "--repo", _REPO,
             "-L", str(limit), "--json",
             "databaseId,status,conclusion,createdAt,startedAt,updatedAt,event"],
            capture_output=True, text=True, timeout=20,
        )
        if cp.returncode != 0:
            return {"available": False, "error": "gh nonzero"}
        return parse_github_runs(cp.stdout, now_iso=now_iso)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}


def fetch_armed_state() -> dict:
    """Workflow active?, SUBMIT_PREDICTION var, secret names present."""
    try:
        def _gh(args):
            return subprocess.run(["gh", *args, "--repo", _REPO],
                                  capture_output=True, text=True, timeout=20)
        wf = _gh(["workflow", "list"])
        var = _gh(["variable", "list"])
        sec = _gh(["secret", "list"])
        submit = None
        for line in var.stdout.splitlines():
            if line.startswith("SUBMIT_PREDICTION"):
                submit = line.split()[1] if len(line.split()) > 1 else None
        return {
            "available": True,
            "run_bot_active": any(
                l.startswith("run-bot") and "active" in l for l in wf.stdout.splitlines()),
            "submit_prediction": submit,
            "secrets": [l.split()[0] for l in sec.stdout.splitlines() if l.strip()],
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dashboard_collectors.py -k github -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add bot/dashboard/collectors.py tests/test_dashboard_collectors.py
git commit -m "feat(dashboard): GitHub run-health and armed-state fetchers"
```

---

### Task 7: Run-summary reader for the coverage card

**Files:**
- Modify: `bot/dashboard/collectors.py`
- Test: `tests/test_dashboard_collectors.py`

> Reads `logs/runs.jsonl` if present (written by the coverage-safety-net plan). Degrades gracefully when the file is absent so the dashboard works standalone.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dashboard_collectors.py
def test_fetch_run_summary_absent(tmp_path):
    out = collectors.fetch_run_summary(str(tmp_path / "missing.jsonl"))
    assert out["available"] is False


def test_fetch_run_summary_aggregates(tmp_path):
    p = tmp_path / "runs.jsonl"
    p.write_text(
        '{"tournament":"33022","truncated_count":0,"forecast_failed":1,"retry_recovered":1}\n'
        '{"tournament":"33022","truncated_count":3,"forecast_failed":0,"retry_recovered":0}\n'
    )
    out = collectors.fetch_run_summary(str(p))
    assert out["available"] is True
    assert out["total_truncated"] == 3
    assert out["latest"]["truncated_count"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dashboard_collectors.py -k run_summary -v`
Expected: FAIL — no attribute `fetch_run_summary`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to bot/dashboard/collectors.py
import json
import os


def fetch_run_summary(path: str = "logs/runs.jsonl") -> dict:
    """Aggregate the run-summary JSONL (truncation/retry signal)."""
    if not os.path.exists(path):
        return {"available": False}
    try:
        rows = [json.loads(l) for l in open(path) if l.strip()]
        if not rows:
            return {"available": False}
        return {
            "available": True,
            "total_truncated": sum(r.get("truncated_count", 0) for r in rows),
            "total_recovered": sum(r.get("retry_recovered", 0) for r in rows),
            "latest": rows[-1],
            "count": len(rows),
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dashboard_collectors.py -k run_summary -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bot/dashboard/collectors.py tests/test_dashboard_collectors.py
git commit -m "feat(dashboard): run-summary reader (graceful when absent)"
```

---

### Task 8: DashboardState (TTL cache + background refresh)

**Files:**
- Create: `bot/dashboard/state.py`
- Test: `tests/test_dashboard_server.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dashboard_server.py
from bot.dashboard import state as state_mod


def test_state_refresh_isolates_failures():
    def good():
        return {"available": True, "v": 1}
    def bad():
        raise RuntimeError("boom")
    st = state_mod.DashboardState(collectors_map={"good": good, "bad": bad},
                                  interval=999)
    st.refresh_once()
    snap = st.get()
    assert snap["panels"]["good"] == {"available": True, "v": 1}
    assert snap["panels"]["bad"]["available"] is False     # failure isolated
    assert "generated_at" in snap


def test_state_get_is_copy():
    st = state_mod.DashboardState(collectors_map={"x": lambda: {"a": 1}}, interval=999)
    st.refresh_once()
    st.get()["panels"]["x"]["a"] = 999
    assert st.get()["panels"]["x"]["a"] == 1               # external mutation can't poison cache
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dashboard_server.py -k state -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bot.dashboard.state'`.

- [ ] **Step 3: Write minimal implementation**

```python
# bot/dashboard/state.py
"""In-memory dashboard state with a background refresh thread."""
from __future__ import annotations

import copy
import threading
import time
from datetime import datetime, timezone
from typing import Callable


class DashboardState:
    def __init__(self, collectors_map: dict[str, Callable[[], dict]], interval: float = 60.0):
        self._collectors = collectors_map
        self._interval = interval
        self._lock = threading.Lock()
        self._snapshot: dict = {"panels": {}, "generated_at": None}
        self._stop = threading.Event()

    def refresh_once(self) -> None:
        panels = {}
        for name, fn in self._collectors.items():
            try:
                panels[name] = fn()
            except Exception as exc:  # noqa: BLE001 - one dead source must not break others
                panels[name] = {"available": False, "error": str(exc)}
        snap = {"panels": panels,
                "generated_at": datetime.now(timezone.utc).isoformat()}
        with self._lock:
            self._snapshot = snap

    def get(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._snapshot)

    def run_forever(self) -> None:
        while not self._stop.is_set():
            self.refresh_once()
            self._stop.wait(self._interval)

    def start_background(self) -> threading.Thread:
        t = threading.Thread(target=self.run_forever, daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._stop.set()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dashboard_server.py -k state -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bot/dashboard/state.py tests/test_dashboard_server.py
git commit -m "feat(dashboard): DashboardState TTL cache with failure-isolated refresh"
```

---

### Task 9: Server routing (`route()` + handler)

**Files:**
- Create: `bot/dashboard/server.py`
- Create: `bot/dashboard/page.py`
- Test: `tests/test_dashboard_server.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dashboard_server.py
import json
from bot.dashboard import server as server_mod


def test_route_status_returns_json():
    st = state_mod.DashboardState(collectors_map={"x": lambda: {"a": 1}}, interval=999)
    st.refresh_once()
    status, ctype, body = server_mod.route("/api/status", st)
    assert status == 200 and "application/json" in ctype
    assert json.loads(body)["panels"]["x"]["a"] == 1


def test_route_root_returns_html():
    st = state_mod.DashboardState(collectors_map={}, interval=999)
    status, ctype, body = server_mod.route("/", st)
    assert status == 200 and "text/html" in ctype
    assert "<html" in body.lower() and "/api/status" in body  # page polls the endpoint


def test_route_unknown_404():
    st = state_mod.DashboardState(collectors_map={}, interval=999)
    status, _, _ = server_mod.route("/nope", st)
    assert status == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dashboard_server.py -k route -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bot.dashboard.server'`.

- [ ] **Step 3: Write minimal implementation**

```python
# bot/dashboard/page.py
"""Static HTML+JS for the dashboard (polls /api/status, no secrets embedded)."""

PAGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Forecasting bot monitor</title>
<style>
 body{font-family:system-ui,sans-serif;background:#0f1115;color:#e6e6e6;margin:0;padding:20px}
 h1{font-size:18px} .stamp{color:#8a93a2;font-size:12px}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px;margin-top:14px}
 .card{background:#171a21;border:1px solid #232833;border-radius:10px;padding:14px}
 .card h2{font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:#9aa4b2;margin:0 0 8px}
 .bad{color:#e0667a}.ok{color:#5bd6a0}.warn{color:#e7b75f}
 table{width:100%;border-collapse:collapse;font-size:13px}td{padding:2px 4px}
</style></head><body>
<h1>Forecasting bot monitor <span class="stamp" id="stamp"></span></h1>
<div class="grid" id="grid"></div>
<script>
async function tick(){
 try{
  const r = await fetch('/api/status'); const s = await r.json();
  document.getElementById('stamp').textContent = 'data as of '+(s.generated_at||'—');
  const g = document.getElementById('grid'); g.innerHTML='';
  for(const [name,p] of Object.entries(s.panels||{})){
   const d=document.createElement('div'); d.className='card';
   const body = p && p.available===false
     ? '<span class="bad">unavailable'+(p.error?(' — '+p.error):'')+'</span>'
     : '<pre style="white-space:pre-wrap;margin:0;font-size:12px">'+
        JSON.stringify(p,null,1)+'</pre>';
   d.innerHTML='<h2>'+name+'</h2>'+body; g.appendChild(d);
  }
 }catch(e){ document.getElementById('stamp').textContent='fetch error: '+e; }
}
tick(); setInterval(tick, 10000);
</script></body></html>"""
```

```python
# bot/dashboard/server.py
"""Localhost-only dashboard server (stdlib http.server)."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bot.dashboard.page import PAGE_HTML
from bot.dashboard.state import DashboardState


def route(path: str, state: DashboardState) -> tuple[int, str, str]:
    """Pure router: path -> (status, content_type, body)."""
    if path == "/" or path.startswith("/index"):
        return 200, "text/html; charset=utf-8", PAGE_HTML
    if path == "/api/status":
        return 200, "application/json", json.dumps(state.get(), default=str)
    return 404, "text/plain", "not found"


def make_handler(state: DashboardState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            status, ctype, body = route(self.path, state)
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):  # silence default stderr logging
            pass
    return Handler


def serve(state: DashboardState, port: int = 8787) -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    httpd.serve_forever()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dashboard_server.py -k route -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add bot/dashboard/server.py bot/dashboard/page.py tests/test_dashboard_server.py
git commit -m "feat(dashboard): localhost server with pure route() and polling page"
```

---

### Task 10: Entrypoint `scripts/dashboard.py`

**Files:**
- Create: `scripts/dashboard.py`
- Test: `tests/test_dashboard_server.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dashboard_server.py
import importlib.util, pathlib


def _load_dashboard_script():
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "dashboard.py"
    spec = importlib.util.spec_from_file_location("dashboard_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_collectors_map_wires_sources(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "KEY")
    monkeypatch.setenv("METACULUS_TOKEN", "TOK")
    mod = _load_dashboard_script()
    cmap = mod.build_collectors_map()
    assert set(cmap) >= {"cost", "coverage", "resolved", "runs", "armed", "run_summary"}
    # each entry is callable with no args
    assert all(callable(fn) for fn in cmap.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dashboard_server.py -k build_collectors_map -v`
Expected: FAIL — file `scripts/dashboard.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/dashboard.py
"""Local auto-refreshing dashboard for the forecasting bot.

Run:  .venv/bin/python scripts/dashboard.py [--port 8787] [--refresh-interval 60]
Then open http://127.0.0.1:8787  (localhost only; no secrets are served).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv(override=False)

from bot.dashboard import collectors  # noqa: E402
from bot.dashboard.state import DashboardState  # noqa: E402
from bot.dashboard.server import serve  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard")

TOURNAMENTS = ["33022", "minibench"]


def build_collectors_map() -> dict:
    or_key = os.getenv("OPENROUTER_API_KEY", "")
    mc_token = os.getenv("METACULUS_TOKEN", "")
    return {
        "cost": lambda: collectors.fetch_openrouter_budget(or_key),
        "coverage": lambda: collectors.fetch_metaculus_coverage(mc_token, TOURNAMENTS),
        "resolved": lambda: collectors.fetch_recent_resolved(mc_token, TOURNAMENTS),
        "runs": lambda: collectors.fetch_github_runs(),
        "armed": lambda: collectors.fetch_armed_state(),
        "run_summary": lambda: collectors.fetch_run_summary(),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--refresh-interval", type=float, default=60.0)
    args = ap.parse_args(argv)

    state = DashboardState(build_collectors_map(), interval=args.refresh_interval)
    state.refresh_once()
    state.start_background()
    logger.info("dashboard at http://127.0.0.1:%d  (refresh %ss)",
                args.port, args.refresh_interval)
    try:
        serve(state, port=args.port)
    except KeyboardInterrupt:
        state.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dashboard_server.py -k build_collectors_map -v`
Expected: PASS.

- [ ] **Step 5: Run the FULL suite (no regressions) + a manual smoke**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (baseline 509 + the new dashboard tests), 1 skipped.

Manual smoke (optional, needs real `.env`): `.venv/bin/python scripts/dashboard.py --port 8799 &` then `curl -s http://127.0.0.1:8799/api/status | head -c 300` should return JSON with a `panels` key; `kill %1` after.

- [ ] **Step 6: Commit**

```bash
git add scripts/dashboard.py tests/test_dashboard_server.py
git commit -m "feat(dashboard): entrypoint wiring live collectors into the server"
```

---

### Task 11: Optional launchd KeepAlive + README note

**Files:**
- Create: `deploy/com.aaron.metaculus-bot-dashboard.plist`
- Modify: `SETUP.md`

- [ ] **Step 1: Write the plist**

```xml
<!-- deploy/com.aaron.metaculus-bot-dashboard.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.aaron.metaculus-bot-dashboard</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/aaronbao/Developer/sideprojects/forecasting/.venv/bin/python</string>
    <string>/Users/aaronbao/Developer/sideprojects/forecasting/scripts/dashboard.py</string>
    <string>--port</string><string>8787</string>
  </array>
  <key>WorkingDirectory</key>
  <string>/Users/aaronbao/Developer/sideprojects/forecasting</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key>
  <string>/Users/aaronbao/Library/Application Support/metaculus-bot/dashboard.out.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/aaronbao/Library/Application Support/metaculus-bot/dashboard.err.log</string>
</dict></plist>
```

- [ ] **Step 2: Document load/unload in SETUP.md**

Append a short "Live dashboard" section to `SETUP.md`:

```markdown
## Live dashboard
Run manually: `.venv/bin/python scripts/dashboard.py` → http://127.0.0.1:8787
Always-on (launchd):
  cp deploy/com.aaron.metaculus-bot-dashboard.plist ~/Library/LaunchAgents/
  launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.aaron.metaculus-bot-dashboard.plist
Stop: `launchctl bootout gui/$(id -u)/com.aaron.metaculus-bot-dashboard`
```

- [ ] **Step 3: Commit**

```bash
git add deploy/com.aaron.metaculus-bot-dashboard.plist SETUP.md
git commit -m "feat(dashboard): optional launchd KeepAlive + setup docs"
```

---

## Self-Review checklist (run after all tasks)
- [ ] Full suite green: `.venv/bin/python -m pytest -q`
- [ ] No secret (`OPENROUTER_API_KEY`, `METACULUS_TOKEN`, `data.label`, `creator_user_id`) appears in any `/api/status` output or committed file.
- [ ] Server binds `127.0.0.1` only (not `0.0.0.0`).
- [ ] Every fetcher returns `{"available": False, ...}` instead of raising.
- [ ] Manual smoke: page loads, panels render, "data as of" updates.
