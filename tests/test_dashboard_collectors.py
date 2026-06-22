from pytest import approx as pytestapprox

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
    monkeypatch.setattr(collectors, "_MIN_METACULUS_SPACING", 0.0)  # no throttle in tests
    monkeypatch.setattr(collectors.httpx, "get", fake_get)
    posts = collectors.fetch_posts("TOK", tournament="33022", forecaster_id=None)
    assert [p["id"] for p in posts] == [1, 2]
    assert calls == [0, 100, 200]  # paginated by limit=100 until empty


def test_metaculus_get_retries_once_on_429(monkeypatch):
    monkeypatch.setattr(collectors, "_MIN_METACULUS_SPACING", 0.0)
    monkeypatch.setattr(collectors.time, "sleep", lambda s: None)  # don't really sleep
    seq = [_FakeResp(429, {}), _FakeResp(200, {"results": []})]
    calls = []
    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(1)
        return seq[len(calls) - 1]
    monkeypatch.setattr(collectors.httpx, "get", fake_get)
    resp = collectors._metaculus_get("http://x", "TOK")
    assert resp.status_code == 200      # recovered after one 429 retry
    assert len(calls) == 2


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
