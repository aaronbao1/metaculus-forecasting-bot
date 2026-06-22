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
