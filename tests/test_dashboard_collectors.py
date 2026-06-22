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
