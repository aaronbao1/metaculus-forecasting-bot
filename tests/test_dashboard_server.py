# tests/test_dashboard_server.py
from bot.dashboard import state as state_mod


def test_state_refresh_isolates_failures():
    def good():
        return {"available": True, "v": 1}
    def bad():
        raise RuntimeError("boom")
    st = state_mod.DashboardState({"good": good, "bad": bad})
    st.refresh_once()
    snap = st.get()
    assert snap["panels"]["good"] == {"available": True, "v": 1}
    assert snap["panels"]["bad"]["available"] is False     # failure isolated
    assert "generated_at" in snap


def test_state_get_is_copy():
    st = state_mod.DashboardState({"x": lambda: {"a": 1}})
    st.refresh_once()
    st.get()["panels"]["x"]["a"] = 999
    assert st.get()["panels"]["x"]["a"] == 1               # external mutation can't poison cache


import json
from bot.dashboard import server as server_mod


def test_route_status_returns_json():
    st = state_mod.DashboardState({"x": lambda: {"a": 1}})
    st.refresh_once()
    status, ctype, body = server_mod.route("/api/status", st)
    assert status == 200 and "application/json" in ctype
    assert json.loads(body)["panels"]["x"]["a"] == 1


def test_route_root_returns_html():
    st = state_mod.DashboardState({})
    status, ctype, body = server_mod.route("/", st)
    assert status == 200 and "text/html" in ctype
    assert "<html" in body.lower() and "/api/status" in body  # page polls the endpoint


def test_route_unknown_404():
    st = state_mod.DashboardState({})
    status, _, _ = server_mod.route("/nope", st)
    assert status == 404


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
    for spec in cmap.values():
        fn, interval = spec
        assert callable(fn) and interval > 0
    # Metaculus collectors poll less often than the cheap ones
    assert cmap["resolved"][1] > cmap["cost"][1]
    assert cmap["coverage"][1] > cmap["runs"][1]


def test_per_collector_cadence():
    calls = {"slow": 0, "fast": 0}
    def slow():
        calls["slow"] += 1
        return {"available": True}
    def fast():
        calls["fast"] += 1
        return {"available": True}
    st = state_mod.DashboardState({"slow": (slow, 1000), "fast": (fast, 0)})
    st.refresh_once()                       # force all
    assert calls == {"slow": 1, "fast": 1}
    st.refresh_due()                        # only elapsed collectors
    assert calls["fast"] == 2               # interval 0 -> always due
    assert calls["slow"] == 1               # interval 1000 -> not due yet


def test_refresh_due_isolates_failures():
    def boom():
        raise RuntimeError("x")
    st = state_mod.DashboardState({"bad": (boom, 0)})
    st.refresh_due()
    assert st.get()["panels"]["bad"]["available"] is False
