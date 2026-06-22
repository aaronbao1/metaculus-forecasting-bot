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
