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
