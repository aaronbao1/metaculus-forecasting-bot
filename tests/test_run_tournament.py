# tests/test_run_tournament.py
import importlib.util, pathlib


def _load():
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_tournament.py"
    spec = importlib.util.spec_from_file_location("run_tournament_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def test_run_one_tournament_appends_run_summary(monkeypatch):
    mod = _load()
    captured = []
    monkeypatch.setattr(mod, "append_run", lambda rec: captured.append(rec))

    class FakeBot:
        _last_run_stats = {"tournament": "T", "truncated_count": 0, "forecast_ok": 1}
        async def forecast_on_tournament(self, t, return_exceptions=True):
            return ["ok"]

    reports = await mod._run_one_tournament(FakeBot(), "T")
    assert reports == ["ok"]
    assert captured == [{"tournament": "T", "truncated_count": 0, "forecast_ok": 1}]


async def test_run_one_tournament_survives_append_failure(monkeypatch):
    mod = _load()
    def boom(rec):
        raise IOError("disk full")
    monkeypatch.setattr(mod, "append_run", boom)

    class FakeBot:
        _last_run_stats = {"tournament": "T"}
        async def forecast_on_tournament(self, t, return_exceptions=True):
            return ["ok"]

    # append failure must NOT propagate (run already succeeded)
    assert await mod._run_one_tournament(FakeBot(), "T") == ["ok"]
