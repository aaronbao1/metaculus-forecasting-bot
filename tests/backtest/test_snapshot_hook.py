"""Post-run snapshot accumulator hook: archives this run's questions into the
backtest corpus. The hook must NEVER fail a tournament run.

Also covers the return-value contract of main(): it must return only the
records appended during that run (read_forecasts(...)[n_before:]), so that
the __main__ block can call post_run_snapshot(new_records) after the event
loop exits — eliminating the implicit nest_asyncio dependency."""
import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import scripts.run_tournament as rt


class _FakeStore:
    def __init__(self, root):
        self.root = root


class _FakeReport:
    fetched = 2
    withheld: dict = {}
    errors: list = []


def _install_fakes(monkeypatch, calls, fail=False):
    """Patch the corpus seams the hook imports at call time."""
    import backtest.corpus.snapshot as snapshot_mod
    import backtest.corpus.store as store_mod

    def fake_snapshot(store, post_ids):
        if fail:
            raise RuntimeError("metaculus down")
        calls.append((store, post_ids))
        return _FakeReport()

    monkeypatch.setattr(snapshot_mod, "snapshot_questions", fake_snapshot)
    monkeypatch.setattr(store_mod, "CorpusStore", _FakeStore)


def test_collects_deduped_sorted_int_ids_into_default_store(monkeypatch):
    monkeypatch.delenv("SNAPSHOT_AFTER_RUN", raising=False)
    calls = []
    _install_fakes(monkeypatch, calls)
    records = [{"question_id": 5, "type": "binary"},
               {"question_id": 3, "type": "binary"},
               {"question_id": 5, "type": "multiple_choice"},
               {"question_id": "7", "type": "binary"},   # string ids coerced
               {"question_id": None}]
    rt.post_run_snapshot(records)
    assert len(calls) == 1
    store, ids = calls[0]
    assert ids == [3, 5, 7]
    assert store.root == Path("data/backtest")


def test_env_flag_false_disables_snapshot(monkeypatch):
    monkeypatch.setenv("SNAPSHOT_AFTER_RUN", "false")
    calls = []
    _install_fakes(monkeypatch, calls)
    rt.post_run_snapshot([{"question_id": 1}])
    assert calls == []


def test_env_flag_defaults_to_true(monkeypatch):
    monkeypatch.delenv("SNAPSHOT_AFTER_RUN", raising=False)
    calls = []
    _install_fakes(monkeypatch, calls)
    rt.post_run_snapshot([{"question_id": 1}])
    assert len(calls) == 1


def test_empty_records_do_not_call_snapshot(monkeypatch):
    monkeypatch.delenv("SNAPSHOT_AFTER_RUN", raising=False)
    calls = []
    _install_fakes(monkeypatch, calls)
    rt.post_run_snapshot([])
    rt.post_run_snapshot([{"question_id": None}])
    assert calls == []


def test_snapshot_failure_is_swallowed_and_logged(monkeypatch, caplog):
    monkeypatch.delenv("SNAPSHOT_AFTER_RUN", raising=False)
    _install_fakes(monkeypatch, calls=[], fail=True)
    with caplog.at_level(logging.ERROR):
        rt.post_run_snapshot([{"question_id": 1}])   # must not raise
    assert "snapshot failed" in caplog.text


# ---------------------------------------------------------------------------
# Return-value contract for main()
# ---------------------------------------------------------------------------

def _make_fake_bot(tournament_records: list[dict]):
    """Return a fake bot whose forecast_on_tournament yields the given records
    (appended to read_forecasts) and returns a list with one mock report."""
    bot = MagicMock()
    bot.forecast_on_tournament = AsyncMock(return_value=[MagicMock()])
    return bot


def test_main_returns_only_new_records(monkeypatch):
    """main() must return read_forecasts(...)[n_before:] — the slice of records
    added during this run — so the __main__ block can pass them directly to
    post_run_snapshot without nest_asyncio."""
    pre_existing = [{"question_id": 1, "type": "binary"}]
    new_this_run  = [{"question_id": 2, "type": "binary"},
                     {"question_id": 3, "type": "binary"}]
    all_records   = pre_existing + new_this_run

    # Simulate read_forecasts returning pre_existing on first call, then
    # all_records on second call (after the bot has run).
    call_count = {"n": 0}

    def fake_read_forecasts(_path):
        call_count["n"] += 1
        return pre_existing if call_count["n"] == 1 else all_records

    fake_bot = _make_fake_bot(new_this_run)

    monkeypatch.setattr(rt, "read_forecasts", fake_read_forecasts)
    monkeypatch.setattr(rt, "build_framework_bot", lambda _cfg: fake_bot)

    # Provide a minimal MetaculusApi stub so the import inside main() works.
    fake_api = MagicMock()
    fake_api.CURRENT_AI_COMPETITION_ID = 3349
    fake_api.CURRENT_MINIBENCH_ID      = 3350

    with patch.dict("sys.modules", {"forecasting_tools": MagicMock(
            MetaculusApi=fake_api)}):
        result = asyncio.run(rt.main())

    assert result == new_this_run


def test_main_returns_empty_list_when_no_new_records(monkeypatch):
    """When the bot produces no forecasts, main() returns []."""
    existing = [{"question_id": 99}]

    monkeypatch.setattr(rt, "read_forecasts", lambda _p: existing)
    monkeypatch.setattr(rt, "build_framework_bot",
                        lambda _cfg: _make_fake_bot([]))

    fake_api = MagicMock()
    fake_api.CURRENT_AI_COMPETITION_ID = 3349
    fake_api.CURRENT_MINIBENCH_ID      = 3350

    with patch.dict("sys.modules", {"forecasting_tools": MagicMock(
            MetaculusApi=fake_api)}):
        result = asyncio.run(rt.main())

    assert result == []
