import json
from datetime import datetime, timezone

from backtest.corpus.forecastbench import import_forecastbench
from backtest.corpus.store import CorpusStore

QUESTIONS_FILE = {
    "forecast_due_date": "2024-07-21",
    "questions": [
        {"id": "11111", "source": "metaculus",
         "question": "Will Y happen by 2025?",
         "background": "FB background.",
         "resolution_criteria": "Resolves YES if Y.",
         "market_info_open_datetime": "2024-01-01T00:00:00+00:00",
         "market_info_close_datetime": "2024-12-31T00:00:00+00:00",
         "freeze_datetime": "2024-07-21T00:00:00+00:00",
         "freeze_datetime_value": "0.62",
         "combination_of": "N/A",
         "resolution_date": "2025-01-02"},
        # combination questions are derived — skipped
        {"id": ["11111", "22222"], "source": "metaculus",
         "question": "combo", "combination_of": [{"id": "11111"}]},
        # no resolution row -> imported unresolved
        {"id": "33333", "source": "manifold",
         "question": "Open one?", "combination_of": "N/A",
         "market_info_open_datetime": "2024-02-01T00:00:00+00:00",
         "market_info_close_datetime": "2026-12-31T00:00:00+00:00",
         "freeze_datetime": "2024-07-21T00:00:00+00:00",
         "freeze_datetime_value": None},
    ],
}

RESOLUTIONS_FILE = {
    "resolutions": [
        {"id": "11111", "source": "metaculus",
         "resolution_date": "2025-01-02", "resolved_to": 1},
    ],
}


def _write_tree(tmp_path):
    root = tmp_path / "forecastbench-datasets"
    qdir = root / "datasets" / "question_sets"
    rdir = root / "datasets" / "resolution_sets"
    qdir.mkdir(parents=True)
    rdir.mkdir(parents=True)
    (qdir / "2024-07-21-llm.json").write_text(json.dumps(QUESTIONS_FILE))
    (rdir / "2024-07-21_resolution_set.json").write_text(
        json.dumps(RESOLUTIONS_FILE))
    return root


def test_import_forecastbench(tmp_path):
    root = _write_tree(tmp_path)
    store = CorpusStore(tmp_path / "corpus")
    report = import_forecastbench(store, root)
    assert report.fetched == 2            # combo skipped
    q = store.get_question("forecastbench:metaculus:11111")
    assert q.qtype.value == "binary"
    assert q.resolution is True           # joined from resolution_sets
    assert q.title == "Will Y happen by 2025?"
    assert q.provenance["source"] == "forecastbench"
    assert q.provenance["origin_platform"] == "metaculus"
    assert q.provenance["license"] == "CC BY-SA 4.0"
    # freeze value becomes a one-point cp series (crowd value at freeze)
    assert store.value_at(
        "forecastbench:metaculus:11111", "cp",
        datetime(2024, 8, 1, tzinfo=timezone.utc)) == 0.62
    open_q = store.get_question("forecastbench:manifold:33333")
    assert open_q.resolution is None
    assert store.value_at(
        "forecastbench:manifold:33333", "cp",
        datetime(2024, 8, 1, tzinfo=timezone.utc)) is None


def test_import_forecastbench_dedupes_across_nightly_sets(tmp_path):
    root = _write_tree(tmp_path)
    qdir = root / "datasets" / "question_sets"
    (qdir / "2024-07-28-llm.json").write_text(json.dumps(QUESTIONS_FILE))
    store = CorpusStore(tmp_path / "corpus")
    report = import_forecastbench(store, root)
    assert report.fetched == 2            # unique qids, not rows
