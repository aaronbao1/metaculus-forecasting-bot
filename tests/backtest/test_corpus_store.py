from dataclasses import replace
from datetime import datetime, timezone

from backtest.core import BacktestQuestion, QType
from backtest.corpus.store import CorpusStore, FetchReport


def _utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


def _numeric_question():
    return BacktestQuestion(
        qid="metaculus:31678", platform="metaculus", native_id="31678",
        qtype=QType.NUMERIC, title="How many widgets by 2027?",
        description="Background.", resolution_criteria="Counted by X.",
        fine_print="Edge cases.", open_time=_utc(2026, 1, 1),
        scheduled_close_time=_utc(2026, 12, 31),
        spot_scoring_time=_utc(2026, 12, 1),
        actual_resolve_time=_utc(2027, 1, 5), resolution=41.5,
        options=(),
        scaling={"range_min": 0, "range_max": 100, "zero_point": None,
                 "open_lower_bound": False, "open_upper_bound": True,
                 "cdf_size": 201, "unit": "widgets"},
        question_weight=0.5,
        provenance={"source": "metaculus", "withheld": ["aggregations"]})


def test_fetch_report_defaults():
    r = FetchReport()
    assert r.fetched == 0 and r.withheld == {} and r.errors == []


def test_init_creates_db_and_blobs_dir(tmp_path):
    CorpusStore(tmp_path / "corpus")
    assert (tmp_path / "corpus" / "corpus.db").exists()
    assert (tmp_path / "corpus" / "blobs").is_dir()


def test_upsert_get_round_trip(tmp_path):
    store = CorpusStore(tmp_path)
    q = _numeric_question()
    store.upsert_question(q)
    got = store.get_question("metaculus:31678")
    assert got == q          # frozen dataclass equality covers every field


def test_get_missing_returns_none(tmp_path):
    assert CorpusStore(tmp_path).get_question("metaculus:0") is None


def test_upsert_replaces_existing_row(tmp_path):
    store = CorpusStore(tmp_path)
    q = _numeric_question()
    store.upsert_question(q)
    store.upsert_question(replace(q, title="Updated title"))
    assert store.get_question(q.qid).title == "Updated title"


def test_binary_bool_and_none_resolution_round_trip(tmp_path):
    store = CorpusStore(tmp_path)
    base = _numeric_question()
    yes = replace(base, qid="metaculus:1", qtype=QType.BINARY,
                  resolution=True, scaling=None)
    unresolved = replace(base, qid="metaculus:2", resolution=None)
    store.upsert_question(yes)
    store.upsert_question(unresolved)
    assert store.get_question("metaculus:1").resolution is True
    assert store.get_question("metaculus:2").resolution is None
