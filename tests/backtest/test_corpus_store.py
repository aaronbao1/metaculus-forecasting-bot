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


# ---------------------------------------------------------------------------
# Task 9: series blobs, value_at, contamination
# ---------------------------------------------------------------------------

def test_put_series_and_value_at_step_lookup(tmp_path):
    store = CorpusStore(tmp_path)
    store.upsert_question(_numeric_question())
    store.put_series("metaculus:31678", "cp", [
        ("2026-03-01T00:00:00+00:00", 0.40),
        ("2026-01-01T00:00:00+00:00", 0.25),     # unsorted on purpose
        ("2026-05-01T00:00:00+00:00", 0.55),
    ])
    # step function: last point with ts <= t
    assert store.value_at("metaculus:31678", "cp", _utc(2025, 12, 1)) is None
    assert store.value_at("metaculus:31678", "cp", _utc(2026, 1, 1)) == 0.25
    assert store.value_at("metaculus:31678", "cp", _utc(2026, 4, 15)) == 0.40
    assert store.value_at("metaculus:31678", "cp", _utc(2027, 1, 1)) == 0.55
    # absent kind / absent qid
    assert store.value_at("metaculus:31678", "market_price",
                          _utc(2026, 4, 15)) is None
    assert store.value_at("metaculus:0", "cp", _utc(2026, 4, 15)) is None
    # put_series REPLACES the series
    store.put_series("metaculus:31678", "cp",
                     [("2026-06-01T00:00:00+00:00", 0.9)])
    assert store.value_at("metaculus:31678", "cp", _utc(2026, 4, 15)) is None
    # blob lives under blobs/
    assert list((tmp_path / "blobs").glob("*__cp.json"))


def test_value_at_naive_query_against_aware_series(tmp_path):
    store = CorpusStore(tmp_path)
    store.put_series("metaculus:31678", "cp",
                     [("2026-01-01T00:00:00+00:00", 0.25)])
    assert store.value_at("metaculus:31678", "cp",
                          datetime(2026, 2, 1)) == 0.25


def test_contamination_set_get(tmp_path):
    store = CorpusStore(tmp_path)
    assert store.get_contamination("metaculus:1", "h1") is None
    store.set_contamination("metaculus:1", "h1", "clean")
    store.set_contamination("metaculus:1", "h2", "contaminated")
    assert store.get_contamination("metaculus:1", "h1") == "clean"
    assert store.get_contamination("metaculus:1", "h2") == "contaminated"
    store.set_contamination("metaculus:1", "h1", "contaminated-confirmed")
    assert (store.get_contamination("metaculus:1", "h1")
            == "contaminated-confirmed")


# ---------------------------------------------------------------------------
# Task 9 addendum: WAL/check_same_thread, close/context-manager, falsy float
# ---------------------------------------------------------------------------

def test_resolution_zero_float_round_trip(tmp_path):
    """resolution=0.0 must round-trip as 0.0, not None/False (falsy boundary)."""
    store = CorpusStore(tmp_path)
    q = replace(_numeric_question(), resolution=0.0)
    store.upsert_question(q)
    got = store.get_question(q.qid)
    assert got.resolution == 0.0
    assert got.resolution is not None
    assert got.resolution is not False


def test_close_method(tmp_path):
    """close() should succeed without error."""
    store = CorpusStore(tmp_path)
    store.upsert_question(_numeric_question())
    store.close()  # must not raise


def test_context_manager(tmp_path):
    """CorpusStore as a context manager: data readable inside, conn closed on exit."""
    with CorpusStore(tmp_path) as store:
        store.upsert_question(_numeric_question())
        assert store.get_question("metaculus:31678") is not None
    # After __exit__, connection is closed; further queries should raise
    import pytest
    with pytest.raises(Exception):
        store.get_question("metaculus:31678")


# ---------------------------------------------------------------------------
# Task 9: query filters
# ---------------------------------------------------------------------------

def _seed_query_corpus(store):
    base = _numeric_question()
    store.upsert_question(replace(
        base, qid="metaculus:1", qtype=QType.BINARY, resolution=True,
        scaling=None, open_time=_utc(2026, 5, 1)))
    store.upsert_question(replace(
        base, qid="metaculus:2", resolution=None))           # unresolved
    store.upsert_question(replace(
        base, qid="polymarket:a", platform="polymarket",
        qtype=QType.BINARY, resolution=False, scaling=None,
        open_time=_utc(2026, 1, 10)))


def test_query_filters(tmp_path):
    store = CorpusStore(tmp_path)
    _seed_query_corpus(store)
    assert [q.qid for q in store.query()] == ["metaculus:1", "polymarket:a"]
    assert [q.qid for q in store.query(resolved_only=False)] == \
        ["metaculus:1", "metaculus:2", "polymarket:a"]
    assert [q.qid for q in store.query(platforms=["polymarket"])] == \
        ["polymarket:a"]
    assert [q.qid for q in store.query(qtypes=[QType.BINARY],
                                       open_after=_utc(2026, 3, 1))] == \
        ["metaculus:1"]
    assert [q.qid for q in store.query(qtypes=["numeric"],
                                       resolved_only=False)] == \
        ["metaculus:2"]
    assert store.query(close_before=_utc(2025, 1, 1)) == []
    assert len(store.query(limit=1)) == 1


# ---------------------------------------------------------------------------
# Store hardening: UTC-normalised timestamps, collision-proof blobs, empty series
# ---------------------------------------------------------------------------

def test_mixed_offset_series_sorted_by_true_utc_time(tmp_path):
    """Points given with mixed offsets must be stored in true chronological
    (UTC) order so value_at returns the chronologically-correct value.

    The tricky case: raw-string sort and UTC sort differ.
      "2026-06-01T14:00:00-05:00" sorts BEFORE "2026-06-01T16:00:00+02:00"
      lexicographically (14 < 16) but is LATER in UTC (19:00 vs 14:00).
    """
    store = CorpusStore(tmp_path)
    store.put_series("metaculus:31678", "cp", [
        ("2026-06-01T14:00:00-05:00", 0.90),   # UTC 19:00 — latest
        ("2026-06-01T16:00:00+02:00", 0.10),   # UTC 14:00 — earliest
        ("2026-06-01T15:00:00Z",      0.50),   # UTC 15:00 — middle
    ])
    # Before earliest point (UTC 13:59)
    assert store.value_at("metaculus:31678", "cp",
                          _utc(2026, 6, 1, 13, 59)) is None
    # Between earliest (14:00) and middle (15:00): expect 0.10
    assert store.value_at("metaculus:31678", "cp",
                          _utc(2026, 6, 1, 14, 30)) == 0.10
    # Between middle (15:00) and latest (19:00): expect 0.50
    assert store.value_at("metaculus:31678", "cp",
                          _utc(2026, 6, 1, 17)) == 0.50
    # After all points: expect 0.90
    assert store.value_at("metaculus:31678", "cp",
                          _utc(2026, 6, 1, 20)) == 0.90


def test_blob_filename_collision_distinct_qids(tmp_path):
    """'p:0xAB/CD' and 'p:0xAB_CD' must NOT share a blob file; each qid's
    value_at must return its own value."""
    store = CorpusStore(tmp_path)
    qid_a = "p:0xAB/CD"
    qid_b = "p:0xAB_CD"
    store.put_series(qid_a, "cp", [("2026-01-01T00:00:00+00:00", 0.11)])
    store.put_series(qid_b, "cp", [("2026-01-01T00:00:00+00:00", 0.99)])
    probe = _utc(2026, 6, 1)
    assert store.value_at(qid_a, "cp", probe) == 0.11
    assert store.value_at(qid_b, "cp", probe) == 0.99


def test_empty_series_value_at_returns_none(tmp_path):
    """put_series([]) writes an empty series; value_at must return None."""
    store = CorpusStore(tmp_path)
    store.put_series("metaculus:31678", "cp", [])
    assert store.value_at("metaculus:31678", "cp", _utc(2026, 6, 1)) is None


def test_query_stratum_requires_hash_and_joins(tmp_path):
    import pytest
    store = CorpusStore(tmp_path)
    _seed_query_corpus(store)
    store.set_contamination("metaculus:1", "abc123", "clean")
    store.set_contamination("polymarket:a", "abc123", "contaminated")
    with pytest.raises(ValueError):
        store.query(stratum="clean")
    assert [q.qid for q in store.query(stratum="clean",
                                       model_set_hash="abc123")] == \
        ["metaculus:1"]
    assert [q.qid for q in store.query(stratum="contaminated",
                                       model_set_hash="abc123")] == \
        ["polymarket:a"]
