"""The committed 5-question mini corpus and its loader — the substrate of
the CI smoke gate. 3 binary, 1 MC, 1 numeric; all resolved; all with a CP
series."""
from datetime import datetime, timezone
from pathlib import Path

from backtest.core import QType
from backtest.corpus.fixtures import load_fixture_corpus
from backtest.corpus.store import CorpusStore

FIXTURES = Path(__file__).parent / "fixtures" / "mini_corpus"


def test_loads_five_questions_with_resolutions(tmp_path):
    store = CorpusStore(tmp_path / "store")
    assert load_fixture_corpus(store, FIXTURES) == 5
    qs = store.query(platforms=["metaculus"], resolved_only=True)
    assert len(qs) == 5
    by_type: dict = {}
    for q in qs:
        by_type.setdefault(q.qtype, []).append(q)
    assert len(by_type[QType.BINARY]) == 3
    assert len(by_type[QType.MULTIPLE_CHOICE]) == 1
    assert len(by_type[QType.NUMERIC]) == 1
    assert all(q.resolution is not None for q in qs)
    mc = by_type[QType.MULTIPLE_CHOICE][0]
    assert mc.resolution in mc.options
    num = by_type[QType.NUMERIC][0]
    assert num.scaling["cdf_size"] == 201


def test_horizon_window_fits_close_minus_7d(tmp_path):
    store = CorpusStore(tmp_path / "store")
    load_fixture_corpus(store, FIXTURES)
    q = store.get_question("metaculus:90001")
    assert q.open_time <= q.scheduled_close_time
    # close-7d (2025-02-22) must be inside [open, close) so the smoke
    # experiment's single horizon never skips a fixture question
    from datetime import timedelta
    assert q.scheduled_close_time - timedelta(days=7) >= q.open_time


def test_cp_series_step_lookup(tmp_path):
    store = CorpusStore(tmp_path / "store")
    load_fixture_corpus(store, FIXTURES)
    t = datetime(2025, 2, 22, tzinfo=timezone.utc)
    assert store.value_at("metaculus:90001", "cp", t) == 0.82
    early = datetime(2025, 1, 10, tzinfo=timezone.utc)
    assert store.value_at("metaculus:90001", "cp", early) == 0.55
    mc = store.value_at("metaculus:90004", "cp", t)
    assert mc == {"0-3": 0.15, "4-6": 0.60, "7 or more": 0.25}
