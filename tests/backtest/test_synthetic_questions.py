"""Task 8: synthetic numeric questions from target series (Retrocast ext. 4
seed) — unlimited leak-free numeric questions for distribution-plumbing
experiments."""
from datetime import datetime, timedelta, timezone

from backtest.core import QType
from backtest.corpus.store import CorpusStore
from backtest.corpus.synthetic import generate_questions

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _store_with_series(tmp_path, n=400):
    store = CorpusStore(tmp_path / "corpus")
    store.put_series("fred:TEST", "target", [
        ((T0 + timedelta(days=i)).isoformat(), 100.0 + i) for i in range(n)])
    return store


def test_generates_resolved_numeric_questions(tmp_path):
    store = _store_with_series(tmp_path)
    qs = generate_questions(store, "fred:TEST", n=5, horizon_days=30,
                            seed=1, name="Test Index")
    assert len(qs) == 5
    for q in qs:
        assert q.platform == "synthetic" and q.qtype is QType.NUMERIC
        assert q.resolution is not None
        assert q.provenance["synthetic"] is True
        assert q.provenance["source_qid"] == "fred:TEST"
        # resolution equals the series value at close (linear series: 100+day)
        close_day = (q.scheduled_close_time - T0).days
        assert q.resolution == 100.0 + close_day
        # bounds contain the trailing window and the resolution
        assert q.scaling["range_min"] < q.resolution < q.scaling["range_max"]
        assert q.open_time < q.scheduled_close_time
        # target series copied so TimeSeriesSystem can address it by qid
        assert store.series_points(q.qid, "target")
        # stored in the corpus
        assert store.get_question(q.qid) is not None


def test_deterministic_under_seed(tmp_path):
    a = generate_questions(_store_with_series(tmp_path / "a"), "fred:TEST",
                           n=4, horizon_days=30, seed=7)
    b = generate_questions(_store_with_series(tmp_path / "b"), "fred:TEST",
                           n=4, horizon_days=30, seed=7)
    assert [q.qid for q in a] == [q.qid for q in b]
    assert [q.resolution for q in a] == [q.resolution for q in b]


def test_too_short_series_yields_nothing(tmp_path):
    store = _store_with_series(tmp_path, n=40)   # < min_history + horizon
    qs = generate_questions(store, "fred:TEST", n=5, horizon_days=30, seed=1)
    assert qs == []


def test_cli_synth_questions(tmp_path, capsys):
    store_root = tmp_path / "corpus"
    store = CorpusStore(store_root)
    store.put_series("fred:TEST", "target", [
        ((T0 + timedelta(days=i)).isoformat(), 100.0 + i) for i in range(400)])
    store.close()
    from scripts.backtest import main
    rc = main(["synth-questions", "--corpus", str(store_root),
               "--source-qid", "fred:TEST", "--n", "3",
               "--horizon-days", "30", "--seed", "2"])
    assert rc == 0
    assert "generated=3" in capsys.readouterr().out
