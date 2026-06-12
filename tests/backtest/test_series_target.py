"""Task 3: target-series support — full-series reads + CSV importer."""
from datetime import datetime, timezone

from backtest.corpus.series_csv import import_series_csv
from backtest.corpus.store import CorpusStore


def _store(tmp_path):
    return CorpusStore(tmp_path / "corpus")


def test_series_points_sorted_and_filtered(tmp_path):
    store = _store(tmp_path)
    store.put_series("t:1", "target", [
        ("2026-01-03T00:00:00+00:00", 3.0),
        ("2026-01-01T00:00:00+00:00", 1.0),
        ("2026-01-02T00:00:00+00:00", 2.0),
    ])
    pts = store.series_points("t:1", "target")
    assert [v for _, v in pts] == [1.0, 2.0, 3.0]
    until = datetime(2026, 1, 2, tzinfo=timezone.utc)
    assert [v for _, v in store.series_points("t:1", "target", until=until)] \
        == [1.0, 2.0]                                  # inclusive <=


def test_series_points_missing_is_empty(tmp_path):
    assert _store(tmp_path).series_points("t:none", "target") == []


def test_import_series_csv(tmp_path):
    store = _store(tmp_path)
    csv = tmp_path / "s.csv"
    csv.write_text("date,value\n2026-01-01,1.5\n2026-01-02,2.5\nbad,row\n")
    report = import_series_csv(store, csv, qid="csv:test", kind="target")
    assert report.fetched == 2
    assert len(report.errors) == 1
    pts = store.series_points("csv:test", "target")
    assert [v for _, v in pts] == [1.5, 2.5]


def test_import_series_csv_no_header(tmp_path):
    store = _store(tmp_path)
    csv = tmp_path / "s.csv"
    csv.write_text("2026-01-01T12:00:00Z,7\n2026-01-02T12:00:00Z,8\n")
    report = import_series_csv(store, csv, qid="csv:nh", kind="target")
    assert report.fetched == 2 and not report.errors
