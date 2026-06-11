import json
from datetime import datetime

from backtest.corpus.halawi import import_halawi
from backtest.corpus.store import CorpusStore

RECORDS = [
    {"id": "q-101", "question": "Will the 2024 X summit occur?",
     "background": "Verbatim dataset background.",
     "resolution_criteria": "Resolves YES if the summit occurs in 2024.",
     "date_begin": "2024-01-05", "date_close": "2024-06-30",
     "date_resolve_at": "2024-07-02", "resolution": 1.0,
     "source": "metaculus"},
    # resolved BEFORE open -> quarantined (resolution-date validation,
    # spec §3.4: >=3.8% of even the curated set had this defect)
    {"id": "q-102", "question": "Bad dates question?",
     "background": "", "date_begin": "2024-05-01",
     "date_close": "2024-06-01", "date_resolve_at": "2024-04-01",
     "resolution": 0.0, "source": "cset"},
    # no usable resolution -> skipped with error
    {"id": "q-103", "question": "No resolution recorded?",
     "date_begin": "2024-01-01", "date_close": "2024-02-01"},
]


def test_import_halawi_jsonl(tmp_path):
    dataset_dir = tmp_path / "halawi"
    dataset_dir.mkdir()
    with open(dataset_dir / "train.jsonl", "w") as f:
        for r in RECORDS:
            f.write(json.dumps(r) + "\n")
    store = CorpusStore(tmp_path / "corpus")
    report = import_halawi(store, dataset_dir)
    assert report.fetched == 1
    assert len(report.errors) == 2     # bad dates + missing resolution
    q = store.get_question("halawi:q-101")
    assert q.platform == "halawi"
    assert q.qtype.value == "binary"
    assert q.resolution is True
    # dataset text stored verbatim (spec §3.2: text may drift from the
    # current Metaculus copy — provenance carries the origin platform)
    assert q.description == "Verbatim dataset background."
    assert q.open_time == datetime.fromisoformat("2024-01-05")
    assert q.actual_resolve_time == datetime.fromisoformat("2024-07-02")
    assert q.provenance["source"] == "halawi"
    assert q.provenance["origin_platform"] == "metaculus"


def test_import_halawi_json_array(tmp_path):
    dataset_dir = tmp_path / "halawi"
    dataset_dir.mkdir()
    (dataset_dir / "test.json").write_text(json.dumps([RECORDS[0]]))
    store = CorpusStore(tmp_path / "corpus")
    report = import_halawi(store, dataset_dir)
    assert report.fetched == 1
    assert report.errors == []


def test_import_halawi_zero_resolution_is_false(tmp_path):
    dataset_dir = tmp_path / "halawi"
    dataset_dir.mkdir()
    rec = dict(RECORDS[0], id="q-110", resolution=0.0)
    (dataset_dir / "one.json").write_text(json.dumps(rec))
    store = CorpusStore(tmp_path / "corpus")
    import_halawi(store, dataset_dir)
    assert store.get_question("halawi:q-110").resolution is False


def test_import_halawi_non_binary_numeric_resolution_is_skipped(tmp_path):
    """A record with resolution=0.5 (ambiguous/annulled) must be SKIPPED.

    The importer must not store it as a confident True/False outcome.
    An error mentioning 'resolution' must be appended instead.
    """
    dataset_dir = tmp_path / "halawi"
    dataset_dir.mkdir()
    rec = dict(RECORDS[0], id="q-half", resolution=0.5)
    (dataset_dir / "one.json").write_text(json.dumps(rec))
    store = CorpusStore(tmp_path / "corpus")
    report = import_halawi(store, dataset_dir)

    # Record must NOT be stored
    assert store.get_question("halawi:q-half") is None
    # Must have been counted as an error, not as fetched
    assert report.fetched == 0
    assert len(report.errors) == 1
    assert "resolution" in report.errors[0].lower()
