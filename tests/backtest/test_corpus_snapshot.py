import json

import backtest.corpus.metaculus as metaculus_mod
from backtest.corpus.snapshot import snapshot_questions
from backtest.corpus.store import CorpusStore

SNAP_DETAIL = {
    "id": 31678,
    "title": "Will X happen before 2027?",
    "status": "resolved",
    "question": {
        "type": "binary",
        "description": "Background paragraph.",
        "resolution_criteria": "Resolves YES if X occurs before 2027.",
        "open_time": "2026-04-01T00:00:00Z",
        "scheduled_close_time": "2026-06-01T00:00:00Z",
        "actual_resolve_time": "2026-06-02T12:00:00Z",
        "resolution": "yes",
        "aggregations": {"recency_weighted": {"history": [
            {"start_time": 1775000000.0, "end_time": None,
             "centers": [0.55]},
        ]}},
    },
}


def test_snapshot_archives_raw_json_and_ingests(tmp_path, monkeypatch):
    monkeypatch.setenv("METACULUS_TOKEN", "tok")

    async def fake_throttle():
        pass

    async def fake_get(url, params=None, headers=None, client=None):
        assert headers == {"Authorization": "Token tok"}
        assert url.endswith("/posts/31678/")
        return SNAP_DETAIL

    monkeypatch.setattr(metaculus_mod, "_throttle", fake_throttle)
    monkeypatch.setattr(metaculus_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    report = snapshot_questions(store, [31678])
    assert report.fetched == 1
    # ingested through the shared metaculus mapping
    q = store.get_question("metaculus:31678")
    assert q is not None and q.resolution is True
    assert q.provenance["snapshot"] is True
    # raw payload archived at full fidelity, permanently
    blobs = list((tmp_path / "blobs" / "snapshots")
                 .glob("metaculus_31678_*.json"))
    assert len(blobs) == 1
    assert json.loads(blobs[0].read_text())["id"] == 31678


def test_snapshot_isolates_per_post_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("METACULUS_TOKEN", "tok")

    async def fake_throttle():
        pass

    async def fake_get(url, params=None, headers=None, client=None):
        if url.endswith("/posts/31678/"):
            return SNAP_DETAIL
        raise RuntimeError("403")

    monkeypatch.setattr(metaculus_mod, "_throttle", fake_throttle)
    monkeypatch.setattr(metaculus_mod, "_get_json", fake_get)
    store = CorpusStore(tmp_path)
    report = snapshot_questions(store, [31678, 99999])
    assert report.fetched == 1
    assert any("99999" in e for e in report.errors)
