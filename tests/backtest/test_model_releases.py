from dataclasses import replace
from datetime import datetime, timezone

import backtest.corpus.model_releases as mr
from backtest.core import BacktestQuestion, QType
from backtest.corpus.store import CorpusStore

DATES = {
    "openai/gpt-5.4": "2026-02-10",
    "anthropic/claude-opus-4-6": "2026-01-20",
    "google/gemini-3.1-pro-preview": "2026-03-05",
}

MODEL_SET = list(DATES)


def _q(qid, open_time):
    return BacktestQuestion(
        qid=qid, platform="metaculus", native_id=qid.split(":")[1],
        qtype=QType.BINARY, title="T?", open_time=open_time,
        resolution=True)


def test_release_table_covers_current_ensemble_slugs():
    # the three slugs of bot/config.py:_default_ensemble()
    for slug in MODEL_SET:
        assert slug in mr.RELEASE_DATES
        datetime.fromisoformat(mr.RELEASE_DATES[slug])   # valid iso date


def test_model_set_hash_deterministic_and_order_independent(monkeypatch):
    monkeypatch.setattr(mr, "RELEASE_DATES", dict(DATES))
    h1 = mr.model_set_hash(MODEL_SET)
    h2 = mr.model_set_hash(list(reversed(MODEL_SET)))
    assert h1 == h2
    assert len(h1) == 12 and all(c in "0123456789abcdef" for c in h1)
    # hash changes when a release date changes
    monkeypatch.setattr(mr, "RELEASE_DATES",
                        dict(DATES, **{"openai/gpt-5.4": "2026-04-01"}))
    assert mr.model_set_hash(MODEL_SET) != h1
    # unknown slugs are total: deterministic hash, no exception
    monkeypatch.setattr(mr, "RELEASE_DATES", dict(DATES))
    h_u = mr.model_set_hash(MODEL_SET + ["vendor/unlisted-model"])
    assert h_u == mr.model_set_hash(MODEL_SET + ["vendor/unlisted-model"])
    assert len(h_u) == 12


def test_stratify_clean_vs_contaminated(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "RELEASE_DATES", dict(DATES))
    store = CorpusStore(tmp_path)
    h = mr.model_set_hash(MODEL_SET)
    # opened AFTER the latest release date (2026-03-05) -> clean
    clean_q = _q("metaculus:1",
                 datetime(2026, 4, 1, tzinfo=timezone.utc))
    assert mr.stratify(store, clean_q, MODEL_SET) == "clean"
    assert store.get_contamination("metaculus:1", h) == "clean"
    # opened BEFORE the latest release date -> contaminated
    dirty_q = _q("metaculus:2",
                 datetime(2026, 2, 1, tzinfo=timezone.utc))
    assert mr.stratify(store, dirty_q, MODEL_SET) == "contaminated"
    assert store.get_contamination("metaculus:2", h) == "contaminated"
    # unknown open_time -> conservative: contaminated
    unknown_q = replace(_q("metaculus:3", None))
    assert mr.stratify(store, unknown_q, MODEL_SET) == "contaminated"
