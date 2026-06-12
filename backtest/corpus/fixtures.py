"""Loader for the committed mini-corpus fixtures (CI smoke gate).

Fixture format — one JSON file per question:

    {
      "question": { ...BacktestQuestion fields; datetimes as iso8601
                    strings; qtype as the QType enum value... },
      "cp_series": [[iso_timestamp, value], ...]    # optional
    }

Also runnable as a module to build a store on disk (used by the CI
workflow):  python -m backtest.corpus.fixtures <fixtures_dir> <store_root>
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from backtest.core import BacktestQuestion, QType
from backtest.corpus.store import CorpusStore

_DT_FIELDS = ("open_time", "scheduled_close_time", "spot_scoring_time",
              "actual_resolve_time")


def _question_from_fixture(d: dict) -> BacktestQuestion:
    kw = dict(d)
    kw["qtype"] = QType(kw["qtype"])
    for f in _DT_FIELDS:
        if kw.get(f):
            kw[f] = datetime.fromisoformat(kw[f])
    kw["options"] = tuple(kw.get("options") or ())
    return BacktestQuestion(**kw)


def load_fixture_corpus(store: CorpusStore, fixtures_dir: Path) -> int:
    """Load every *.json fixture into `store`; returns the count loaded."""
    n = 0
    for path in sorted(Path(fixtures_dir).glob("*.json")):
        blob = json.loads(path.read_text())
        q = _question_from_fixture(blob["question"])
        store.upsert_question(q)
        series = blob.get("cp_series") or []
        if series:
            store.put_series(q.qid, "cp", [(ts, v) for ts, v in series])
        n += 1
    return n


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Load committed mini-corpus fixtures into a CorpusStore")
    ap.add_argument("fixtures_dir", type=Path)
    ap.add_argument("store_root", type=Path)
    args = ap.parse_args()
    count = load_fixture_corpus(CorpusStore(args.store_root), args.fixtures_dir)
    print(f"loaded {count} fixture questions into {args.store_root}")
