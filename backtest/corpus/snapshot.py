"""Forward snapshot accumulator (spec §3.2 item 4 / extension 1).

Every question this bot's token ever forecast stays fully visible to it
forever — so archiving every live-touched question at full fidelity turns
the live bot into a corpus-building machine and decays the post-lockdown
corpus problem over months.

Reuses backtest/corpus/metaculus.py internals (auth, throttle, _get_json,
_ingest_detail) via module attributes — monkeypatching metaculus's
attributes in tests covers this module too. Stores the RAW detail JSON
under blobs/snapshots/ in addition to the normalized store row.

Sync entry point (contract): wraps asyncio.run, so the Phase-5 hook at the
end of scripts/run_tournament.py can call it from synchronous teardown.
Do not call from inside a running event loop.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timezone

from backtest.corpus import metaculus as metaculus_mod
from backtest.corpus.store import CorpusStore, FetchReport


def snapshot_questions(store: CorpusStore,
                       post_ids: list[int]) -> FetchReport:
    return asyncio.run(_snapshot_async(store, post_ids))


async def _snapshot_async(store: CorpusStore,
                          post_ids: list[int]) -> FetchReport:
    headers = metaculus_mod._auth_headers()
    report = FetchReport()
    snap_dir = store.blobs / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    for post_id in post_ids:
        await metaculus_mod._throttle()
        try:
            detail = await metaculus_mod._get_json(
                f"{metaculus_mod.API_BASE}/posts/{post_id}/",
                headers=headers)
        except Exception as e:      # per-question isolation
            report.errors.append(f"post {post_id}: {e}")
            continue
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (snap_dir / f"metaculus_{post_id}_{stamp}.json").write_text(
            json.dumps(detail))
        qid = metaculus_mod._ingest_detail(store, detail, report)
        if qid:
            q = store.get_question(qid)
            store.upsert_question(replace(
                q, provenance=dict(q.provenance, snapshot=True)))
    return report
