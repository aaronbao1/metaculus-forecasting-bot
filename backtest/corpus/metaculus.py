"""Tier-aware Metaculus fetcher (spec §3.2 source 1).

Post-March-2026 lockdown: a plain token sees IDs/titles/types/timestamps
but description/resolution/aggregations come back null — EXCEPT questions
this bot ever forecast, which stay fully visible. This fetcher stores what
is visible, records withheld fields in provenance["withheld"] and
FetchReport.withheld, and simply re-fetching after the Bot Benchmarking
Access Tier / Data Needs Form approval lands upgrades rows in place
(upsert_question replaces).

Enumeration: GET /api/posts/ with statuses/forecast_type/tournaments
params, limit<=100 offset pagination. Detail: GET /api/posts/{id}/.
Auth: "Authorization: Token <METACULUS_TOKEN>". Self-throttle ~3.5s +
jitter between requests (forecasting-tools convention).

CP history: aggregations.recency_weighted.history interval entries
(start_time <= t < end_time) are flattened to (start_time, value) step
points — CorpusStore.value_at's last-point-<=-t lookup then reproduces
CP-at-T (matches forecasting-tools get_cp_at_time semantics).
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
from datetime import datetime, timezone

from backtest.core import BacktestQuestion, QType
from backtest.corpus.store import CorpusStore, FetchReport

logger = logging.getLogger(__name__)

API_BASE = "https://www.metaculus.com/api"
PAGE_SIZE = 100
THROTTLE_S = 3.5
HTTP_TIMEOUT_S = 30.0


def _auth_headers() -> dict:
    token = os.environ.get("METACULUS_TOKEN", "").strip()
    if not token:
        raise RuntimeError("METACULUS_TOKEN is not set")
    return {"Authorization": f"Token {token}"}


async def _throttle() -> None:
    await asyncio.sleep(THROTTLE_S + random.uniform(0.0, 0.5))


async def _get_json(url: str, params: dict | None = None,
                    headers: dict | None = None, client=None):
    import httpx
    if client is not None:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S,
                                 follow_redirects=True) as one_shot:
        resp = await one_shot.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()


def _parse_time(v) -> datetime | None:
    if v in (None, ""):
        return None
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v, tz=timezone.utc)
    return datetime.fromisoformat(str(v).replace("Z", "+00:00"))


def _cp_points(history: list[dict],
               options: tuple = ()) -> list[tuple[str, object]]:
    """Flatten recency_weighted.history intervals to step points.
    MC questions (len(values) == len(options)) store an option->prob dict."""
    pts: list[tuple[str, object]] = []
    for entry in history or []:
        start = entry.get("start_time")
        values = entry.get("centers") or entry.get("means")
        if start is None or not values:
            continue
        if len(values) == 1:
            value = values[0]
        elif options and len(values) == len(options):
            value = dict(zip(options, values))
        else:
            value = list(values)
        pts.append((_parse_time(start).isoformat(), value))
    return pts


def _ingest_detail(store: CorpusStore, post: dict,
                   report: FetchReport) -> str | None:
    """Map one /api/posts/{id}/ payload into the store. Returns qid."""
    q = post.get("question") or {}
    if not q.get("type"):
        report.errors.append(f"post {post.get('id')}: no question payload")
        return None
    qid = f"metaculus:{post['id']}"
    open_time = _parse_time(q.get("open_time"))
    actual_resolve_time = _parse_time(q.get("actual_resolve_time"))
    # resolution-date validation at import (spec §3.4)
    if (open_time and actual_resolve_time
            and actual_resolve_time < open_time):
        report.errors.append(
            f"quarantined {qid}: resolve_time < open_time")
        return None
    withheld: list[str] = []

    description = q.get("description")
    if description is None:          # withheld by tier, not missing
        withheld.append("description")
        description = ""

    resolution = q.get("resolution")
    is_resolved = (post.get("status") == "resolved"
                   or actual_resolve_time is not None)
    if resolution is None and is_resolved:
        withheld.append("resolution")
    qtype = QType(q["type"])
    if qtype is QType.BINARY and isinstance(resolution, str):
        # "annulled"/"ambiguous" stay None (unscoreable, not withheld)
        resolution = {"yes": True, "no": False}.get(resolution.lower())
    elif qtype in (QType.NUMERIC, QType.DISCRETE) \
            and resolution is not None:
        try:
            resolution = float(resolution)
        except (TypeError, ValueError):
            pass    # "above_upper_bound"/"below_lower_bound" stay strings

    history = ((q.get("aggregations") or {})
               .get("recency_weighted") or {}).get("history")
    if history is None:
        withheld.append("aggregations")

    for f in withheld:
        report.withheld[f] = report.withheld.get(f, 0) + 1

    store.upsert_question(BacktestQuestion(
        qid=qid, platform="metaculus", native_id=str(post["id"]),
        qtype=qtype, title=post.get("title") or q.get("title") or "",
        description=description,
        resolution_criteria=q.get("resolution_criteria") or "",
        fine_print=q.get("fine_print") or "",
        open_time=open_time,
        scheduled_close_time=_parse_time(q.get("scheduled_close_time")),
        spot_scoring_time=_parse_time(q.get("spot_scoring_time")),
        actual_resolve_time=actual_resolve_time,
        resolution=resolution,
        options=tuple(q.get("options") or ()),
        scaling=q.get("scaling"),
        question_weight=float(q.get("question_weight") or 1.0),
        provenance={"source": "metaculus",
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "withheld": withheld}))
    if history:
        pts = _cp_points(history, tuple(q.get("options") or ()))
        if pts:
            store.put_series(qid, "cp", pts)
    report.fetched += 1
    return qid


async def fetch_metaculus(store: CorpusStore, *, tournaments=None,
                          statuses="resolved",
                          limit=None) -> FetchReport:
    headers = _auth_headers()
    report = FetchReport()
    params: dict = {"statuses": statuses, "limit": PAGE_SIZE, "offset": 0,
                    "include_descriptions": "true"}
    if tournaments:
        params["tournaments"] = ",".join(tournaments)

    post_ids: list[int] = []
    while True:
        page = await _get_json(f"{API_BASE}/posts/", params=params,
                               headers=headers)
        for post in page.get("results") or []:
            post_ids.append(post["id"])
            if limit is not None and len(post_ids) >= limit:
                break
        if limit is not None and len(post_ids) >= limit:
            break
        if not page.get("next"):
            break
        params = dict(params, offset=params["offset"] + PAGE_SIZE)
        await _throttle()

    for post_id in post_ids:
        await _throttle()
        try:
            detail = await _get_json(f"{API_BASE}/posts/{post_id}/",
                                     headers=headers)
            _ingest_detail(store, detail, report)
        except Exception as e:     # per-question isolation (spec §7)
            logger.warning("metaculus post %s failed: %s", post_id, e)
            report.errors.append(f"post {post_id}: {e}")
    return report
