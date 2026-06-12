"""Clio-style failure clustering over a run's worst losses (plan item 3).

Anthropic's Clio pipeline (arXiv 2412.13678) at post-mortem scale: facet
extraction per losing forecast -> one clustering pass over the facet
summaries -> labeled failure taxonomy. No embeddings dependency — at N<=100
losses a single LLM clustering call is cheaper and auditable. Automates the
manual run-1 post-mortem (research-notes/09) whose taxonomy seeds the cause
labels, and feeds scripts/playbook_update.py.

All LLM calls go through the provided llm (CachedLLM in the CLI — replay
mode by default, so re-rendering taxonomy reports is $0).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

CAUSES = ("research-miss", "elicitation", "aggregation", "calibration",
          "genuinely-hard", "data-defect", "other")

FACET_PROMPT = """You are post-morteming one losing forecast from a forecasting bot.

QUESTION: {title}
RESOLUTION CRITERIA: {resolution_criteria}
RESOLVED: {resolution}
BOT FINAL PROBABILITY: {final}
ENSEMBLE MEMBER PROBABILITIES: {member_probs}
MARKET PRICE AT FORECAST TIME: {price}
RESEARCH DOSSIER EXCERPT:
{research}

Diagnose the loss. Respond with ONLY a JSON object:
{{"summary": "<one sentence: why the bot was wrong>",
  "primary_cause": "<one of: {causes}>",
  "evidence_was_available": <true if the dossier contained evidence that,
   weighed correctly, points to the actual outcome>}}"""

CLUSTER_PROMPT = """Group these forecasting-failure summaries into at most {max_clusters}
clusters of shared failure mode. Every index must appear in exactly one
cluster.

SUMMARIES:
{summaries}

Respond with ONLY a JSON object:
{{"clusters": [{{"label": "<short name>", "description": "<one sentence>",
  "members": [<indices>]}}]}}"""


def _latest_artifacts(run_dir: Path) -> dict:
    from backtest.core import artifact_from_dict
    latest: dict = {}
    path = run_dir / "artifacts.jsonl"
    if not path.exists():
        return latest
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            a = artifact_from_dict(json.loads(line))
            latest[(a.qid, a.horizon, a.system_name)] = a
    return latest


def _stage_value(artifact, stage: str, field: str, key: str):
    for s in (artifact.stages if artifact else []):
        if s.stage == stage:
            val = getattr(s, field).get(key)
            if val is not None:
                return val
    return None


def _parse_json(text: str) -> dict | None:
    try:
        return json.loads(text[text.index("{"):text.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return None


async def analyze_failures(run_dir: Path, store, llm, model: str, *,
                           top: int = 25, variant: str | None = None,
                           max_clusters: int = 8) -> dict:
    """Returns the failures payload and writes failures.json + failures.md
    into run_dir."""
    run_dir = Path(run_dir)
    rows = []
    scores_path = run_dir / "scores.jsonl"
    if scores_path.exists():
        rows = [json.loads(line)
                for line in scores_path.read_text().splitlines()
                if line.strip()]
    if variant is not None:
        rows = [r for r in rows if r["variant"] == variant]
    rows = [r for r in rows if r.get("baseline_score") is not None]
    rows.sort(key=lambda r: r["baseline_score"])
    worst = rows[:top]
    artifacts = _latest_artifacts(run_dir)

    facets = []
    for r in worst:
        a = artifacts.get((r["qid"], r["horizon"], r["variant"]))
        q = store.get_question(r["qid"]) if store is not None else None
        final = None
        if a is not None and a.forecast is not None:
            final = a.forecast.binary_prob
        member_probs = _stage_value(a, "aggregation", "inputs",
                                    "member_probs")
        research = (_stage_value(a, "research", "outputs", "text") or "")
        prompt = FACET_PROMPT.format(
            title=(q.title if q else r["qid"]),
            resolution_criteria=(q.resolution_criteria if q else ""),
            resolution=r.get("resolution"),
            final=("?" if final is None else f"{final:.3f}"),
            member_probs=member_probs,
            price=r.get("price_at_T"),
            research=research[:1200],
            causes=", ".join(CAUSES))
        facet = None
        try:
            facet = _parse_json(await llm.invoke(model, prompt, 0.0,
                                                 run_index=0))
        except Exception as e:
            logger.warning("facet extraction failed for %s: %s",
                           r["qid"], e)
        if not facet or "summary" not in facet:
            facet = {"summary": f"unparsed facet for {r['qid']}",
                     "primary_cause": "unparsed",
                     "evidence_was_available": None}
        facet.update({
            "qid": r["qid"], "variant": r["variant"],
            "horizon": r["horizon"],
            "baseline_score": r["baseline_score"],
            "final": final, "resolution": r.get("resolution"),
            "price_at_T": r.get("price_at_T"),
            # wrong side of the market = lost the head-to-head (code, not LLM)
            "wrong_side_of_market": (
                r["h2h_market"] < 0
                if r.get("h2h_market") is not None else None),
        })
        facets.append(facet)

    clusters = []
    if facets:
        numbered = "\n".join(f"{i}: {f['summary']}"
                             for i, f in enumerate(facets))
        parsed = None
        try:
            parsed = _parse_json(await llm.invoke(
                model, CLUSTER_PROMPT.format(
                    max_clusters=max_clusters, summaries=numbered),
                0.0, run_index=0))
        except Exception as e:
            logger.warning("clustering failed: %s", e)
        if parsed and isinstance(parsed.get("clusters"), list):
            clusters = parsed["clusters"]
        else:
            clusters = [{"label": "unclustered",
                         "description": "clustering call failed",
                         "members": list(range(len(facets)))}]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir), "variant": variant, "model": model,
        "n_failures": len(facets), "facets": facets, "clusters": clusters,
    }
    (run_dir / "failures.json").write_text(json.dumps(payload, indent=2))
    (run_dir / "failures.md").write_text(_render_md(payload))
    return payload


def _render_md(p: dict) -> str:
    lines = [f"# Failure taxonomy — {p['run_dir']}",
             f"*{p['n_failures']} worst losses"
             + (f", variant {p['variant']}" if p["variant"] else "")
             + f"; facet/cluster model {p['model']}.*", ""]
    for c in p["clusters"]:
        members = c.get("members") or []
        lines.append(f"## {c.get('label', '?')} ({len(members)})")
        lines.append(c.get("description", ""))
        for i in members:
            if 0 <= i < len(p["facets"]):
                f = p["facets"][i]
                side = (" — wrong side of market"
                        if f.get("wrong_side_of_market") else "")
                lines.append(
                    f"- `{f['qid']}` ({f['baseline_score']:.1f} pts, "
                    f"cause: {f.get('primary_cause')}){side}: "
                    f"{f['summary']}")
        lines.append("")
    return "\n".join(lines)
