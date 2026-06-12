"""GABRIEL-style evidence rating (incorporation plan item 3).

Methodology from OpenAI's GABRIEL / "GPT as a Measurement Tool" (NBER
w34834): natural-language attributes scored 0-100 by an LLM, with repeated
sampling as a reliability signal ("consensus columns"). Implemented in-repo
rather than via the openai-gabriel package because every call must route
through the caller's LLM seam (CachedLLM in backtests — cached, cost-capped,
reproducible; OpenRouter live).

The first production use is the Halawi-style relevance filter the research
pipeline never had: rate dossier items on relevance-to-resolution-criteria,
drop the noise (naive news dumps measurably hurt — ForecastBench).

invoke contract: ``async invoke(prompt: str, sample_idx: int) -> str``.
Callers map sample_idx onto their cache's run_index so consensus samples
stay distinct. Per-item failures NEVER raise — a failed rating is None and
filtering fails open (an unrated item is kept).
"""
from __future__ import annotations

import asyncio
import json
import logging
import statistics
from dataclasses import dataclass

logger = logging.getLogger(__name__)

RATE_PROMPT = """You are scoring one piece of evidence for a forecasting question.

QUESTION: {question_text}
RESOLUTION CRITERIA: {resolution_criteria}

EVIDENCE ITEM:
{item}

Score the item 0-100 on each attribute:
{attribute_lines}

Scoring guide: 0 = not at all, 50 = somewhat, 100 = maximally. Judge against
the RESOLUTION CRITERIA, not the question's general topic.

Respond with ONLY a JSON object, no prose:
{{"scores": {{{attribute_keys}}}}}"""

ATTRIBUTE_RUBRICS = {
    "relevance": ("bears directly on whether the resolution criteria will "
                  "be met (not merely same broad topic)"),
    "reliability": ("source and specificity suggest the claim is "
                    "trustworthy (named sources, primary data)"),
    "recency_sensitivity": ("the item reports a NEW development that should "
                            "update a forecast, vs background"),
}


@dataclass(frozen=True)
class ItemRating:
    scores: dict[str, float]          # attribute -> mean score (0-100)
    std: dict[str, float]             # attribute -> std across samples


def _parse_scores(text: str, attributes: tuple[str, ...]) -> dict | None:
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        scores = json.loads(text[start:end])["scores"]
        return {a: float(scores[a]) for a in attributes}
    except (ValueError, KeyError, TypeError):
        return None


def _rate_prompt(item: str, attributes: tuple[str, ...],
                 question_ctx: dict) -> str:
    lines = "\n".join(
        f"- {a}: {ATTRIBUTE_RUBRICS.get(a, a)}" for a in attributes)
    keys = ", ".join(f'"{a}": <0-100>' for a in attributes)
    return RATE_PROMPT.format(
        question_text=question_ctx.get("question_text", ""),
        resolution_criteria=question_ctx.get("resolution_criteria", ""),
        item=item, attribute_lines=lines, attribute_keys=keys)


async def rate_items(items: list[str], attributes: tuple[str, ...],
                     question_ctx: dict, invoke,
                     n_samples: int = 1) -> list[ItemRating | None]:
    """Rate every item on every attribute; n_samples independent samples per
    item are averaged (mean) with their std as the agreement signal."""

    async def rate_one(item: str) -> ItemRating | None:
        prompt = _rate_prompt(item, attributes, question_ctx)
        samples: list[dict] = []
        for s in range(n_samples):
            try:
                text = await invoke(prompt, s)
            except Exception as e:
                logger.warning("rating invoke failed: %s", e)
                continue
            parsed = _parse_scores(text, attributes)
            if parsed is not None:
                samples.append(parsed)
        if not samples:
            return None
        scores = {a: statistics.fmean(s[a] for s in samples)
                  for a in attributes}
        std = {a: (statistics.pstdev([s[a] for s in samples])
                   if len(samples) > 1 else 0.0) for a in attributes}
        return ItemRating(scores=scores, std=std)

    return list(await asyncio.gather(*(rate_one(i) for i in items)))


def split_dossier_items(text: str) -> tuple[str, list[str]]:
    """Split a provider dossier into (header, bullet items). Items are the
    '- ...' lines our providers emit one-per-article; everything before the
    first bullet is the header. Rejoining header + items + trailing newline
    reproduces the input for bullet-list dossiers."""
    lines = text.splitlines()
    items: list[str] = []
    header_lines: list[str] = []
    for line in lines:
        if line.lstrip().startswith("- "):
            items.append(line)
        elif not items:
            header_lines.append(line)
        else:                          # continuation of a wrapped bullet
            items[-1] = items[-1] + "\n" + line
    header = "\n".join(header_lines)
    if header and items:
        header += "\n"
    return header, items


async def filter_dossier(text: str, question_ctx: dict, invoke,
                         threshold: float = 40.0,
                         n_samples: int = 1) -> tuple[str, list]:
    """Relevance-filter a bullet-list dossier: items rated below threshold
    on 'relevance' are dropped; unrated (failed) items are KEPT (fail-open —
    losing evidence is worse than keeping noise). Non-bullet dossiers pass
    through untouched. Returns (filtered_text, ratings)."""
    header, items = split_dossier_items(text)
    if not items:
        return text, []
    ratings = await rate_items(items, ("relevance",), question_ctx, invoke,
                               n_samples=n_samples)
    kept = [it for it, r in zip(items, ratings)
            if r is None or r.scores["relevance"] >= threshold]
    if not kept:
        return "", ratings
    note = (f"(relevance-filtered: kept {len(kept)}/{len(items)} items at "
            f"threshold {threshold:.0f})\n")
    return header + note + "\n".join(kept) + "\n", ratings
