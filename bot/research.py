"""Multi-source research compositor. Every configured provider runs in
PARALLEL and every success becomes a section of one composed report —
distinct-source count was the #1 Fall-2025 winner predictor (r=0.42, p=0.006),
so breadth is the point, not redundancy. A research failure must NEVER cost us
the question: any subset of providers may fail (logged, skipped); worst case
we forecast on model knowledge with an empty-research flag."""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import date

from bot.config import BotConfig
from bot.prompts import RESEARCH_PROMPT_CURRENT, RESEARCH_PROMPT_HISTORICAL

logger = logging.getLogger(__name__)

_SECTION_TITLES = {
    "asknews": "News search (AskNews)",
    "search_llm": "Research assistant (search LLM)",
    "markets": "Prediction-market signals",
}


@dataclass
class ResearchResult:
    text: str
    provider: str            # comma-joined successful providers, or "none"
    providers: tuple = ()    # successful providers, in configured order


def _prompt_args(question: dict, cfg: BotConfig) -> dict:
    return dict(
        question_text=question["question_text"],
        resolution_criteria=question.get("resolution_criteria", ""),
        fine_print=question.get("fine_print", ""),
        today=(cfg.as_of.date() if cfg.as_of else date.today()).isoformat(),
    )


async def _asknews_research(question: dict, cfg: BotConfig) -> str:
    # AskNewsSearcher accepts either OAuth (client id + secret) or an API key
    has_oauth = bool(os.getenv("ASKNEWS_CLIENT_ID")
                     and os.getenv("ASKNEWS_SECRET"))
    if not (has_oauth or os.getenv("ASKNEWS_API_KEY")):
        raise RuntimeError("AskNews credentials not configured")
    from forecasting_tools import AskNewsSearcher
    searcher = AskNewsSearcher()
    return await searcher.get_formatted_news_async(question["question_text"])


async def _search_llm_research(question: dict, cfg: BotConfig) -> str:
    from forecasting_tools import GeneralLlm
    llm = GeneralLlm(model=f"openrouter/{cfg.search_model}", temperature=0.1)
    args = _prompt_args(question, cfg)
    # return_exceptions so one track's failure neither discards the other
    # track's report nor orphans its still-running task
    current, historical = await asyncio.gather(
        llm.invoke(RESEARCH_PROMPT_CURRENT.format(**args)),
        llm.invoke(RESEARCH_PROMPT_HISTORICAL.format(**args)),
        return_exceptions=True)
    parts = []
    if isinstance(historical, BaseException):
        logger.warning("search-LLM historical track failed: %s", historical)
    else:
        parts.append(f"### Outside view (historical)\n{historical}")
    if isinstance(current, BaseException):
        logger.warning("search-LLM current track failed: %s", current)
    else:
        parts.append(f"### Inside view (current)\n{current}")
    if not parts:
        raise RuntimeError(
            f"both search-LLM tracks failed: {historical}; {current}")
    return "\n\n".join(parts)


async def _markets_research(question: dict, cfg: BotConfig) -> str:
    from bot.markets import market_research
    return await market_research(question["question_text"])


async def run_research(question: dict, cfg: BotConfig) -> ResearchResult:
    # late-bind through module attributes so tests can monkeypatch; a typo'd
    # provider name in config must not break run_research's never-raise
    # contract for every question in the run
    names, fns = [], []
    for name in cfg.research_providers:
        fn = globals().get(f"_{name}_research")
        if fn is None:
            logger.error("unknown research provider %r — skipping", name)
            continue
        names.append(name)
        fns.append(fn)

    async def guarded(name, fn):
        return await asyncio.wait_for(
            fn(question, cfg), timeout=cfg.research_timeout_s)

    results = await asyncio.gather(
        *(guarded(n, f) for n, f in zip(names, fns)), return_exceptions=True)

    sections, ok = [], []
    for name, r in zip(names, results):
        if isinstance(r, BaseException):
            logger.warning("research provider %s failed: %s", name, r)
            continue
        if not (r and r.strip()):
            logger.warning("research provider %s returned nothing", name)
            continue
        title = _SECTION_TITLES.get(name, name)
        sections.append(f"## {title}\n{r.strip()}")
        ok.append(name)

    if not ok:
        logger.error("ALL research providers failed for question: %s",
                     question.get("question_text", "")[:80])
        return ResearchResult(text="", provider="none", providers=())
    return ResearchResult(text="\n\n".join(sections),
                          provider=",".join(ok), providers=tuple(ok))
