"""Two-track research (outside-view/historical + inside-view/current) with a
provider fallback chain. A research failure must NEVER cost us the question:
worst case we forecast on model knowledge with an empty-research flag.
Distinct-source count was the #1 Fall-2025 winner predictor."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date

from bot.config import BotConfig
from bot.prompts import RESEARCH_PROMPT_CURRENT, RESEARCH_PROMPT_HISTORICAL

logger = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    text: str
    provider: str          # "asknews" | "search_llm" | "none"


def _prompt_args(question: dict) -> dict:
    return dict(
        question_text=question["question_text"],
        resolution_criteria=question.get("resolution_criteria", ""),
        fine_print=question.get("fine_print", ""),
        today=date.today().isoformat(),
    )


async def _asknews_research(question: dict, cfg: BotConfig) -> str:
    if not (os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET")):
        raise RuntimeError("AskNews credentials not configured")
    from forecasting_tools import AskNewsSearcher
    searcher = AskNewsSearcher()
    return await searcher.get_formatted_news_async(question["question_text"])


async def _search_llm_research(question: dict, cfg: BotConfig) -> str:
    from forecasting_tools import GeneralLlm
    llm = GeneralLlm(model=f"openrouter/{cfg.search_model}", temperature=0.1)
    args = _prompt_args(question)
    current = await llm.invoke(RESEARCH_PROMPT_CURRENT.format(**args))
    historical = await llm.invoke(RESEARCH_PROMPT_HISTORICAL.format(**args))
    return (f"## Outside view (historical)\n{historical}\n\n"
            f"## Inside view (current)\n{current}")


async def run_research(question: dict, cfg: BotConfig) -> ResearchResult:
    for name in cfg.research_providers:
        fn = globals()[f"_{name}_research"]   # module attr so tests can patch
        try:
            text = await fn(question, cfg)
            if text and text.strip():
                return ResearchResult(text=text, provider=name)
        except Exception as e:
            logger.warning("research provider %s failed: %s", name, e)
    logger.error("ALL research providers failed for question: %s",
                 question.get("question_text", "")[:80])
    return ResearchResult(text="", provider="none")
