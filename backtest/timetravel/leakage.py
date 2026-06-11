"""Per-document LLM leak screen. A document that reveals information dated
after as_of is dropped, and every discard is stamped into the evidence
ledger via ACTIVE_CONTEXT — leakage is auditable, never silent. Unparseable
verdicts drop the document (conservative default: a doc we cannot clear is
treated as leaked)."""
from __future__ import annotations

import logging
import re
from datetime import datetime

from backtest.core import LedgerEntry
from backtest.timetravel.asof import ACTIVE_CONTEXT
from backtest.timetravel.llm_cache import CachedLLM

logger = logging.getLogger(__name__)

LEAK_JUDGE_MODEL = "openai/gpt-5.4-mini"     # cheap judge; cached + temp 0
_DOC_CHARS = 1200

_JUDGE_PROMPT = """You are auditing research documents for a forecast made on {as_of}.
For each numbered document below, answer: does the text reveal any
information from after {as_of} (events, outcomes, announcements, or
references that postdate it)?

{docs}

Answer with exactly one line per document, in order, formatted as:
<number>: LEAK
or
<number>: CLEAN
No other text."""


def _render_doc(i: int, doc: dict) -> str:
    date_s = doc.get("date") or "undated"
    text = (doc.get("text") or doc.get("snippet") or "")[:_DOC_CHARS]
    return f"--- Document {i} (dated {date_s}) ---\n{doc.get('title', '')}\n{text}"


async def screen_docs(docs: list[dict], as_of: datetime,
                      llm: CachedLLM) -> list[dict]:
    """Returns the surviving docs; stamps every discard into the active
    context's ledger (when a context is set)."""
    if not docs:
        return []
    rendered = "\n\n".join(_render_doc(i + 1, d) for i, d in enumerate(docs))
    prompt = _JUDGE_PROMPT.format(as_of=as_of.isoformat(), docs=rendered)
    completion = await llm.invoke(LEAK_JUDGE_MODEL, prompt, temperature=0.0,
                                  run_index=0)
    verdicts = {}
    for line in completion.splitlines():
        m = re.match(r"\s*(\d+)\s*:\s*(LEAK|CLEAN)\s*$", line.strip(), re.I)
        if m:
            verdicts[int(m.group(1))] = m.group(2).upper()
    ctx = ACTIVE_CONTEXT.get()
    survivors = []
    for i, doc in enumerate(docs, start=1):
        if verdicts.get(i, "LEAK") == "CLEAN":   # unparseable -> drop
            survivors.append(doc)
            continue
        logger.info("leak screen dropped doc %d: %s", i, doc.get("url"))
        if ctx is not None:
            ctx.record(LedgerEntry(
                provider="leak_screen", query=doc.get("title", ""),
                url=doc.get("url"), doc_date=doc.get("date"),
                date_trust="low",
                note=("discarded: judged to reveal information after "
                      f"{as_of.isoformat()}")))
    return survivors
