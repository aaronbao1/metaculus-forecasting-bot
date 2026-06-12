"""Playbook reflector: failure facets -> candidate playbook deltas
(ACE-style Reflector/Curator, plan item 5 — limited scope).

Reads a failures.json produced by `backtest failures`, asks a cheap
reflection model for GENERALIZABLE lessons (never question-specific facts),
dedupes against the existing playbook, and appends survivors to the
`## Candidates` section of skills/forecasting-playbook.md with provenance.

NEVER touches `## Promoted` — promotion is a manual edit gated by an
experiments-gate pass (see the playbook header).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path

_repo_root = str(_Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
del _repo_root

import asyncio
import hashlib
import json
import re
from datetime import date
from pathlib import Path

REFLECT_PROMPT = """You are curating a forecasting playbook from post-mortems of a bot's losing
forecasts. Below are failure clusters and per-question diagnoses.

{failures}

Propose at most {max_new} playbook bullets. Each must be a GENERALIZABLE
procedure or check a forecaster should apply to FUTURE questions — never a
fact about a specific past question, and never a vague platitude.

Respond with ONLY a JSON object:
{{"bullets": [{{"lesson": "<imperative one-sentence rule>",
   "applies_to": "<question domain or failure mode it targets>"}}]}}"""


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _existing_bullets(playbook_text: str) -> set[str]:
    out = set()
    for line in playbook_text.splitlines():
        if line.lstrip().startswith("- "):
            bare = re.sub(r"<!--.*?-->", "", line).strip()[2:]
            out.add(_normalize(bare.split("(applies to:")[0]))
    return out


def _failures_digest(payload: dict) -> str:
    parts = []
    for c in payload.get("clusters", []):
        parts.append(f"CLUSTER {c.get('label')}: {c.get('description')}")
    for f in payload.get("facets", []):
        parts.append(f"- [{f.get('primary_cause')}] {f.get('summary')} "
                     f"(score {f.get('baseline_score')})")
    return "\n".join(parts)


async def update_playbook(failures: dict, playbook_path, invoke, *,
                          max_new: int = 5,
                          today: str | None = None) -> int:
    """Append deduped candidate bullets; returns the number added."""
    playbook_path = Path(playbook_path)
    text = playbook_path.read_text()
    if "## Candidates" not in text:
        text = text.rstrip() + "\n\n## Candidates\n"
    existing = _existing_bullets(text)

    prompt = REFLECT_PROMPT.format(failures=_failures_digest(failures),
                                   max_new=max_new)
    raw = await invoke(prompt, 0)
    try:
        proposals = json.loads(
            raw[raw.index("{"):raw.rindex("}") + 1])["bullets"]
    except (ValueError, KeyError):
        return 0

    source = failures.get("run_dir", "unknown-run")
    stamp = today or date.today().isoformat()
    added = []
    for b in proposals[:max_new]:
        lesson = (b.get("lesson") or "").strip()
        if not lesson or _normalize(lesson) in existing:
            continue
        existing.add(_normalize(lesson))
        bullet_id = "pb-" + hashlib.sha256(
            _normalize(lesson).encode()).hexdigest()[:8]
        applies = (b.get("applies_to") or "").strip()
        suffix = f" (applies to: {applies})" if applies else ""
        added.append(f"- {lesson}{suffix} "
                     f"<!-- id:{bullet_id} source:{source} date:{stamp} -->")
    if added:
        playbook_path.write_text(text.rstrip() + "\n"
                                 + "\n".join(added) + "\n")
    return len(added)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="playbook-update",
        description="propose candidate playbook bullets from failures.json")
    p.add_argument("failures_json")
    p.add_argument("--playbook", default="skills/forecasting-playbook.md")
    p.add_argument("--max-new", type=int, default=5)
    p.add_argument("--model", default=None,
                   help="reflection model (default: cfg.parser_model)")
    p.add_argument("--cache-dir", default=None,
                   help="LLM cache dir (default: <run_dir>/../llm_cache)")
    p.add_argument("--live", action="store_true",
                   help="allow fresh LLM calls (default: replay-only)")
    args = p.parse_args(argv)

    failures_path = Path(args.failures_json)
    if not failures_path.exists():
        print(f"failures file not found: {failures_path}")
        return 2
    failures = json.loads(failures_path.read_text())

    from backtest.timetravel.llm_cache import CachedLLM
    from bot.config import BotConfig
    cache_dir = (Path(args.cache_dir) if args.cache_dir
                 else failures_path.parent.parent / "llm_cache")
    llm = CachedLLM(cache_dir, live=args.live)
    model = args.model or BotConfig().parser_model

    async def invoke(prompt, sample_idx):
        return await llm.invoke(model, prompt, 0.2, run_index=sample_idx)

    added = asyncio.run(update_playbook(
        failures, args.playbook, invoke, max_new=args.max_new))
    print(f"playbook: added {added} candidate bullet(s) to {args.playbook} "
          "(promotion stays manual + experiments-gated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
