"""Forecasting playbook loader (incorporation plan item 5, limited scope).

The playbook file (skills/forecasting-playbook.md) has two sections:
`## Promoted` — bullets injected into forecast prompts when
BotConfig.playbook_path is set — and `## Candidates` — proposals written by
scripts/playbook_update.py that are NEVER injected. Promotion is a manual
edit gated by an experiments-gate pass (ACE-style: deltas are human-readable
and reviewable; nothing self-promotes).
"""
from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

DEFAULT_MAX_CHARS = 4000

_COMMENT_RE = re.compile(r"\s*<!--.*?-->", re.DOTALL)
_DATE_RE = re.compile(r"date:(\d{4}-\d{2}-\d{2})")


def _promoted_bullets(text: str,
                      as_of: datetime | None = None) -> list[str]:
    """Bullets under '## Promoted', up to the next '## ' heading.

    Temporal guard: bullets are distilled from RESOLVED outcomes, so under a
    backtest as_of only bullets whose provenance date precedes as_of may be
    injected — an undated bullet is dropped (fail closed). Live (as_of=None)
    keeps everything."""
    lines = text.splitlines()
    bullets: list[str] = []
    in_section = False
    for line in lines:
        if line.strip().lower().startswith("## promoted"):
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section and line.lstrip().startswith("- "):
            if as_of is not None:
                m = _DATE_RE.search(line)
                if m is None or date.fromisoformat(m.group(1)) \
                        > as_of.date():
                    continue
            bullet = _COMMENT_RE.sub("", line).rstrip()
            if bullet.strip(" -"):
                bullets.append(bullet)
    return bullets


def load_playbook(path, max_chars: int = DEFAULT_MAX_CHARS,
                  as_of: datetime | None = None) -> str:
    """Promoted bullets as a text block; "" when the file or section is
    empty/absent. Truncates at a bullet boundary so a partial lesson is
    never injected. as_of applies the temporal guard (backtests)."""
    path = Path(path)
    if not path.exists():
        return ""
    bullets = _promoted_bullets(path.read_text(), as_of=as_of)
    kept: list[str] = []
    total = 0
    for b in bullets:
        if total + len(b) + 1 > max_chars:
            break
        kept.append(b)
        total += len(b) + 1
    return "\n".join(kept)
