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
from pathlib import Path

DEFAULT_MAX_CHARS = 4000

_COMMENT_RE = re.compile(r"\s*<!--.*?-->", re.DOTALL)


def _promoted_bullets(text: str) -> list[str]:
    """Bullets under '## Promoted', up to the next '## ' heading."""
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
            bullet = _COMMENT_RE.sub("", line).rstrip()
            if bullet.strip(" -"):
                bullets.append(bullet)
    return bullets


def load_playbook(path, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Promoted bullets as a text block; "" when the file or section is
    empty/absent. Truncates at a bullet boundary so a partial lesson is
    never injected."""
    path = Path(path)
    if not path.exists():
        return ""
    bullets = _promoted_bullets(path.read_text())
    kept: list[str] = []
    total = 0
    for b in bullets:
        if total + len(b) + 1 > max_chars:
            break
        kept.append(b)
        total += len(b) + 1
    return "\n".join(kept)
