#!/usr/bin/env python3
"""Build a clean, browsable export of the project's created knowledge.

Allowlist-based: only paths matched by INCLUDE_RULES are copied, and any path
matching an exclusion is dropped (exclusions win). Later tasks add generated
INDEX.md / MANIFEST.json / RESULTS_SUMMARY.md and atomic publishing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = Path.home() / "Library/Application Support/metaculus-bot/export"


@dataclass(frozen=True)
class Rule:
    base: str      # dir relative to repo root whose matched files are copied
    pattern: str   # glob relative to base (supports **)
    dest: str      # destination prefix under the export root


INCLUDE_RULES: tuple[Rule, ...] = (
    Rule("", "FORECASTING-BOT-RESEARCH.md", "research"),
    Rule("research-notes", "*.md", "research/research-notes"),
    Rule("docs", "**/*.md", "docs"),
    Rule("bot", "**/*.py", "code/bot"),
    Rule("backtest", "**/*.py", "code/backtest"),
    Rule("scripts", "**/*.py", "code/scripts"),
    Rule("scripts", "**/*.sh", "code/scripts"),
    Rule("tests", "**/*.py", "code/tests"),
    Rule("experiments", "*.yaml", "configs"),
    Rule("", "pyproject.toml", "configs"),
    Rule("", "uv.lock", "configs"),
    Rule("", ".env.template", "configs"),
    Rule("logs", "forecasts.jsonl", "results"),
    Rule("logs/backtest", "**/report.md", "results/backtest-reports"),
    Rule("logs", "gate_*.log", "results/run-logs"),
    Rule("logs", "live_preview*.log", "results/run-logs"),
)

EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", ".pytest_cache", "node_modules"}


def is_excluded(rel: Path) -> bool:
    if set(rel.parts) & EXCLUDE_DIRS:
        return True
    if rel.name in {".DS_Store", ".env"}:
        return True
    if rel.suffix == ".pyc":
        return True
    if any(p.endswith(".egg-info") for p in rel.parts):
        return True
    return False


def resolve_files(repo_root: Path) -> list[tuple[Path, Path]]:
    """Return sorted (src_abs, dest_rel) pairs for every file to export."""
    pairs: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    for rule in INCLUDE_RULES:
        base = repo_root / rule.base if rule.base else repo_root
        if not base.exists():
            continue
        for src in sorted(base.glob(rule.pattern)):
            if not src.is_file():
                continue
            rel_repo = src.relative_to(repo_root)
            if is_excluded(rel_repo):
                continue
            dest_rel = Path(rule.dest) / src.relative_to(base)
            if dest_rel in seen:
                continue
            seen.add(dest_rel)
            pairs.append((src, dest_rel))
    return pairs
