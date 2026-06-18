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


import hashlib
import json
import subprocess


DESCRIPTIONS = {
    "research/FORECASTING-BOT-RESEARCH.md": "Master research synthesis for the bot.",
    "research/research-notes/01-halawi-deep-read.md": "Deep read of the Halawi LLM-forecasting paper.",
    "research/research-notes/02-llm-forecasting-literature.md": "LLM forecasting literature review.",
    "research/research-notes/03-superforecasting-science.md": "Superforecasting science notes.",
    "research/research-notes/04-bot-architectures.md": "Survey of forecasting-bot architectures.",
    "research/research-notes/05-metaculus-mechanics.md": "Metaculus scoring / season mechanics.",
    "research/research-notes/06-deep-research-verified-findings.md": "Verified deep-research findings.",
    "research/research-notes/07-incorporation-research.md": "Incorporation research.",
    "research/research-notes/08-improvement-roadmap.md": "Improvement roadmap.",
    "research/research-notes/09-run1-evaluation.md": "Run-1 evaluation.",
    "research/research-notes/10-cutoff-audit.md": "Knowledge-cutoff leakage audit.",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(repo_root: Path) -> tuple[str, str]:
    def run(args: list[str]) -> str:
        try:
            r = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                capture_output=True, text=True, check=True,
            )
            return r.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"
    return run(["rev-parse", "--short", "HEAD"]), run(["log", "-1", "--format=%s"])


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _dir_size(path: Path) -> str:
    if not path.exists():
        return "—"
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return human_bytes(total)


def first_heading(path: Path) -> str:
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if s.startswith("#"):
            return s.lstrip("#").strip()
    return "—"


def build_results_summary(repo_root: Path) -> str:
    out = ["# Results Summary", ""]
    fc = repo_root / "logs/forecasts.jsonl"
    if fc.exists():
        records = []
        for line in fc.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        by_type: dict[str, int] = {}
        for r in records:
            t = r.get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
        out.append(f"## Live forecasts: {len(records)}")
        out.append("")
        for t in sorted(by_type):
            out.append(f"- {t}: {by_type[t]}")
        out.append("")
        out.append("| question_id | type | url |")
        out.append("|---|---|---|")
        for r in records:
            out.append(f"| {r.get('question_id', '?')} | {r.get('type', '?')} | {r.get('page_url', '')} |")
        out.append("")
    else:
        out.append("_No forecasts.jsonl found._")
        out.append("")
    bt = repo_root / "logs/backtest"
    reports = sorted(bt.glob("**/report.md")) if bt.exists() else []
    out.append(f"## Backtest reports: {len(reports)}")
    out.append("")
    for rep in reports:
        run_id = rep.parent.relative_to(bt)
        first = next((l for l in rep.read_text(errors="ignore").splitlines() if l.strip()), "")
        out.append(f"- `{run_id}` — {first}")
    out.append("")
    return "\n".join(out)


def build_index(out_dir: Path, repo_root: Path, commit: str, subject: str,
                ts: str, results_summary: str) -> str:
    GENERATED = {"INDEX.md", "MANIFEST.json", "RESULTS_SUMMARY.md"}
    lines = [
        "# Metaculus Forecasting Bot — Knowledge Export",
        "",
        f"- **Git commit:** `{commit}` — {subject}",
        f"- **Generated (UTC):** {ts}",
        "",
        "Auto-generated daily mirror of the project's research, code, configs, and "
        "result artifacts. Do not edit by hand — overwritten on the next sync.",
        "",
        "## Contents",
        "",
    ]
    files = sorted(p.relative_to(out_dir) for p in out_dir.rglob("*") if p.is_file())
    by_top: dict[str, list[Path]] = {}
    for f in files:
        if f.name in GENERATED:
            continue
        by_top.setdefault(f.parts[0], []).append(f)
    for top in sorted(by_top):
        lines += [f"### {top}/", "", "| file | description |", "|---|---|"]
        for f in by_top[top]:
            key = str(f)
            if key in DESCRIPTIONS:
                desc = DESCRIPTIONS[key]
            elif f.suffix == ".md":
                desc = first_heading(out_dir / f)
            else:
                desc = "—"
            lines.append(f"| `{f}` | {desc} |")
        lines.append("")
    lines += [
        "## Excluded from this export (and why)",
        "",
        "| path | size | reason |",
        "|---|---|---|",
        f"| `data/backtest/` | {_dir_size(repo_root / 'data/backtest')} | "
        "Downloaded datasets (ForecastBench, Halawi) + built corpus.db; reproducible "
        "per `docs/backtesting/DATA.md`. |",
        f"| `logs/backtest/**` (cache) | {_dir_size(repo_root / 'logs/backtest')} | "
        "LLM response cache + per-question artifacts; only report.md is exported. |",
        "| `.env` | — | Live API keys. |",
        "",
        "## Latest results",
        "",
        results_summary,
    ]
    return "\n".join(lines)
