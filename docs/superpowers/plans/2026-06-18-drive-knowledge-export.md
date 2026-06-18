# Daily Knowledge Export → Google Drive — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every day at 08:00 local, mirror the project's created knowledge (research + code + configs + result artifacts) to a Google Drive folder, unattended, with a generated `INDEX.md`/`MANIFEST.json`/`RESULTS_SUMMARY.md`.

**Architecture:** A stdlib-only Python builder (`scripts/export_knowledge.py`) resolves an explicit allowlist into a clean staging tree built atomically; a bash wrapper (`scripts/sync_to_drive.sh`) runs the builder then `rclone sync`s the tree to `gdrive:metaculus-forecasting-bot-export`; a `launchd` job (`deploy/com.aaron.metaculus-bot-export.plist`) fires it daily, matching the existing `com.aaron.metaculus-bot-trigger` job's conventions.

**Tech Stack:** Python 3.12 stdlib (pathlib, hashlib, json, shutil, subprocess), pytest, bash, rclone (Homebrew), macOS launchd.

Spec: `docs/superpowers/specs/2026-06-18-drive-export-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/export_knowledge.py` | Pure builder: allowlist → staging tree, generate INDEX/MANIFEST/RESULTS_SUMMARY, atomic swap, `--dry-run` CLI. |
| `tests/test_export_knowledge.py` | Unit tests against a synthetic temp repo. |
| `scripts/sync_to_drive.sh` | Orchestrator: lock → build → verify remote → `rclone sync`. Honors `DRY_RUN=1`. |
| `deploy/com.aaron.metaculus-bot-export.plist` | launchd job: daily 08:00, logs to Application Support dir. |

`export_knowledge.py` is built up across Tasks 1–4 (each task appends to the same file). Tasks 5–7 add the wrapper, the launchd job, and the live setup.

---

### Task 1: Allowlist resolution + exclusions

**Files:**
- Create: `scripts/export_knowledge.py`
- Test: `tests/test_export_knowledge.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_export_knowledge.py
from pathlib import Path

import scripts.export_knowledge as ek


def make_repo(root: Path) -> Path:
    """Synthetic repo mirroring the real layout's relevant bits."""
    (root / "research-notes").mkdir(parents=True)
    (root / "research-notes/01-x.md").write_text("# Note One\nbody")
    (root / "FORECASTING-BOT-RESEARCH.md").write_text("# Master Research")
    (root / "docs/backtesting").mkdir(parents=True)
    (root / "docs/backtesting/DATA.md").write_text("# Data")
    (root / "bot").mkdir()
    (root / "bot/__init__.py").write_text("")
    (root / "bot/__pycache__").mkdir()
    (root / "bot/__pycache__/x.pyc").write_text("junk")
    (root / "scripts").mkdir()
    (root / "scripts/run.sh").write_text("#!/bin/bash\n")
    (root / "experiments").mkdir()
    (root / "experiments/smoke.yaml").write_text("a: 1")
    (root / "pyproject.toml").write_text("[project]")
    (root / ".env.template").write_text("KEY=")
    (root / ".env").write_text("KEY=secret")
    (root / "data/backtest").mkdir(parents=True)
    (root / "data/backtest/corpus.db").write_text("BIG")
    (root / "logs/backtest/exp/run1").mkdir(parents=True)
    (root / "logs/backtest/exp/run1/report.md").write_text("# Report\nMSE 0.10")
    (root / "logs/backtest/exp/run1/cache.json").write_text("CACHE")
    (root / "logs/forecasts.jsonl").write_text(
        '{"question_id": 1, "type": "numeric", "page_url": "u1", "final": {"50.0": 3}}\n'
    )
    (root / "logs/gate_c_run.log").write_text("gate log")
    return root


def dests(repo: Path) -> set[str]:
    return {str(dest) for _src, dest in ek.resolve_files(repo)}


def test_allowlisted_files_mapped_to_dest(tmp_path):
    repo = make_repo(tmp_path / "repo")
    d = dests(repo)
    assert "research/FORECASTING-BOT-RESEARCH.md" in d
    assert "research/research-notes/01-x.md" in d
    assert "docs/backtesting/DATA.md" in d
    assert "code/bot/__init__.py" in d
    assert "code/scripts/run.sh" in d
    assert "configs/smoke.yaml" in d
    assert "configs/pyproject.toml" in d
    assert "configs/.env.template" in d
    assert "results/forecasts.jsonl" in d
    assert "results/run-logs/gate_c_run.log" in d


def test_excludes_secrets_data_and_cache(tmp_path):
    repo = make_repo(tmp_path / "repo")
    flat = "\n".join(dests(repo))
    assert ".env\n" not in flat + "\n"
    assert "corpus.db" not in flat
    assert ".pyc" not in flat
    assert "cache.json" not in flat


def test_backtest_report_included_but_not_siblings(tmp_path):
    repo = make_repo(tmp_path / "repo")
    d = dests(repo)
    assert "results/backtest-reports/exp/run1/report.md" in d
    assert not any("cache.json" in x for x in d)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_export_knowledge.py -q`
Expected: FAIL — `ModuleNotFoundError`/`AttributeError: module 'scripts.export_knowledge' has no attribute 'resolve_files'`.

(Note: `scripts/` has no `__init__.py` and is a namespace package here; if the import fails, add an empty `scripts/__init__.py` is NOT wanted — instead run pytest from repo root which puts cwd on `sys.path`. Confirm `pyproject.toml`/pytest config picks up repo root. If import still fails, the test file can `import importlib.util` to load by path — but try the plain import first.)

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_export_knowledge.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/export_knowledge.py tests/test_export_knowledge.py
git commit -m "feat(export): allowlist-based file resolution with exclusions"
```

---

### Task 2: Generators — git info, sha256, RESULTS_SUMMARY, INDEX

**Files:**
- Modify: `scripts/export_knowledge.py` (append functions)
- Test: `tests/test_export_knowledge.py` (append tests)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_export_knowledge.py

def test_results_summary_counts_and_lists(tmp_path):
    repo = make_repo(tmp_path / "repo")
    md = ek.build_results_summary(repo)
    assert "Live forecasts: 1" in md
    assert "numeric: 1" in md
    assert "u1" in md                       # page_url in the table
    assert "Backtest reports: 1" in md
    assert "exp/run1" in md                 # run id of the report


def test_index_lists_sections_and_exclusions(tmp_path):
    repo = make_repo(tmp_path / "repo")
    out = tmp_path / "out"
    # lay down a couple of files so the index has something to enumerate
    (out / "research/research-notes").mkdir(parents=True)
    (out / "research/research-notes/01-x.md").write_text("# Note One\n")
    (out / "code/bot").mkdir(parents=True)
    (out / "code/bot/__init__.py").write_text("")
    idx = ek.build_index(out, repo, "abc123", "some subject", "2026-06-18 00:00:00 UTC", "RESULTS_BODY")
    assert "abc123" in idx and "some subject" in idx
    assert "### research/" in idx
    assert "### code/" in idx
    assert "Excluded from this export" in idx
    assert "corpus.db" in idx or "data/backtest" in idx
    assert "RESULTS_BODY" in idx           # results summary inlined
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_export_knowledge.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'build_results_summary'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/export_knowledge.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_export_knowledge.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/export_knowledge.py tests/test_export_knowledge.py
git commit -m "feat(export): INDEX/RESULTS_SUMMARY generators + git/hash helpers"
```

---

### Task 3: Atomic build — build_export + MANIFEST + swap

**Files:**
- Modify: `scripts/export_knowledge.py` (append)
- Test: `tests/test_export_knowledge.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_export_knowledge.py
import json as _json


def test_build_export_writes_tree_manifest_and_generated(tmp_path):
    repo = make_repo(tmp_path / "repo")
    out = tmp_path / "out"
    manifest = ek.build_export(repo, out)

    assert (out / "research/FORECASTING-BOT-RESEARCH.md").is_file()
    assert (out / "results/backtest-reports/exp/run1/report.md").is_file()
    assert (out / "INDEX.md").is_file()
    assert (out / "RESULTS_SUMMARY.md").is_file()
    assert (out / "MANIFEST.json").is_file()
    assert not (out / "code/bot/__pycache__/x.pyc").exists()

    # manifest integrity: every listed hash matches the file on disk
    data = _json.loads((out / "MANIFEST.json").read_text())
    assert data["file_count"] == len(data["files"])
    assert all(e["path"] != "MANIFEST.json" for e in data["files"])
    for entry in data["files"]:
        assert ek.sha256(out / entry["path"]) == entry["sha256"]


def test_rebuild_is_atomic_and_clears_stale(tmp_path):
    repo = make_repo(tmp_path / "repo")
    out = tmp_path / "out"
    ek.build_export(repo, out)
    stale = out / "research/STALE.md"
    stale.write_text("old")
    ek.build_export(repo, out)            # second build
    assert not stale.exists()             # stale file gone after swap
    assert (out / "research/FORECASTING-BOT-RESEARCH.md").is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_export_knowledge.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'build_export'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/export_knowledge.py
import os
import shutil
import tempfile
from datetime import datetime, timezone


def build_manifest(out_dir: Path, commit: str, ts: str) -> dict:
    files = []
    total = 0
    for p in sorted(out_dir.rglob("*")):
        if not p.is_file() or p.name == "MANIFEST.json":
            continue
        size = p.stat().st_size
        total += size
        files.append({"path": str(p.relative_to(out_dir)), "sha256": sha256(p), "bytes": size})
    return {
        "generated_at_utc": ts,
        "git_commit": commit,
        "file_count": len(files),
        "total_bytes": total,
        "files": files,
    }


def _swap(tmp: Path, out: Path) -> None:
    """Atomically replace `out` with `tmp` (same filesystem)."""
    if out.exists():
        old = out.with_name(out.name + ".old")
        shutil.rmtree(old, ignore_errors=True)
        os.replace(out, old)
        os.replace(tmp, out)
        shutil.rmtree(old, ignore_errors=True)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, out)


def build_export(repo_root: Path, out_dir: Path) -> dict:
    pairs = resolve_files(repo_root)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".export-tmp-", dir=out_dir.parent))
    try:
        for src, dest_rel in pairs:
            dest = tmp / dest_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        commit, subject = git_info(repo_root)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        results = build_results_summary(repo_root)
        (tmp / "RESULTS_SUMMARY.md").write_text(results)
        (tmp / "INDEX.md").write_text(
            build_index(tmp, repo_root, commit, subject, ts, results)
        )
        manifest = build_manifest(tmp, commit, ts)
        (tmp / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
        _swap(tmp, out_dir)
        return manifest
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_export_knowledge.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/export_knowledge.py tests/test_export_knowledge.py
git commit -m "feat(export): atomic build_export + MANIFEST.json"
```

---

### Task 4: CLI — `--out`, `--repo-root`, `--dry-run`

**Files:**
- Modify: `scripts/export_knowledge.py` (append)
- Test: `tests/test_export_knowledge.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_export_knowledge.py

def test_dry_run_writes_nothing(tmp_path, capsys):
    repo = make_repo(tmp_path / "repo")
    out = tmp_path / "out"
    rc = ek.main(["--repo-root", str(repo), "--out", str(out), "--dry-run"])
    assert rc == 0
    assert not out.exists()
    captured = capsys.readouterr().out
    assert "dry-run" in captured.lower()
    assert "research/FORECASTING-BOT-RESEARCH.md" in captured


def test_main_builds_when_not_dry(tmp_path):
    repo = make_repo(tmp_path / "repo")
    out = tmp_path / "out"
    rc = ek.main(["--repo-root", str(repo), "--out", str(out)])
    assert rc == 0
    assert (out / "INDEX.md").is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_export_knowledge.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'main'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/export_knowledge.py
import argparse


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the knowledge export tree.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="staging output dir")
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="repo root to export from")
    ap.add_argument("--dry-run", action="store_true", help="print planned files; write nothing")
    args = ap.parse_args(argv)

    if args.dry_run:
        pairs = resolve_files(args.repo_root)
        total = sum(src.stat().st_size for src, _ in pairs)
        print(f"[dry-run] {len(pairs)} files, {human_bytes(total)} would be exported to {args.out}")
        for src, dest in pairs:
            print(f"  {dest}  ({human_bytes(src.stat().st_size)})")
        return 0

    manifest = build_export(args.repo_root, args.out)
    print(f"Exported {manifest['file_count']} files "
          f"({human_bytes(manifest['total_bytes'])}) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_export_knowledge.py -q`
Expected: PASS (9 tests).

- [ ] **Step 5: Verify against the REAL repo (sanity, no upload)**

Run: `python scripts/export_knowledge.py --dry-run`
Expected: prints a few hundred files mapped under `research/ docs/ code/ configs/ results/`, NO `.env`, NO `data/backtest`, NO `logs/backtest/**` except `report.md`.

- [ ] **Step 6: Commit**

```bash
git add scripts/export_knowledge.py tests/test_export_knowledge.py
git commit -m "feat(export): CLI with --out/--repo-root/--dry-run"
```

---

### Task 5: Sync wrapper `scripts/sync_to_drive.sh`

**Files:**
- Create: `scripts/sync_to_drive.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Build the knowledge export tree and mirror it to Google Drive via rclone.
# Honors DRY_RUN=1 (builds + prints, no upload). Run by launchd daily.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUPPORT_DIR="$HOME/Library/Application Support/metaculus-bot"
STAGING="$SUPPORT_DIR/export"
LOCK="$SUPPORT_DIR/export.lock"
LOG="$SUPPORT_DIR/export.log"
REMOTE="gdrive:metaculus-forecasting-bot-export"

PYTHON="$REPO_ROOT/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"

mkdir -p "$SUPPORT_DIR"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

# single-instance lock (mkdir is atomic)
if ! mkdir "$LOCK" 2>/dev/null; then
  log "another sync is running ($LOCK exists); exiting"
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

if [ "${DRY_RUN:-0}" = "1" ]; then
  log "DRY_RUN=1: building export (no upload)"
  "$PYTHON" "$REPO_ROOT/scripts/export_knowledge.py" --out "$STAGING" --repo-root "$REPO_ROOT" --dry-run
  exit 0
fi

log "building export tree -> $STAGING"
"$PYTHON" "$REPO_ROOT/scripts/export_knowledge.py" --out "$STAGING" --repo-root "$REPO_ROOT"

if ! command -v rclone >/dev/null 2>&1; then
  log "ERROR: rclone not installed. Run: brew install rclone"
  exit 2
fi
if ! rclone listremotes 2>/dev/null | grep -q '^gdrive:'; then
  log "ERROR: rclone remote 'gdrive' not configured. Run: rclone config"
  exit 3
fi

log "syncing -> $REMOTE"
rclone sync "$STAGING" "$REMOTE" --create-empty-src-dirs --log-file "$LOG" --log-level INFO
log "sync complete"
```

- [ ] **Step 2: Make executable**

Run: `chmod +x scripts/sync_to_drive.sh`

- [ ] **Step 3: Dry-run test (no rclone/remote needed)**

Run: `DRY_RUN=1 scripts/sync_to_drive.sh`
Expected: log lines + the builder's dry-run file list; exit 0; `~/Library/Application Support/metaculus-bot/export` NOT created.

- [ ] **Step 4: Commit**

```bash
git add scripts/sync_to_drive.sh
git commit -m "feat(export): sync_to_drive.sh wrapper (lock + rclone sync, DRY_RUN)"
```

---

### Task 6: launchd job + one-time setup (rclone install, OAuth, load)

**Files:**
- Create: `deploy/com.aaron.metaculus-bot-export.plist`

- [ ] **Step 1: Write the plist**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.aaron.metaculus-bot-export</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>/Users/aaronbao/Developer/sideprojects/forecasting/scripts/sync_to_drive.sh</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>8</integer>
    <key>Minute</key><integer>0</integer>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>/Users/aaronbao/Library/Application Support/metaculus-bot/export.out.log</string>
  <key>StandardErrorPath</key><string>/Users/aaronbao/Library/Application Support/metaculus-bot/export.err.log</string>
</dict>
</plist>
```

- [ ] **Step 2: Commit the plist (install happens in Task 7 after OAuth)**

```bash
git add deploy/com.aaron.metaculus-bot-export.plist
git commit -m "feat(export): launchd job (daily 08:00)"
```

- [ ] **Step 3: Install rclone**

Run: `brew install rclone && rclone version | head -1`
Expected: prints an `rclone vX.Y.Z` line.

- [ ] **Step 4 (USER, manual): Configure the Google Drive remote**

The agent stops here and hands these exact keystrokes to Aaron (browser OAuth can't be automated):

```
rclone config
  n)  New remote
  name> gdrive
  Storage> drive               (type the number for "Google Drive")
  client_id>                   (blank — press Enter)
  client_secret>               (blank — press Enter)
  scope> 3                     (drive.file — access to files created by rclone only)
  service_account_file>        (blank — Enter)
  Edit advanced config? n
  Use web browser to authenticate? y   → approve in browser
  Configure this as a Shared Drive? n
  y) Yes this is OK
  q) Quit config
```

Verify: `rclone listremotes` shows `gdrive:`.

- [ ] **Step 5: First live sync (manual trigger, verify before scheduling)**

Run: `scripts/sync_to_drive.sh`
Expected: log ends with `sync complete`; the folder `metaculus-forecasting-bot-export` appears in Google Drive containing `INDEX.md`, `MANIFEST.json`, and the `research/ docs/ code/ configs/ results/` tree.

- [ ] **Step 6: Install & load the launchd job**

Run:
```bash
cp deploy/com.aaron.metaculus-bot-export.plist ~/Library/LaunchAgents/
launchctl unload ~/Library/LaunchAgents/com.aaron.metaculus-bot-export.plist 2>/dev/null || true
launchctl load -w ~/Library/LaunchAgents/com.aaron.metaculus-bot-export.plist
launchctl list | grep metaculus-bot-export
```
Expected: the job is listed. `RunAtLoad` triggers an immediate sync; confirm `~/Library/Application Support/metaculus-bot/export.log` shows a fresh `sync complete`.

---

### Task 7: Final verification & docs

**Files:**
- Modify: `docs/superpowers/specs/2026-06-18-drive-export-design.md` (optional: append "Implemented" note)

- [ ] **Step 1: Full test suite green**

Run: `python -m pytest tests/test_export_knowledge.py -q`
Expected: 9 passed.

- [ ] **Step 2: Confirm scheduled job is registered**

Run: `launchctl list | grep metaculus-bot-export`
Expected: one line with the label.

- [ ] **Step 3: Confirm Drive folder reflects current state**

Manually open Drive → `metaculus-forecasting-bot-export` → open `INDEX.md`, confirm git commit + timestamp are current and the exclusions table shows the dataset/log sizes.

- [ ] **Step 4: Final commit (if spec annotated)**

```bash
git add -A
git commit -m "docs(export): mark daily Drive export implemented"
```

---

## Self-Review

**Spec coverage:**
- Allowlist + dest mapping → Task 1. ✅
- Exclusions (secrets/data/cache, exclusions win) → Task 1 (`is_excluded`) + tests. ✅
- INDEX.md (header, contents w/ descriptions, data inventory, results) → Task 2. ✅
- RESULTS_SUMMARY.md (counts, type breakdown, backtest reports) → Task 2. ✅
- MANIFEST.json (sha256, bytes, commit, ts) → Task 3. ✅
- Atomic swap → Task 3 (`_swap`) + atomic-rebuild test. ✅
- `--dry-run`/CLI → Task 4. ✅
- sync_to_drive.sh (lock, remote check, exit codes, DRY_RUN) → Task 5. ✅
- launchd daily 08:00 + logging conventions → Task 6. ✅
- rclone install + `drive.file` OAuth → Task 6 steps 3–4. ✅
- Setup cost / handoff steps → Task 6. ✅

**Placeholder scan:** No TBD/TODO; every code step shows complete code; manual OAuth step lists exact keystrokes. ✅

**Type consistency:** `resolve_files`→`list[tuple[Path,Path]]` used by `build_export`; `build_index(out_dir, repo_root, commit, subject, ts, results_summary)` signature matches its call in `build_export`; `build_results_summary(repo_root)`, `build_manifest(out_dir, commit, ts)`, `sha256(path)`, `git_info(repo_root)`, `human_bytes`, `main(argv)` consistent across tasks and tests. ✅
