# Daily Knowledge Export → Google Drive — Design (2026-06-18)

Mirrors the project's *created* knowledge (research + code + configs + result
artifacts) to a Google Drive folder once per day, unattended, via `rclone` +
`launchd`. The Drive folder is self-describing: a generated `INDEX.md` and
`MANIFEST.json` document everything, including what was deliberately left out.

## Goal / non-goals

**Goal:** every day at 08:00 local, a browsable Google Drive folder reflects the
current state of the repo's research writing, source code, config, and small
result artifacts — with a generated index/report so a reader (or future Aaron)
can understand the export without the repo.

**Non-goals:**
- Two-way sync (this is strictly machine → Drive).
- Backing up secrets, downloaded external datasets, or the LLM cache (see
  Exclusions).
- A cloud `/schedule` routine — it runs without the local files/results, wrong
  tool. This runs locally on Aaron's Mac, matching the existing
  `com.aaron.metaculus-bot-trigger` launchd job.
- Dated point-in-time snapshots — Google Drive keeps its own version history, so
  a single live mirror is enough (YAGNI). Revisit only if version history proves
  insufficient.

## What gets exported

Staging tree rebuilt from scratch each run, then mirrored to
`gdrive:metaculus-forecasting-bot-export`:

```
metaculus-forecasting-bot-export/
├── INDEX.md            generated master report
├── MANIFEST.json       machine-readable inventory: relpath → {sha256, bytes}, + git commit, UTC ts
├── research/           FORECASTING-BOT-RESEARCH.md + research-notes/*.md
├── docs/               docs/ tree (backtesting docs, superpowers plans + specs)
├── code/               bot/ backtest/ scripts/ tests/   (source only)
├── configs/            experiments/*.yaml, pyproject.toml, uv.lock, .env.template
└── results/            logs/forecasts.jsonl, logs/backtest/**/report.md, logs/gate_*.log
                        + RESULTS_SUMMARY.md
```

### Include rules (allowlist)
Source of truth is an explicit allowlist in `export_knowledge.py`, NOT "copy
everything minus ignores". Each entry is `(src_glob_or_path, dest_subdir)`:

| Source | Dest |
|---|---|
| `FORECASTING-BOT-RESEARCH.md` | `research/` |
| `research-notes/*.md` | `research/research-notes/` |
| `docs/**` (excluding `.DS_Store`) | `docs/` |
| `bot/**/*.py`, `backtest/**/*.py`, `scripts/**` (`*.py`,`*.sh`), `tests/**/*.py` | `code/<dir>/` |
| `experiments/*.yaml`, `pyproject.toml`, `uv.lock`, `.env.template` | `configs/` |
| `logs/forecasts.jsonl` | `results/` |
| `logs/backtest/**/report.md` | `results/backtest-reports/<run-id>/report.md` |
| `logs/gate_*.log`, `logs/live_preview*.log` | `results/run-logs/` |

### Exclusions (documented in INDEX.md, never uploaded)
- `.env` — live API keys.
- `data/backtest/` — ~277 MB downloaded external datasets (ForecastBench, Halawi)
  and the 31 MB built `corpus.db`. Reproducible from `docs/backtesting/DATA.md`.
- `logs/backtest/**` except `report.md` — ~67 MB LLM response cache + per-question
  artifacts.
- `.venv/`, `.git/`, `__pycache__/`, `*.pyc`, `*.egg-info/`, `.pytest_cache/`,
  `.DS_Store`.

Anything matching an exclusion is skipped even if an include glob would catch it
(exclusions win).

## Generated files

**`INDEX.md`** — sections:
1. Header: project one-liner, git commit (short sha + subject), UTC build time.
2. Contents: per top-level export dir, a table of files with a one-line
   description. Descriptions for the 10 `research-notes/` files and key docs are
   hardcoded in a `DESCRIPTIONS` dict; everything else falls back to the file's
   first markdown heading or "—".
3. Data inventory: the excluded datasets/logs, their sizes, and how to regenerate
   (pointer to `docs/backtesting/DATA.md`).
4. Latest results snapshot: inlines the contents of `RESULTS_SUMMARY.md`.

**`RESULTS_SUMMARY.md`** — derived, not copied:
- `forecasts.jsonl`: total count, breakdown by `type` (binary/numeric/MC/date),
  list of `question_id` + `page_url` + median.
- backtest `report.md` files found: run id + first heading + headline metric line.

**`MANIFEST.json`** — `{generated_at_utc, git_commit, file_count, total_bytes,
files: [{path, sha256, bytes}]}`. Lets the sync step and any consumer verify
integrity / detect changes.

## Components & data flow

```
launchd (daily 08:00, com.aaron.metaculus-bot-export)
  └─> scripts/sync_to_drive.sh
        ├─ acquire lock (~/Library/Application Support/metaculus-bot/export.lock)
        ├─ python scripts/export_knowledge.py --out <staging>     # build atomically
        ├─ verify gdrive: remote exists  (rclone listremotes | grep)
        └─ rclone sync <staging> gdrive:metaculus-forecasting-bot-export \
             --create-empty-src-dirs --log-file=<export.log> --log-level INFO
```

### `scripts/export_knowledge.py`
Pure, side-effect-isolated builder. Public surface:
- `build_export(repo_root: Path, out_dir: Path) -> Manifest` — builds into a
  fresh temp dir, then atomically `os.replace`s it into `out_dir` (so a partial
  build is never synced).
- `--dry-run` — print the resolved file list + total bytes + exclusions hit;
  write nothing.
- `--out PATH` — staging dir (default `~/Library/Application Support/metaculus-bot/export`).
Depends only on stdlib (`pathlib`, `hashlib`, `json`, `shutil`, `subprocess` for
`git rev-parse`). No third-party deps so it runs under the project venv or bare
python3.

### `scripts/sync_to_drive.sh`
Thin orchestrator. Bash, `set -euo pipefail`. Resolves repo root from its own
path. Flock-style lock via `mkdir` (atomic). If `gdrive:` remote missing → exit
3 with a message telling the user to run `rclone config`. Honors `DRY_RUN=1`
(passes `--dry-run` to the builder and `--dry-run` to rclone).

### `deploy/com.aaron.metaculus-bot-export.plist`
launchd template (paths absolute, matching the existing bot-trigger job):
- `StartCalendarInterval` → `{Hour: 8, Minute: 0}` (daily 08:00).
- `RunAtLoad` true (first sync happens at install, good for verification).
- `StandardOutPath`/`StandardErrorPath` →
  `~/Library/Application Support/metaculus-bot/export.{out,err}.log`.
Installed via `launchctl load -w`.

## Error handling
- **Builder fails** → temp dir discarded, staging untouched, sync not attempted,
  non-zero exit logged.
- **`gdrive:` remote not configured** → sync script exits 3 with setup
  instructions; nothing uploaded.
- **rclone/network failure** → rclone's own retries; launchd reruns next day.
  Non-zero exit surfaces in `export.err.log`.
- **Overlapping runs** → lock dir; second invocation exits 0 with "already
  running".
- **No partial state on Drive** → atomic local swap means rclone only ever sees a
  complete tree; `rclone sync` then makes Drive match it.

## Testing (TDD)
`tests/test_export_knowledge.py`, against a synthetic temp repo:
1. Allowlisted files land in the right dest subdir.
2. Excluded paths (`.env`, `data/backtest/x`, `logs/backtest/run/cache.json`,
   `__pycache__`) never appear in output.
3. `report.md` under `logs/backtest/**` IS included; sibling cache files are not.
4. `MANIFEST.json` lists every output file with a correct sha256 and byte count.
5. `RESULTS_SUMMARY.md` counts forecasts.jsonl lines and breaks down by type.
6. Atomic swap: a pre-existing stale file in `out_dir` is gone after a rebuild.
7. `--dry-run` writes nothing.

Manual/integration: `DRY_RUN=1 scripts/sync_to_drive.sh` (no remote needed) →
inspect staging tree + INDEX.md; then a real `rclone sync` after OAuth, verify
folder appears in Drive.

## Setup cost (one-time, by Aaron)
1. `brew install rclone` (I run this).
2. `rclone config` → new remote named `gdrive`, type `drive`, scope
   `drive.file` (app-created files only — least privilege), browser OAuth (~2
   min). I provide exact keystrokes.
3. `launchctl load -w ~/Library/LaunchAgents/com.aaron.metaculus-bot-export.plist`
   (I provide; or I run it).

## Files created
- `scripts/export_knowledge.py`
- `scripts/sync_to_drive.sh`
- `deploy/com.aaron.metaculus-bot-export.plist`
- `tests/test_export_knowledge.py`
- this spec
