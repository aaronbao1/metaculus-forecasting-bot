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
[ -x "$PYTHON" ] || PYTHON="$(command -v python3 || true)"

STALE_LOCK_SECONDS=21600  # 6h; far longer than any real sync

mkdir -p "$SUPPORT_DIR"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

if [ -z "${PYTHON:-}" ] || [ ! -x "$PYTHON" ]; then
  log "ERROR: no python interpreter found (.venv/bin/python or python3 on PATH)"
  exit 4
fi

# single-instance lock (mkdir is atomic)
if ! mkdir "$LOCK" 2>/dev/null; then
  # Lock held by someone else (or a crashed run). Reclaim only if stale.
  lock_mtime="$(stat -f %m "$LOCK" 2>/dev/null || echo 0)"
  now="$(date +%s)"
  lock_age=$(( now - lock_mtime ))
  if [ "$lock_age" -gt "$STALE_LOCK_SECONDS" ]; then
    log "removing stale lock ($LOCK, age ${lock_age}s > ${STALE_LOCK_SECONDS}s)"
    rmdir "$LOCK" 2>/dev/null || true
    if ! mkdir "$LOCK" 2>/dev/null; then
      log "could not acquire lock after removing stale one; exiting"
      exit 0
    fi
  else
    log "another sync is running ($LOCK exists, age ${lock_age}s); exiting"
    exit 0
  fi
fi
# Trap installed only after THIS invocation acquired the lock, so a foreign
# (non-stale) lock we exited on above is never removed.
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT INT TERM

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
