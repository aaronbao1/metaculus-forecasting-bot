"""Append-only JSONL forecast log. This is the substrate for ALL
self-improvement: calibration fitting, config A/B comparison, and error
post-mortems condition on what was forecast, by which members, under which
config hash."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _append_jsonl(record: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {**record, "logged_at": datetime.now(timezone.utc).isoformat()}
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_forecast(record: dict, path) -> None:
    _append_jsonl(record, path)


def read_forecasts(path) -> list[dict]:
    return _read_jsonl(path)


def append_run(record: dict, path="logs/runs.jsonl") -> None:
    """Append-only per-tournament run-summary (coverage/truncation/retry signal).
    Separate file from forecasts.jsonl so per-type forecast readers are unaffected."""
    _append_jsonl(record, path)


def read_runs(path="logs/runs.jsonl") -> list[dict]:
    return _read_jsonl(path)
