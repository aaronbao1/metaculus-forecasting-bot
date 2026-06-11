"""Append-only JSONL forecast log. This is the substrate for ALL
self-improvement: calibration fitting, config A/B comparison, and error
post-mortems condition on what was forecast, by which members, under which
config hash."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def append_forecast(record: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {**record, "logged_at": datetime.now(timezone.utc).isoformat()}
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def read_forecasts(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
