"""Generic two-column (timestamp, value) CSV importer for target series —
the offline-testable path into kind="target" (the quantity a numeric
question asks about). Header row optional; unparsable rows are reported,
never fatal."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from backtest.corpus.store import CorpusStore, FetchReport


def _parse_ts(raw: str) -> str:
    dt = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    return dt.isoformat()


def import_series_csv(store: CorpusStore, path, qid: str,
                      kind: str = "target") -> FetchReport:
    report = FetchReport()
    points: list[tuple[str, float]] = []
    with Path(path).open() as fh:
        for i, row in enumerate(csv.reader(fh)):
            if not row or len(row) < 2:
                continue
            try:
                points.append((_parse_ts(row[0]), float(row[1])))
            except ValueError as e:
                if i == 0:
                    continue          # header row
                report.errors.append(f"row {i + 1}: {e}")
    if points:
        store.put_series(qid, kind, points)
    report.fetched = len(points)
    return report
