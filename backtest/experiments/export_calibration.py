"""Calibration flywheel: convert a backtest run directory into rows that
scripts/fit_calibration.py consumes, so the binary Platt map can be fitted
offline on hundreds of resolved backtest forecasts instead of waiting on
live history (spec §10 item 3).

EXACT field mapping (live logs/forecasts.jsonl row, as written by
bot/ensemble_bot.py _log() -> exported row):

  question_id  <- ForecastArtifact.qid (string "metaculus:90001"; live logs
                  use the int post id — join_resolutions only ever uses the
                  value as a dict key into `resolutions`, so any consistent
                  hashable id works)
  type         <- forecast.qtype value: "binary" | "multiple_choice" |
                  "numeric" | "discrete" (join_resolutions filters on
                  type == "binary", exactly like live rows)
  final        <- forecast.binary_prob for binary; None otherwise
                  (join_resolutions skips rows with final None)
  raw          <- first StageRecord whose outputs contain "aggregate_raw"
                  (fallback key: "raw") — the pre-calibration aggregate,
                  mirroring the live binary rows' `raw` field; None when the
                  system logged no aggregation stage (e.g. baselines/stub)
  config_hash  <- ForecastArtifact.config_fingerprint
  resolution   <- joined from the run's scores.jsonl (rows keyed "qid" or
                  "question_id"); binary bools coerced to 1/0. NOT part of
                  the live row shape — additive, carried here so the join
                  needs no network. Missing scores.jsonl => None (and
                  join_resolutions then skips the row, same as an
                  unresolved live question).
  horizon, system, as_of  <- artifact metadata (additive; ignored by
                  fit_calibration, useful for slicing exports)

fit_calibration.join_resolutions reads (lines 16-25 of scripts/fit_calibration.py):
  - row.get("type") == "binary"        -> filter to binary only
  - row.get("question_id")             -> key into resolutions dict
  - row.get("final")                   -> prediction float (None => skipped)
  resolutions dict: question_id -> 1/0 (int)

Raw vs final probabilities:
  - `final`  is the post-calibration output (forecast.binary_prob) — this
             is what fit_calibration.join_resolutions fits the Platt map ON.
             Fitting on final (post-calibration) produces a second-order
             correction; fitting on raw (pre-calibration) produces a first-
             order map. We export both so the caller can choose.
  - `raw`    is the pre-calibration ensemble aggregate
             (from StageRecord.outputs["aggregate_raw"] or "raw") — not
             consumed by join_resolutions directly, but available for
             diagnostic slicing or fitting a fresh Platt map from scratch.

Offline flywheel usage (no network):

    from backtest.experiments.export_calibration import (
        artifacts_to_calibration_rows, resolutions_from_rows)
    from scripts.fit_calibration import join_resolutions
    from bot.calibration import MIN_FIT_N, fit_platt, save_platt

    rows = artifacts_to_calibration_rows(run_dir)
    preds, outcomes = join_resolutions(rows, resolutions_from_rows(rows))
    if len(preds) >= MIN_FIT_N:
        save_platt(fit_platt(preds, outcomes), "calibration/binary.json")

CAUTION: a Platt map fitted on backtest artifacts inherits the backtest's
biases (clean-stratum skew, missing Perplexity research track). Point the
adapter's variant at a separate calibration_dir when experimenting; only
promote to the live calibration/ dir deliberately.
"""
from __future__ import annotations

import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def _resolution_index(run_dir: Path) -> dict[str, object]:
    """qid -> resolution, from the run's scores.jsonl (the runner records
    the resolution it scored against)."""
    out: dict[str, object] = {}
    for row in _read_jsonl(run_dir / "scores.jsonl"):
        qid = row.get("qid") or row.get("question_id")
        if qid is not None and "resolution" in row:
            out[str(qid)] = row["resolution"]
    return out


def _raw_from_stages(stages: list[dict]) -> object:
    for st in stages or []:
        outs = st.get("outputs") or {}
        if "aggregate_raw" in outs:
            return outs["aggregate_raw"]
        if "raw" in outs:
            return outs["raw"]
    return None


def artifacts_to_calibration_rows(run_dir) -> list[dict]:
    """All artifacts with a final forecast, as fit_calibration-shaped rows.

    Skips artifacts where forecast is None (failed/errored questions).
    Binary booleans in scores.jsonl are coerced to 1/0 so join_resolutions
    can call int() on them.
    """
    run_dir = Path(run_dir)
    res_idx = _resolution_index(run_dir)
    rows: list[dict] = []
    for d in _read_jsonl(run_dir / "artifacts.jsonl"):
        fc = d.get("forecast")
        if not fc:
            continue   # skipped/failed question: nothing to calibrate on
        qtype = fc.get("qtype")
        # `final` is the post-calibration binary probability — the value
        # fit_calibration.join_resolutions reads for fitting the Platt map.
        # For non-binary types we set None; join_resolutions will skip them.
        final = fc.get("binary_prob") if qtype == "binary" else None
        res = res_idx.get(str(d["qid"]))
        # Coerce binary bools to 1/0; numeric resolutions pass through as-is
        # (join_resolutions / resolutions_from_rows filter to binary anyway).
        if qtype == "binary" and isinstance(res, bool):
            res = int(res)
        rows.append({
            "question_id": d["qid"],
            "type": qtype,
            # final = post-calibration probability (what join_resolutions fits on)
            "final": final,
            # raw = pre-calibration ensemble aggregate (diagnostic; not read by
            # join_resolutions directly, but preserved for offline analysis or
            # fitting a fresh first-order Platt map)
            "raw": _raw_from_stages(d.get("stages")),
            "config_hash": d.get("config_fingerprint"),
            "resolution": res,
            "horizon": d.get("horizon"),
            "system": d.get("system_name"),
            "as_of": d.get("as_of"),
        })
    return rows


def resolutions_from_rows(rows: list[dict]) -> dict:
    """The `resolutions` dict join_resolutions expects, from exported rows.

    Only binary rows with a non-None resolution are included — this mirrors
    how the live script fetches resolutions only for resolved binary questions.
    """
    return {r["question_id"]: r["resolution"] for r in rows
            if r.get("type") == "binary" and r.get("resolution") is not None}


def write_calibration_export(run_dir, out_path) -> Path:
    """Write rows as a forecasts-jsonl-compatible file (one JSON object per
    line — readable by bot.logging_store.read_forecasts)."""
    rows = artifacts_to_calibration_rows(run_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Export a backtest run's artifacts as "
                    "fit_calibration-compatible JSONL")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("out_path", type=Path)
    args = ap.parse_args()
    p = write_calibration_export(args.run_dir, args.out_path)
    n = len(artifacts_to_calibration_rows(args.run_dir))
    print(f"wrote {n} rows to {p}")
