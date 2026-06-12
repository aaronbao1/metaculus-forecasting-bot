"""Calibration flywheel: a run directory's artifacts.jsonl (+ resolutions
from scores.jsonl) round-trips into scripts/fit_calibration.py's
join_resolutions exactly like live logs/forecasts.jsonl rows do."""
import json
from datetime import datetime, timezone

from backtest.core import (Forecast, ForecastArtifact, QType, StageRecord,
                           artifact_to_dict)
from backtest.experiments.export_calibration import (
    artifacts_to_calibration_rows, resolutions_from_rows,
    write_calibration_export)
from bot.logging_store import read_forecasts
from scripts.fit_calibration import join_resolutions

AS_OF = datetime(2025, 2, 22, tzinfo=timezone.utc)


def _art(qid, qtype, forecast, stages=None, error=None):
    return ForecastArtifact(
        qid=qid, horizon="close-7d", as_of=AS_OF, system_name="stub",
        config_fingerprint="abc123def456", forecast=forecast,
        stages=stages or [], error=error)


def _run_dir(tmp_path):
    rd = tmp_path / "run"
    rd.mkdir()
    arts = [
        _art("metaculus:90001", QType.BINARY,
             Forecast(qtype=QType.BINARY, binary_prob=0.8),
             stages=[StageRecord(stage="aggregation",
                                 outputs={"aggregate_raw": 0.74})]),
        _art("metaculus:90002", QType.BINARY,
             Forecast(qtype=QType.BINARY, binary_prob=0.3)),
        _art("metaculus:90005", QType.NUMERIC,
             Forecast(qtype=QType.NUMERIC,
                      cdf=tuple(i / 200 for i in range(201)))),
        _art("metaculus:90003", QType.BINARY, None, error="boom"),
    ]
    (rd / "artifacts.jsonl").write_text(
        "".join(json.dumps(artifact_to_dict(a)) + "\n" for a in arts))
    scores = [
        {"qid": "metaculus:90001", "resolution": True, "baseline_score": 12.0},
        {"qid": "metaculus:90002", "resolution": False, "baseline_score": 30.0},
        {"qid": "metaculus:90005", "resolution": 42.0, "baseline_score": 5.0},
    ]
    (rd / "scores.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in scores))
    return rd


def test_rows_carry_fit_calibration_fields(tmp_path):
    rows = artifacts_to_calibration_rows(_run_dir(tmp_path))
    assert len(rows) == 3                      # failed artifact skipped
    by_qid = {r["question_id"]: r for r in rows}
    r1 = by_qid["metaculus:90001"]
    assert r1["type"] == "binary"
    assert r1["final"] == 0.8
    assert r1["raw"] == 0.74                   # from the aggregation stage
    assert r1["resolution"] == 1               # bool -> 1/0 for binary
    assert r1["config_hash"] == "abc123def456"
    r2 = by_qid["metaculus:90002"]
    assert r2["resolution"] == 0
    assert r2["raw"] is None                   # no aggregation stage logged
    r5 = by_qid["metaculus:90005"]
    assert r5["type"] == "numeric"
    assert r5["final"] is None                 # join skips non-binary anyway


def test_round_trip_into_fit_calibration_join(tmp_path):
    rows = artifacts_to_calibration_rows(_run_dir(tmp_path))
    preds, outcomes = join_resolutions(rows, resolutions_from_rows(rows))
    assert preds == [0.8, 0.3]
    assert outcomes == [1, 0]


def test_written_export_is_forecasts_jsonl_compatible(tmp_path):
    rd = _run_dir(tmp_path)
    out = write_calibration_export(rd, tmp_path / "export" / "calib.jsonl")
    assert out.exists()
    rows = read_forecasts(out)            # the live log reader parses it
    preds, outcomes = join_resolutions(rows, resolutions_from_rows(rows))
    assert preds == [0.8, 0.3]
    assert outcomes == [1, 0]


def test_missing_scores_file_yields_resolution_none(tmp_path):
    rd = _run_dir(tmp_path)
    (rd / "scores.jsonl").unlink()
    rows = artifacts_to_calibration_rows(rd)
    assert len(rows) == 3
    assert all(r["resolution"] is None for r in rows)
    preds, outcomes = join_resolutions(rows, resolutions_from_rows(rows))
    assert preds == [] and outcomes == []      # join skips unresolved rows
