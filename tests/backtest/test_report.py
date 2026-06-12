"""build_report renders every required section from a fabricated run dir
(no corpus store: corpus_root empty -> diagnostics degrade gracefully)."""
import json
from datetime import datetime, timezone

from backtest.core import (Forecast, ForecastArtifact, LedgerEntry, QType,
                           StageRecord, artifact_to_dict)
from backtest.reports.report import build_report


def _write_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps(
        {"run_id": "demo-0-abc12345", "corpus_root": "",
         "statuses": {"metaculus:1|close-7d|a": "done"}}))
    rows = [
        {"qid": "metaculus:1", "horizon": "close-7d", "variant": "a",
         "log_score": -0.2, "baseline_score": 20.0, "brier": 0.04,
         "h2h_cp": 5.0, "h2h_market": None, "stratum": "clean"},
        {"qid": "metaculus:2", "horizon": "close-7d", "variant": "a",
         "log_score": -0.4, "baseline_score": 10.0, "brier": 0.09,
         "h2h_cp": 3.0, "h2h_market": None, "stratum": "contaminated"},
        {"qid": "metaculus:1", "horizon": "close-7d", "variant": "chance",
         "log_score": -0.69, "baseline_score": 0.0, "brier": 0.25,
         "h2h_cp": None, "h2h_market": None, "stratum": "clean"},
        {"qid": "metaculus:2", "horizon": "close-7d", "variant": "chance",
         "log_score": -0.69, "baseline_score": 0.0, "brier": 0.25,
         "h2h_cp": None, "h2h_market": None, "stratum": "contaminated"},
    ]
    with (run_dir / "scores.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    art = ForecastArtifact(
        qid="metaculus:1", horizon="close-7d",
        as_of=datetime(2024, 5, 25, tzinfo=timezone.utc),
        system_name="a", config_fingerprint="fp",
        forecast=Forecast(qtype=QType.BINARY, binary_prob=0.8),
        stages=[StageRecord(stage="member_llm", model="fake/model-a",
                            cost_usd=0.12, tokens_in=100, tokens_out=50)],
        evidence_ledger=[
            LedgerEntry(provider="asknews", query="q",
                        date_trust="snapshot"),
            LedgerEntry(provider="serper", query="q", date_trust="low")])
    with (run_dir / "artifacts.jsonl").open("w") as f:
        f.write(json.dumps(artifact_to_dict(art)) + "\n")
    return run_dir


def test_build_report_renders_all_sections(tmp_path):
    run_dir = _write_run_dir(tmp_path)
    path = build_report(run_dir)
    assert path == run_dir / "report.md"
    text = path.read_text()
    assert "## Leaderboard" in text
    # ranked by mean baseline score: 'a' (15.0) above 'chance' (0.0)
    assert text.index("| a |") < text.index("| chance |")
    assert "[" in text                       # bootstrap CI rendered for h2h
    assert "## Mean baseline score by contamination stratum" in text
    assert "contaminated" in text
    assert "## Diagnostics" in text          # degrades without a store
    assert "## Evidence-ledger leakage summary" in text
    assert "| snapshot | 1 |" in text
    assert "| low | 1 |" in text
    assert "## Cost" in text
    assert "0.1200" in text
    assert "## Worst 10 by log score" in text
    assert "metaculus:1" in text


def test_build_report_tolerates_empty_run_dir(tmp_path):
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    path = build_report(run_dir)
    assert path.exists()
    assert "Leaderboard" in path.read_text()
