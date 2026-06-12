"""CLI smoke tests: --help, a tiny offline replay run + report, compare.
No network — the tiny run uses only the chance baseline (no LLM)."""
import json
from datetime import datetime, timezone

import pytest

from scripts.backtest import main


def test_top_level_help_lists_subcommands(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("build-corpus", "run", "replay", "compare", "report"):
        assert cmd in out


def test_replay_then_report_tiny_run(tmp_path):
    from backtest.core import BacktestQuestion, QType
    from backtest.corpus.store import CorpusStore
    root = tmp_path / "corpus"
    store = CorpusStore(root)
    store.upsert_question(BacktestQuestion(
        qid="metaculus:9", platform="metaculus", native_id="9",
        qtype=QType.BINARY, title="Tiny?",
        open_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        scheduled_close_time=datetime(2024, 6, 1, tzinfo=timezone.utc),
        actual_resolve_time=datetime(2024, 6, 2, tzinfo=timezone.utc),
        resolution=True))
    spec = tmp_path / "exp.yaml"
    spec.write_text(
        "name: cli-smoke\n"
        "corpus: {qtypes: [binary]}\n"
        "horizons: [close-7d]\n"
        "cutoff_mode: strict\n"
        "profile: smoke\n"
        "variants: []\n"
        "baselines: [chance]\n"
        "seed: 0\n")
    out = tmp_path / "runs"
    assert main(["replay", "--spec", str(spec), "--corpus", str(root),
                 "--out", str(out), "--yes"]) == 0
    run_dirs = [p for p in out.iterdir() if p.is_dir() and p.name != "llm_cache"]
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "manifest.json").exists()
    assert (run_dirs[0] / "scores.jsonl").exists()

    assert main(["report", str(run_dirs[0])]) == 0
    text = (run_dirs[0] / "report.md").read_text()
    assert "Leaderboard" in text
    assert "chance" in text


def test_compare_outputs_paired_stats(tmp_path, capsys):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    rows = []
    for qid, a, b in (("m:1", 10.0, 5.0), ("m:2", 20.0, 12.0)):
        for variant, score in (("a", a), ("b", b)):
            rows.append({"qid": qid, "horizon": "close-7d",
                         "variant": variant, "log_score": -0.3,
                         "baseline_score": score, "brier": None,
                         "h2h_cp": None, "h2h_market": None,
                         "stratum": "clean"})
    with (run_dir / "scores.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    assert main(["compare", str(run_dir), "a", "b"]) == 0
    out = capsys.readouterr().out
    assert '"n": 2' in out
    assert '"mean"' in out


def test_compare_unknown_variant_fails_cleanly(tmp_path, capsys):
    run_dir = tmp_path / "r2"
    run_dir.mkdir()
    (run_dir / "scores.jsonl").write_text("")
    assert main(["compare", str(run_dir), "a", "b"]) == 1
