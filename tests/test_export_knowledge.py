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
