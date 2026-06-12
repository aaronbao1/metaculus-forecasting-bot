"""End-to-end smoke pipeline: fixture corpus -> run_experiment with the
StubSystem override -> artifacts.jsonl + scores.jsonl + manifest.json ->
build_report. Exactly what .github/workflows/backtest-smoke.yml runs in CI.
No network, no LLM."""
import json
from pathlib import Path

from backtest.corpus.fixtures import load_fixture_corpus
from backtest.corpus.store import CorpusStore
from backtest.experiments.runner import run_experiment
from backtest.experiments.spec import ExperimentSpec
from backtest.reports.report import build_report
from backtest.systems.stub import StubSystem

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures" / "mini_corpus"
SMOKE_YAML = REPO / "experiments" / "smoke.yaml"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


async def test_smoke_run_with_stub_override_produces_run_outputs(tmp_path):
    store = CorpusStore(tmp_path / "store")
    assert load_fixture_corpus(store, FIXTURES) == 5
    spec = ExperimentSpec.from_yaml(SMOKE_YAML)
    out = tmp_path / "runs"

    await run_experiment(spec, store, out, live_llm=False, yes=True,
                         systems=[StubSystem()])

    # llm_cache is a sibling dir at out_dir level; filter to actual run dirs
    # (run dirs are named <spec.name>-<seed>-<fingerprint>)
    run_dirs = [p for p in out.iterdir()
                if p.is_dir() and not p.name.startswith("llm_cache")]
    assert len(run_dirs) == 1
    rd = run_dirs[0]
    assert (rd / "manifest.json").exists()

    artifacts = _read_jsonl(rd / "artifacts.jsonl")
    assert len(artifacts) == 5                      # 5 questions x 1 horizon
    assert {a["system_name"] for a in artifacts} == {"stub"}
    assert all(a["forecast"] is not None for a in artifacts)

    scores = _read_jsonl(rd / "scores.jsonl")
    assert scores
    assert all("baseline_score" in s for s in scores)

    report_path = build_report(rd)
    assert report_path.exists()
    assert report_path.read_text().strip()
