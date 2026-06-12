"""Task 12: Clio-style failure clustering over run artifacts."""
import asyncio
import json
from datetime import datetime, timezone

from backtest.core import (BacktestQuestion, Forecast, ForecastArtifact,
                           QType, StageRecord, artifact_to_dict)
from backtest.corpus.store import CorpusStore
from backtest.reports.failures import analyze_failures

AS_OF = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeLLM:
    async def invoke(self, model, prompt, temperature, run_index=0):
        if "Group these forecasting-failure summaries" in prompt:
            n = prompt.count("\n0: ") + prompt.count("\n1: ") + 1
            return json.dumps({"clusters": [
                {"label": "research misses", "description": "missed news",
                 "members": list(range(n))}]})
        if "q-bad" in prompt:
            return "garbage not json"
        return json.dumps({"summary": "missed an imminent announcement",
                           "primary_cause": "research-miss",
                           "evidence_was_available": False})


def _fixture_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    store = CorpusStore(tmp_path / "corpus")
    rows, artifacts = [], []
    for i, (qid, score, h2h) in enumerate([
            ("t:q-worst", -80.0, -10.0), ("t:q-bad", -40.0, None),
            ("t:q-meh", -10.0, 2.0), ("t:q-fine", 50.0, 5.0)]):
        store.upsert_question(BacktestQuestion(
            qid=qid, platform="t", native_id=str(i), qtype=QType.BINARY,
            title=f"Question {qid}?", resolution_criteria="RC",
            resolution=False))
        rows.append({"qid": qid, "horizon": "close-7d", "variant": "current",
                     "baseline_score": score, "resolution": False,
                     "h2h_market": h2h, "price_at_T": 0.2})
        artifacts.append(ForecastArtifact(
            qid=qid, horizon="close-7d", as_of=AS_OF, system_name="current",
            config_fingerprint="x",
            forecast=Forecast(qtype=QType.BINARY, binary_prob=0.9),
            stages=[StageRecord(stage="research",
                                outputs={"text": "dossier text"}),
                    StageRecord(stage="aggregation",
                                inputs={"member_probs": [0.85, 0.9, 0.95]})]))
    rows.append({"qid": "t:q-worst", "horizon": "close-7d",
                 "variant": "chance", "baseline_score": -99.0,
                 "resolution": False})
    (run_dir / "scores.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows))
    (run_dir / "artifacts.jsonl").write_text(
        "\n".join(json.dumps(artifact_to_dict(a)) for a in artifacts))
    return run_dir, store


def test_analyze_failures_selects_worst_and_clusters(tmp_path):
    run_dir, store = _fixture_run(tmp_path)
    payload = asyncio.run(analyze_failures(
        run_dir, store, FakeLLM(), "fake-model", top=2, variant="current"))
    assert payload["n_failures"] == 2
    qids = [f["qid"] for f in payload["facets"]]
    assert qids == ["t:q-worst", "t:q-bad"]       # worst first, chance excluded
    worst = payload["facets"][0]
    assert worst["primary_cause"] == "research-miss"
    assert worst["wrong_side_of_market"] is True
    assert payload["facets"][1]["primary_cause"] == "unparsed"
    assert payload["clusters"][0]["label"] == "research misses"
    assert (run_dir / "failures.json").exists()
    md = (run_dir / "failures.md").read_text()
    assert "research misses" in md and "t:q-worst" in md


def test_cli_failures_subcommand(tmp_path, monkeypatch, capsys):
    run_dir, store = _fixture_run(tmp_path)
    store.close()
    import backtest.reports.failures as failures_mod

    async def fake_analyze(run_dir_, store_, llm, model, **kw):
        assert llm.live is False                  # replay-only by default
        return {"n_failures": 0, "clusters": []}
    monkeypatch.setattr(failures_mod, "analyze_failures", fake_analyze)
    from scripts.backtest import main
    rc = main(["failures", str(run_dir), "--corpus",
               str(tmp_path / "corpus"), "--top", "2"])
    assert rc == 0
    assert "failures" in capsys.readouterr().out
