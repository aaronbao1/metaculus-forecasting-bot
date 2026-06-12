"""Task 16: optimize-prompt CLI wiring (evolve/fitness stubbed)."""
import json

import scripts.optimize_prompt as cli
from backtest.optimize.evolve import EvolveResult


class StubFitness:
    cost_usd = 1.25

    def __init__(self, *a, **kw):
        from tests.backtest.test_optimize_fitness import _q
        self.counting = object()
        self.questions = [_q(i, True) for i in range(4)]


def test_cli_writes_outputs(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    async def fake_evolve(seed, fitness, reflect, model, **kw):
        assert kw["train_qids"] == {"t:0", "t:1"}
        assert kw["val_qids"] == {"t:2", "t:3"}
        return EvolveResult(best_template="EVOLVED\nProbability: ZZ%",
                            best_train_score=12.5, val_score=11.0,
                            history=[{"gen": 0, "train_score": 12.5}])

    monkeypatch.setattr(
        "backtest.optimize.fitness.PromptFitness", StubFitness)
    monkeypatch.setattr(
        "backtest.optimize.evolve.reflective_evolve", fake_evolve)
    rc = cli.main([str(run_dir), "--corpus", str(tmp_path / "corpus"),
                   "--budget-usd", "5"])
    assert rc == 0
    out = run_dir / "optimize"
    assert (out / "best_prompt.txt").read_text().startswith("EVOLVED")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["best_train_score"] == 12.5
    assert summary["spent_usd"] == 1.25
    assert "prompt_overrides" in capsys.readouterr().out


def test_cli_missing_run_dir_exits_2(tmp_path):
    assert cli.main([str(tmp_path / "nope")]) == 2
