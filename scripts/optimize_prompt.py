"""GEPA-style binary-prompt optimization against a cached backtest run.

Usage:
    python scripts/optimize_prompt.py RUN_DIR --corpus data/backtest \\
        --variant current --budget-usd 25 --generations 3

Reads cached research from RUN_DIR's artifacts, evolves the binary prompt
template via reflective mutation (backtest/optimize/evolve.py), evaluating
candidates through the shared llm_cache (un-mutated prompts replay at $0;
mutations spend member calls, hard-capped by --budget-usd). Selection is
train-split only. NEVER writes bot config — the printed variant snippet is
how a winner enters the experiments gate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path

_repo_root = str(_Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
del _repo_root

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(_Path(__file__).resolve().parent.parent / ".env")
del _Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="optimize-prompt",
        description="evolve the binary prompt against a cached backtest run")
    p.add_argument("run_dir")
    p.add_argument("--corpus", default="data/backtest")
    p.add_argument("--variant", default="current",
                   help="run variant whose cached research/members to use")
    p.add_argument("--budget-usd", type=float, default=25.0)
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--children", type=int, default=2)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--reflect-model", default=None,
                   help="reflection model (default: cfg.parser_model)")
    p.add_argument("--seed-file", default=None,
                   help="seed template file (default: bot.prompts"
                        ".BINARY_PROMPT)")
    p.add_argument("--out", default=None,
                   help="output dir (default: RUN_DIR/optimize)")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"run dir not found: {run_dir}")
        return 2

    from backtest.corpus.store import CorpusStore
    from backtest.optimize.evolve import reflective_evolve
    from backtest.optimize.fitness import PromptFitness, temporal_split
    from backtest.timetravel.llm_cache import CachedLLM
    from bot.config import BotConfig
    from bot.prompts import BINARY_PROMPT

    seed_template = (Path(args.seed_file).read_text() if args.seed_file
                     else BINARY_PROMPT)
    store = CorpusStore(Path(args.corpus))
    llm = CachedLLM(run_dir.parent / "llm_cache", live=True)
    fitness = PromptFitness(run_dir, store, args.variant, llm,
                            budget_usd=args.budget_usd)
    if not fitness.questions:
        print(f"no binary questions with cached research for variant "
              f"{args.variant!r} in {run_dir}")
        return 2
    train_qs, val_qs = temporal_split(fitness.questions, args.train_frac)
    print(f"[optimize] {len(train_qs)} train / {len(val_qs)} val questions; "
          f"budget ${args.budget_usd:.2f}")

    reflect_model = args.reflect_model or BotConfig().parser_model
    result = asyncio.run(reflective_evolve(
        seed_template, fitness, fitness.counting, reflect_model,
        train_qids={q.qid for q in train_qs},
        val_qids={q.qid for q in val_qs},
        generations=args.generations, children=args.children))

    out_dir = Path(args.out) if args.out else run_dir / "optimize"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "candidates.jsonl").open("w") as f:
        for row in result.history:
            f.write(json.dumps(row) + "\n")
    (out_dir / "best_prompt.txt").write_text(result.best_template)
    (out_dir / "summary.json").write_text(json.dumps({
        "best_train_score": result.best_train_score,
        "val_score": result.val_score,
        "stopped_reason": result.stopped_reason,
        "candidates": len(result.history),
        "spent_usd": fitness.cost_usd,
        "train_n": len(train_qs), "val_n": len(val_qs),
        "reflect_model": reflect_model,
    }, indent=2))

    print(f"[optimize] best train {result.best_train_score:.2f} "
          f"(val {result.val_score}); spent ${fitness.cost_usd:.2f}; "
          f"{result.stopped_reason} -> {out_dir}")
    print("\nTo validate through the experiments gate, add a variant:\n"
          "  variants:\n"
          "    - name: prompt-evolved\n"
          "      patch:\n"
          "        prompt_overrides:\n"
          f"          binary: \"<contents of {out_dir / 'best_prompt.txt'}>\""
          "\n(run on a temporally LATER window than this run's questions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
