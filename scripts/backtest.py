"""Retrocast CLI: build-corpus | run | replay | compare | report.

replay == run with live_llm=False: every completion must come from the
run's cache (CachedLLM raises CacheMiss otherwise) — $0 re-runs for
post-LLM changes. Heavy imports live inside subcommand functions so
`--help` stays fast and dependency-light.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path

# When executed as `python scripts/backtest.py`, Python adds the scripts/
# directory to sys.path[0]. This file is named backtest.py, which would
# shadow the backtest/ package at the repo root. Fix: insert the repo root
# (parent of the scripts/ dir) before any other path entries.
_repo_root = str(_Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
del _Path, _repo_root
import json
from pathlib import Path


def cmd_build_corpus(args) -> int:
    import asyncio

    from backtest.corpus.store import CorpusStore
    store = CorpusStore(Path(args.corpus))
    if args.source in ("halawi", "forecastbench") and not args.dataset_dir:
        print("--dataset-dir is required for halawi/forecastbench")
        return 2
    if args.source == "metaculus":
        from backtest.corpus.metaculus import fetch_metaculus
        report = asyncio.run(fetch_metaculus(
            store, tournaments=args.tournaments or None, limit=args.limit))
    elif args.source == "polymarket":
        from backtest.corpus.polymarket import fetch_polymarket
        report = asyncio.run(fetch_polymarket(store, limit=args.limit))
    elif args.source == "kalshi":
        from backtest.corpus.kalshi import fetch_kalshi
        report = asyncio.run(fetch_kalshi(store, limit=args.limit))
    elif args.source == "manifold":
        from backtest.corpus.manifold import fetch_manifold
        report = asyncio.run(fetch_manifold(store, limit=args.limit))
    elif args.source == "halawi":
        from backtest.corpus.halawi import import_halawi
        report = import_halawi(store, Path(args.dataset_dir))
    else:  # forecastbench
        from backtest.corpus.forecastbench import import_forecastbench
        report = import_forecastbench(store, Path(args.dataset_dir))
    print(f"fetched={report.fetched} withheld={report.withheld} "
          f"errors={len(report.errors)}")
    return 0


def _cmd_run(args, live_llm: bool) -> int:
    import asyncio

    from backtest.corpus.store import CorpusStore
    from backtest.experiments.runner import run_experiment
    from backtest.experiments.spec import ExperimentSpec
    spec = ExperimentSpec.from_yaml(args.spec)
    store = CorpusStore(Path(args.corpus))
    systems = None
    if getattr(args, "system", None) == "stub":
        # lazy import: StubSystem ships with the CI-gate task (Phase 5)
        from backtest.systems.stub import StubSystem
        systems = [StubSystem()]
    result = asyncio.run(run_experiment(spec, store, Path(args.out),
                                        live_llm=live_llm, yes=args.yes,
                                        systems=systems))
    done = sum(1 for s in result.statuses.values() if s == "done")
    print(f"[retrocast] {result.run_id}: {done}/{len(result.statuses)} "
          f"done -> {result.run_dir}")
    return 0


def cmd_compare(args) -> int:
    from backtest.scoring.stats import paired_bootstrap, paired_t
    scores_path = Path(args.run_dir) / "scores.jsonl"
    rows = []
    if scores_path.exists():
        rows = [json.loads(line)
                for line in scores_path.read_text().splitlines()
                if line.strip()]
    by: dict = {}
    for r in rows:
        by.setdefault(r["variant"], {})[
            (r["qid"], r["horizon"])] = r["baseline_score"]
    if args.variant_a not in by or args.variant_b not in by:
        print(f"variants not found in scores: "
              f"{args.variant_a!r}, {args.variant_b!r} "
              f"(have: {sorted(by)})")
        return 1
    common = sorted(set(by[args.variant_a]) & set(by[args.variant_b]))
    diffs = [by[args.variant_a][k] - by[args.variant_b][k] for k in common]
    if not diffs:
        print("no common (qid, horizon) rows between the two variants")
        return 1
    print(json.dumps({
        "n": len(diffs),
        "baseline_score_diff": paired_bootstrap(diffs),
        "paired_t": paired_t(diffs),
    }, indent=2))
    return 0


def cmd_report(args) -> int:
    from backtest.reports.report import build_report
    path = build_report(Path(args.run_dir))
    print(path)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="backtest",
        description="Retrocast: build corpora, run/replay backtest "
                    "experiments, compare variants, render reports.")
    sub = p.add_subparsers(dest="command", required=True)

    bc = sub.add_parser("build-corpus",
                        help="fetch/import questions into the corpus store")
    bc.add_argument("--source", required=True,
                    choices=["metaculus", "polymarket", "kalshi", "manifold",
                             "halawi", "forecastbench"])
    bc.add_argument("--corpus", default="data/backtest",
                    help="corpus root (default data/backtest)")
    bc.add_argument("--limit", type=int, default=None)
    bc.add_argument("--tournaments", nargs="*", default=None)
    bc.add_argument("--dataset-dir", default=None,
                    help="local dataset dir (halawi/forecastbench)")
    bc.set_defaults(func=cmd_build_corpus)

    for name, live, help_text in (
            ("run", True, "run an experiment (live LLM, cached)"),
            ("replay", False,
             "re-run from the completion cache only (no live LLM)")):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--spec", required=True, help="experiment YAML")
        sp.add_argument("--corpus", default="data/backtest",
                        help="corpus root (default data/backtest)")
        sp.add_argument("--out", default="data/backtest/runs",
                        help="runs output dir")
        sp.add_argument("--yes", action="store_true",
                        help="skip the cost confirmation gate")
        if name == "run":
            sp.add_argument("--system", choices=["stub"], default=None,
                            help="override systems under test; 'stub' runs "
                                 "the deterministic no-network StubSystem "
                                 "(CI smoke gate)")
        sp.set_defaults(func=lambda a, _live=live: _cmd_run(a, _live))

    cp = sub.add_parser("compare",
                        help="paired stats between two variants of one run")
    cp.add_argument("run_dir")
    cp.add_argument("variant_a")
    cp.add_argument("variant_b")
    cp.set_defaults(func=cmd_compare)

    rp = sub.add_parser("report",
                        help="render the markdown report for a run dir")
    rp.add_argument("run_dir")
    rp.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
