# Optimization framework (backtest/optimize/)

Backtest-driven optimization of bot components, per the incorporation study
(`research-notes/07-incorporation-research.md`) — implemented: items 4
(GEPA-style prompt evolution, scoped) and 5 (playbook loop, limited); item 6
(evolutionary program search) is an OPEN FRAMEWORK ONLY, documented below
and deliberately not implemented.

## The contract

- `Candidate(kind, payload)` — what an optimizer proposes:
  - `"params"` — a `BotConfig` patch (`n_trim`, `extremize_a`, caps, …).
  - `"prompt:binary"` — `{"template": <full prompt template>}`.
  - `"code:*"` — RESERVED for evolutionary program search (item 6).
- `fitness.evaluate(candidate, qids=None) -> EvalResult(score, per_item, n,
  cost_usd)` — score is mean Metaculus baseline score; `per_item[].feedback`
  is the textual reflection substrate.
- `ReplayFitness(run_dir, store, variant)` — kind `params`; replays cached
  member probabilities through `from_member_forecasts`. **$0 per call.**
- `PromptFitness(run_dir, store, variant, llm, budget_usd)` — kind
  `prompt:binary`; replays cached research through the member-LLM stage with
  `prompt_overrides` patched. Un-mutated templates hit the shared llm_cache
  at $0; mutated templates spend member calls, hard-capped by `budget_usd`
  (`BudgetExceeded` stops the loop, best-so-far is returned). Refuses a
  `base_config` whose calibration dir holds a fitted Platt (leakage guard).

## Promotion discipline (the only rule that matters)

1. `temporal_split` the run's questions; optimizers SELECT on train only.
   Val is reported, never selected on.
2. A winner enters the live config exclusively through the experiments
   paired-stats gate on a **temporally later** window of clean questions
   (`backtest run` + `backtest compare`). Optimizer outputs are artifacts
   (`best_prompt.txt`, candidates.jsonl), never config edits.
3. Respect the noise floor (research-notes/07 §1): ~780 tightly-paired
   questions to detect 0.005 Brier at 80% power; n=200 detects ~0.012.
   A loop making many sequential selections on a few hundred questions is
   selecting on noise.
4. Known failure modes, priced in: prompt optimization has diminishing
   returns on frontier models (negative on reasoning models — Metaculus's
   own experiment) and overfits with large train sets (Decagon: 500 examples
   underperformed 20–100). Point GEPA loops at cheap members, small train
   sets.

## Running a prompt evolution

```bash
python scripts/optimize_prompt.py logs/backtest/<run-dir> \
    --corpus data/backtest --spec experiments/<spec>.yaml --variant current \
    --budget-usd 25 --generations 3 --children 2
```

Always pass `--spec`: it reconstructs the variant's exact member config
(profile overrides + patch) so the un-mutated seed template replays from the
llm_cache at $0 — without it a config mismatch silently spends fresh calls.

Outputs under `<run-dir>/optimize/`: `candidates.jsonl` (every evaluated
template + train score + worst-question feedback), `best_prompt.txt`,
`summary.json`. The CLI prints the experiments-variant snippet for gate
validation.

## The playbook loop (item 5)

```bash
python scripts/backtest.py failures <run-dir> --corpus data/backtest --live
python scripts/playbook_update.py <run-dir>/failures.json --live
# review skills/forecasting-playbook.md ## Candidates; promote manually
# ONLY after a {playbook_path: skills/forecasting-playbook.md} variant
# passes the gate vs control
```

## Item 6 — how evolutionary program search plugs in (NOT IMPLEMENTED)

OpenEvolve/ShinkaEvolve-style search slots into the same seam:

1. Candidate kind `code:aggregation` (or `code:calibration`,
   `code:features`) with `payload={"source": <python source>}`.
2. A `CodeFitness` mirroring ReplayFitness: sandbox-exec the candidate
   source to a callable `aggregate(probs) -> float`, replay cached member
   probabilities through it, score. $0 per call — which is the only reason
   evolutionary budgets (100–1000 evaluations) are affordable.
3. The external evolver (OpenEvolve works against any OpenAI-compatible
   endpoint, including OpenRouter) drives propose→evaluate; promotion goes
   through the same temporal-split + experiments-gate rules. Note
   OpenEvolve's self-reported benchmark wins were refuted 0-3 in
   verification — treat it as infrastructure, cap eval budgets (~150–500,
   ShinkaEvolve-style), and confirm winners out-of-window.

Nothing in `SYSTEM_FACTORIES` or `Candidate` needs to change for step 1-2 —
add the new kind, the fitness class, and (if the evolved component should
run as its own system) a `"code"` factory in
`backtest/systems/registry.py`.
