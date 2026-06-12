"""Experiment runner (spec §6.5): cost gate → manifest-resumable
per-(question, horizon, variant) loop with failure isolation →
artifacts.jsonl → scoring pass (scores.jsonl) → paired stats (stats.json).

Resume semantics: statuses "done"/"skipped" are not re-run; "error" is
retried on the next invocation. The scoring pass is idempotent — it rebuilds
scores.jsonl from the latest artifact per (qid, horizon, variant) each run.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

from backtest.core import (ForecastArtifact, artifact_from_dict,
                           artifact_to_dict)
from backtest.corpus.crosswalk import market_for
from backtest.experiments.profiles import PROFILES
from backtest.experiments.spec import ExperimentSpec
from backtest.scoring import stats as stats_mod
from backtest.scoring.metaculus_scores import (baseline_score, brier,
                                               head_to_head, log_score)
from backtest.scoring.opponents import cp_opponent, market_opponent
from backtest.systems.baselines import (ChanceBaseline, CPEchoBaseline,
                                        MarketEchoBaseline)
from backtest.systems.ensemble_adapter import CalibratedEnsembleAdapter
from backtest.timetravel.asof import AsOfContext, horizon_to_asof
from backtest.timetravel.llm_cache import CachedLLM
from backtest.timetravel.providers import install_historical_providers
from bot.config import BotConfig

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    statuses: dict = field(default_factory=dict)
    projected_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0


_BASELINE_FACTORIES = {
    "chance": lambda store: ChanceBaseline(),
    "cp_echo": lambda store: CPEchoBaseline(store),
    "market_echo": lambda store: MarketEchoBaseline(store),
}


def _build_systems(spec: ExperimentSpec, store) -> dict:
    """name -> ForecastSystem. Module-level so tests can monkeypatch it."""
    install_historical_providers()
    # The live BotConfig default providers (incl. the Perplexity track)
    # would do leaky live research — backtests run on the historical set.
    # Variants may still patch research_providers via their spec patch.
    profile = PROFILES[spec.profile]
    base = BotConfig(research_providers=(
        "asknews_historical", "gdelt_wayback",
        "exa_historical", "markets_historical"))
    if profile.ensemble_override is not None:
        # 1-member smoke override cannot satisfy the live min_members=3
        base = replace(base, ensemble=list(profile.ensemble_override),
                       min_members=1)
    systems: dict = {}
    for v in spec.variants:
        system = CalibratedEnsembleAdapter.variant(base=base,
                                                   **v.get("patch", {}))
        system.name = v["name"]
        systems[v["name"]] = system
    for b in spec.baselines:
        factory = _BASELINE_FACTORIES.get(b)
        if factory is None:
            raise ValueError(f"unknown baseline: {b!r} "
                             f"(choose from {sorted(_BASELINE_FACTORIES)})")
        systems[b] = factory(store)
    return systems


def _run_id(spec: ExperimentSpec, systems: dict) -> str:
    fps = "|".join(sorted(s.config_fingerprint() for s in systems.values()))
    return (f"{spec.name}-{spec.seed}-"
            f"{hashlib.sha256(fps.encode()).hexdigest()[:8]}")


def _select_questions(spec: ExperimentSpec, store, profile) -> list:
    corpus = dict(spec.corpus)
    # Query ALL matching questions (no SQL LIMIT) so the later seeded shuffle
    # draws uniformly across all platforms, not just alphabetically-first qids.
    kwargs = {"resolved_only": corpus.get("resolved_only", True),
              "limit": None}
    for key in ("platforms", "qtypes", "open_after", "close_before",
                "stratum", "model_set_hash"):
        if key in corpus:
            kwargs[key] = corpus[key]
    # CorpusStore.query raises for stratum-without-hash; the canonical
    # spec YAML says `stratum: clean` with no hash -> default to the
    # live ensemble's model set.
    if "stratum" in corpus and "model_set_hash" not in corpus:
        from backtest.corpus.model_releases import model_set_hash
        kwargs["model_set_hash"] = model_set_hash(
            [m.model for m in BotConfig().ensemble])
    qs = store.query(**kwargs)

    # Filter to horizon-feasible questions only (at least one horizon in
    # spec.horizons must map to a valid as-of datetime) so wasted "skipped"
    # cells don't consume the cap.
    qs = [q for q in qs
          if any(horizon_to_asof(q, h) is not None for h in spec.horizons)]

    # Deterministic seeded shuffle: same seed+corpus → same selection order.
    import random
    random.Random(spec.seed).shuffle(qs)

    # Truncate to the tighter of profile.max_questions / corpus["limit"].
    caps = [c for c in (profile.max_questions, corpus.get("limit")) if c]
    cap = min(caps) if caps else None
    if cap is not None:
        qs = qs[:cap]

    # Stable iteration order within the selected set.
    return sorted(qs, key=lambda q: q.qid)


def _write_manifest(path: Path, manifest: dict) -> None:
    """Atomic manifest write: write to a temp file in the same directory,
    then os.replace() so readers always see a complete JSON document."""
    data = json.dumps(manifest, indent=2)
    dir_ = path.parent
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


async def run_experiment(spec: ExperimentSpec, store, out_dir: Path, *,
                         live_llm: bool = True,
                         yes: bool = False,
                         systems: list | None = None,
                         probe: bool = False) -> RunResult:
    profile = PROFILES[spec.profile]
    questions = _select_questions(spec, store, profile)
    systems_map = (_build_systems(spec, store) if systems is None
                   else {s.name: s for s in systems})
    run_id = _run_id(spec, systems_map)
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Cost estimate + hard brake + human gate (baselines spend nothing,
    # so only spec.variants count).
    # Resume-aware: load the manifest first so we can exclude cells that are
    # already done/skipped from the projection.
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        _existing = json.loads(manifest_path.read_text())
        _existing_statuses = _existing.get("statuses", {})
    else:
        _existing_statuses = {}

    remaining_cells = sum(
        1
        for q in questions
        for horizon in spec.horizons
        for v in spec.variants
        if _existing_statuses.get(f"{q.qid}|{horizon}|{v['name']}")
        not in ("done", "skipped")
    )
    projected = remaining_cells * profile.est_cost_per_question_usd
    print(f"[retrocast] run {run_id}: {len(questions)} questions x "
          f"{len(spec.horizons)} horizons x {len(spec.variants)} LLM "
          f"variants (+{len(spec.baselines)} free baselines) "
          f"~= ${projected:.2f} (cap ${profile.cost_cap_usd:.2f})")
    if projected > profile.cost_cap_usd:
        raise RuntimeError(
            f"cost brake: projected ${projected:.2f} exceeds "
            f"'{spec.profile}' cap ${profile.cost_cap_usd:.2f}")
    if not yes:
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("[retrocast] aborted before spend")
            return RunResult(run_id=run_id, run_dir=run_dir, statuses={},
                             projected_cost_usd=projected)

    # Load or initialise the manifest (manifest_path was defined above in the
    # pre-flight block, so it's already set correctly).
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"run_id": run_id, "spec_name": spec.name,
                    "corpus_root": str(getattr(store, "root", "")),
                    "statuses": {}, "actual_cost_usd": 0.0}
    statuses: dict = manifest["statuses"]
    # Carry forward any spend already accumulated in a previous (partial) run.
    actual_cost_usd: float = float(manifest.get("actual_cost_usd", 0.0))

    # Cache lives at out_dir level, shared across run_ids: keys are
    # content-addressed (model/prompt/temperature/run_index), so sharing is
    # safe and post-LLM-only config changes (new run_id) replay at $0.
    llm = CachedLLM(Path(out_dir) / "llm_cache", live=live_llm)
    artifacts_path = run_dir / "artifacts.jsonl"

    # Probe: when enabled + live_llm, check contamination-confirmed for
    # "contaminated" (or absent) stratum questions before forecasting.
    if probe and live_llm:
        from backtest.corpus.model_releases import model_set_hash
        from backtest.timetravel.probe import memorization_probe
        bot_cfg = BotConfig()
        models = [m.model for m in bot_cfg.ensemble]
        msh = model_set_hash(models)
        for q in questions:
            stratum = store.get_contamination(q.qid, msh)
            if stratum not in ("clean", "contaminated-confirmed"):
                # stratum is "contaminated" or None — run probe
                results = await memorization_probe(q, models, llm)
                if any(results.values()):
                    store.set_contamination(q.qid, msh, "contaminated-confirmed")

    for q in questions:
        for horizon in spec.horizons:
            as_of = horizon_to_asof(q, horizon)
            for name, system in systems_map.items():
                key = f"{q.qid}|{horizon}|{name}"
                if statuses.get(key) in ("done", "skipped"):
                    continue
                if as_of is None:
                    statuses[key] = "skipped"
                    _write_manifest(manifest_path, manifest)
                    continue
                # I4 — already-decided-at-T guard:
                # Skip cells where the question resolved before the as_of
                # point — forecasting an already-resolved question is not
                # forecasting.  Corpus resolve times are currently NULL for
                # most rows, so this guard is future-proofing (safe no-op when
                # actual_resolve_time is None).
                if (q.actual_resolve_time is not None
                        and q.actual_resolve_time <= as_of):
                    statuses[key] = "skipped"
                    _write_manifest(manifest_path, manifest)
                    continue
                ctx = AsOfContext(as_of=as_of, cutoff_mode=spec.cutoff_mode,
                                  llm=llm, store=store)
                try:
                    artifact = await system.forecast(q, ctx)
                except Exception as e:
                    # systems must not raise, but a buggy one must not
                    # kill the run either
                    logger.exception("system %s crashed on %s", name, key)
                    artifact = ForecastArtifact(
                        qid=q.qid, horizon=horizon, as_of=as_of,
                        system_name=name,
                        config_fingerprint=system.config_fingerprint(),
                        forecast=None, cutoff_mode=spec.cutoff_mode,
                        error=f"{type(e).__name__}: {e}")
                artifact.horizon = horizon   # systems can't know the label
                with artifacts_path.open("a") as f:
                    f.write(json.dumps(artifact_to_dict(artifact)) + "\n")
                statuses[key] = ("done" if artifact.error is None
                                 else "error")

                # Accumulate actual spend and persist it before the brake check
                # so the manifest is always valid (i.e. resumable) even if we raise.
                cell_cost = sum(s.cost_usd for s in artifact.stages)
                actual_cost_usd += cell_cost
                manifest["actual_cost_usd"] = actual_cost_usd
                _write_manifest(manifest_path, manifest)

                if actual_cost_usd > profile.cost_cap_usd:
                    raise RuntimeError(
                        f"cost brake (actual spend): ${actual_cost_usd:.4f} "
                        f"exceeds '{spec.profile}' cap "
                        f"${profile.cost_cap_usd:.2f}")

    # After the probe may have upgraded strata, compute the model_set_hash so
    # _score_run can refresh stratum from the store (probe-confirmed strata
    # are in the store but may not be reflected in the artifact if a stub
    # system bypassed the contamination lookup).
    from backtest.corpus.model_releases import model_set_hash as _msh
    try:
        msh_for_scoring = _msh([m.model for m in BotConfig().ensemble])
    except Exception:
        msh_for_scoring = None

    _score_run(run_dir, store, seed=spec.seed, model_set_hash=msh_for_scoring)
    return RunResult(run_id=run_id, run_dir=run_dir,
                     statuses=dict(statuses), projected_cost_usd=projected,
                     actual_cost_usd=actual_cost_usd)


def _score_run(run_dir: Path, store, seed: int = 0,
               model_set_hash: str | None = None) -> None:
    artifacts_path = run_dir / "artifacts.jsonl"
    if not artifacts_path.exists():
        return
    latest: dict = {}
    with artifacts_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            a = artifact_from_dict(json.loads(line))
            latest[(a.qid, a.horizon, a.system_name)] = a

    rows = []
    for (qid, horizon, variant), a in sorted(latest.items()):
        if a.forecast is None:
            continue
        q = store.get_question(qid)
        if q is None or q.resolution is None:
            continue
        # Prefer a fresh store lookup so probe-upgraded strata
        # ("contaminated-confirmed") are reflected even when the artifact
        # was written before the probe ran (e.g. stub systems).
        stratum = a.contamination
        if model_set_hash is not None:
            try:
                fresh = store.get_contamination(qid, model_set_hash)
                if fresh is not None:
                    stratum = fresh
            except Exception:
                pass
        try:
            row = {"qid": qid, "horizon": horizon, "variant": variant,
                   "log_score": log_score(a.forecast, q),
                   "baseline_score": baseline_score(a.forecast, q),
                   "brier": None, "h2h_cp": None, "h2h_market": None,
                   "stratum": stratum,
                   "resolution": q.resolution,
                   # I4 — price provenance; I5 — platform dimension;
                   # I7 — staleness timestamp
                   "platform": q.platform if q is not None else None,
                   "price_at_T": None,
                   "price_at_T_ts": None}
        except Exception as e:
            logger.warning("scoring failed for %s/%s/%s: %s",
                           qid, horizon, variant, e)
            continue
        try:
            row["brier"] = brier(a.forecast, q)
        except Exception as e:
            logger.warning("brier failed for %s/%s/%s: %s",
                           qid, horizon, variant, e)
        try:
            cp = store.value_at(qid, "cp", a.as_of)
            if cp is not None:
                row["h2h_cp"] = head_to_head(a.forecast, cp_opponent(cp), q)
        except Exception as e:
            logger.warning("h2h_cp failed for %s: %s", qid, e)
        try:
            mqid = market_for(store, qid)
            # I4 + I7: record price_at_T and its timestamp using value_at_with_ts
            if mqid is not None:
                price_result = store.value_at_with_ts(
                    mqid, "market_price", a.as_of)
                if price_result is not None:
                    price, price_ts = price_result
                    row["price_at_T"] = float(price)
                    row["price_at_T_ts"] = price_ts
                    row["h2h_market"] = head_to_head(
                        a.forecast, market_opponent(float(price)), q)
        except Exception as e:
            logger.warning("h2h_market failed for %s: %s", qid, e)
        rows.append(row)

    with (run_dir / "scores.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    _paired_stats(run_dir, rows, seed)


def _paired_stats(run_dir: Path, rows: list, seed: int) -> None:
    """Paired bootstrap on baseline_score diffs over the common
    (qid, horizon) set, for every variant pair."""
    by_variant: dict = {}
    for r in rows:
        by_variant.setdefault(r["variant"], {})[
            (r["qid"], r["horizon"])] = r["baseline_score"]
    out = {}
    variants = sorted(by_variant)
    for i, a in enumerate(variants):
        for b in variants[i + 1:]:
            common = sorted(set(by_variant[a]) & set(by_variant[b]))
            diffs = [by_variant[a][k] - by_variant[b][k] for k in common]
            if not diffs:
                continue
            out[f"{a}__vs__{b}"] = {
                "n": len(diffs),
                "baseline_score_diff": stats_mod.paired_bootstrap(
                    diffs, seed=seed),
            }
    (run_dir / "stats.json").write_text(json.dumps(out, indent=2))
