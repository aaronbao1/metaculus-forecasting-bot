"""Regression tests for the post-implementation review batch (confirmed
findings from the adversarial review workflow, 2026-06-11)."""
import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


# --- engines: decile grid mapped onto PERCENTILES (TimesFM finding) --------

def test_offsets_from_quantile_grid_interpolates_lower_half():
    from backtest.submodels.engines import (PERCENTILES, _finalize,
                                            offsets_from_quantile_grid)
    deciles = {float(10 * i): 100.0 + (i - 5) * 3.2 for i in range(1, 10)}
    offsets = offsets_from_quantile_grid(deciles, 100.0)
    fc = _finalize(100.0, offsets, "timesfm", {}, 1.0)
    q = fc.quantiles
    # abs=1e-3: the monotonicity pass nudges values after the clamped q2.5
    # by the 1e-4 floor epsilon
    assert q[10.0] == pytest.approx(87.2, abs=1e-3)
    assert q[25.0] == pytest.approx(92.0, abs=1e-3)     # interp q20/q30
    assert q[75.0] == pytest.approx(108.0, abs=1e-3)
    assert q[90.0] == pytest.approx(112.8, abs=1e-3)
    # genuine spread in the lower half — not collapsed onto the median
    assert q[25.0] - q[10.0] > 1.0 and q[50.0] - q[25.0] > 1.0
    assert all(q[a] < q[b] for a, b in zip(PERCENTILES, PERCENTILES[1:]))


# --- evolve: unparseable / placeholder-inventing children ------------------

SEED = "Forecast {question_text} using {research}.\nProbability: ZZ%"


def test_unbalanced_brace_child_skipped_not_crash():
    from backtest.optimize.evolve import reflective_evolve
    from tests.backtest.test_optimize_evolve import TRAIN, FakeFitness

    class BraceReflect:
        async def invoke(self, model, prompt, temperature, run_index=0):
            return 'Use JSON like {"p": 0.5 and... \nProbability: ZZ%'

    res = asyncio.run(reflective_evolve(
        SEED, FakeFitness(), BraceReflect(), "fake", train_qids=TRAIN,
        generations=2, children=1))
    assert res.stopped_reason == "completed"
    assert len(res.history) == 1                  # all children invalid


def test_novel_placeholder_child_rejected():
    from backtest.optimize.evolve import _valid_child
    assert not _valid_child(
        "Forecast {question_text} {research} {novel}.\nProbability: ZZ%",
        SEED)
    assert _valid_child(SEED + "\nextra line", SEED)


# --- temporal_split: multi-horizon duplicates never straddle the cut -------

def test_temporal_split_dedupes_qids():
    from backtest.optimize.fitness import temporal_split
    from tests.backtest.test_optimize_fitness import _q
    qs = [_q(i, True) for i in (0, 1, 2)] * 2     # one per horizon
    train, val = temporal_split(qs, train_frac=0.6)
    train_qids = {q.qid for q in train}
    val_qids = {q.qid for q in val}
    assert not (train_qids & val_qids)
    assert train_qids | val_qids == {"t:0", "t:1", "t:2"}


# --- synthetic: open_time anchors at t0 (bounds leak) + intraday qids ------

def test_synthetic_open_time_is_anchor(tmp_path):
    from backtest.corpus.store import CorpusStore
    from backtest.corpus.synthetic import generate_questions
    store = CorpusStore(tmp_path / "corpus")
    store.put_series("s:d", "target", [
        ((T0 + timedelta(days=i)).isoformat(), 100.0 + i)
        for i in range(200)])
    qs = generate_questions(store, "s:d", n=3, horizon_days=30, seed=1)
    for q in qs:
        # bounds derive from data <= open_time, so any as_of in
        # [open_time, close) sees only pre-as_of-derived bounds
        assert q.open_time == q.scheduled_close_time - timedelta(days=30)


def test_synthetic_intraday_anchors_get_unique_qids(tmp_path):
    from backtest.corpus.store import CorpusStore
    from backtest.corpus.synthetic import generate_questions
    store = CorpusStore(tmp_path / "corpus")
    store.put_series("s:h", "target", [
        ((T0 + timedelta(hours=6 * i)).isoformat(), 100.0 + i)
        for i in range(800)])                      # 4 points/day, 200 days
    qs = generate_questions(store, "s:h", n=40, horizon_days=30, seed=1)
    assert len({q.qid for q in qs}) == len(qs)
    for q in qs:
        assert store.get_question(q.qid).resolution == q.resolution


# --- config_hash: playbook CONTENT fingerprinted ---------------------------

def test_config_hash_tracks_playbook_content(tmp_path):
    from bot.config import BotConfig, config_hash
    pb = tmp_path / "pb.md"
    pb.write_text("# P\n\n## Promoted\n\n- Rule one. <!-- id:a date:2026-01-01 -->\n\n## Candidates\n")
    cfg = BotConfig(playbook_path=str(pb))
    h1 = config_hash(cfg)
    pb.write_text("# P\n\n## Promoted\n\n- Rule two. <!-- id:b date:2026-01-01 -->\n\n## Candidates\n")
    assert config_hash(cfg) != h1
    # Candidates appends do NOT churn the hash
    h2 = config_hash(cfg)
    pb.write_text(pb.read_text() + "- Candidate. <!-- id:c -->\n")
    assert config_hash(cfg) == h2


# --- playbook temporal guard ----------------------------------------------

def test_playbook_temporal_guard(tmp_path):
    from bot.playbook import load_playbook
    pb = tmp_path / "pb.md"
    pb.write_text(
        "## Promoted\n\n"
        "- Old lesson. <!-- id:a source:r date:2026-01-01 -->\n"
        "- Future lesson. <!-- id:b source:r date:2026-06-01 -->\n"
        "- Undated lesson.\n\n## Candidates\n")
    live = load_playbook(pb)
    assert "Old" in live and "Future" in live and "Undated" in live
    asof = load_playbook(pb, as_of=datetime(2026, 3, 1,
                                            tzinfo=timezone.utc))
    assert "Old" in asof
    assert "Future" not in asof                   # postdates as_of
    assert "Undated" not in asof                  # fail closed


# --- research cache key salted by rating config ----------------------------

def test_research_cache_key_salted_for_rated_provider():
    from backtest.systems.ensemble_adapter import (_research_cache_key,
                                                   _research_cfg_salt)
    from bot.config import BotConfig
    rated_a = BotConfig(research_providers=("asknews_rated",),
                        parser_model="model-a")
    rated_b = BotConfig(research_providers=("asknews_rated",),
                        parser_model="model-b")
    plain_a = BotConfig(research_providers=("asknews_historical",),
                        parser_model="model-a")
    plain_b = BotConfig(research_providers=("asknews_historical",),
                        parser_model="model-b")
    t = T0
    k = lambda cfg: _research_cache_key(  # noqa: E731
        "q:1", t, list(cfg.research_providers),
        cfg_salt=_research_cfg_salt(cfg))
    assert k(rated_a) != k(rated_b)               # rating model matters
    assert k(plain_a) == k(plain_b)               # unrated: unchanged keys


# --- failures: baselines excluded by default -------------------------------

def test_failures_excludes_baselines_by_default(tmp_path):
    from backtest.reports.failures import analyze_failures
    from tests.backtest.test_failures import FakeLLM, _fixture_run
    run_dir, store = _fixture_run(tmp_path)
    payload = asyncio.run(analyze_failures(
        run_dir, store, FakeLLM(), "fake-model", top=10))   # no variant
    assert all(f["variant"] != "chance" for f in payload["facets"])


# --- playbook_update: bullets land INSIDE Candidates -----------------------

def test_playbook_update_inserts_inside_candidates_section(tmp_path):
    from scripts.playbook_update import update_playbook
    pb = tmp_path / "pb.md"
    pb.write_text("# P\n\n## Candidates\n\n## Promoted\n\n"
                  "- Sacred. <!-- id:s date:2026-01-01 -->\n")

    async def invoke(prompt, sample_idx):
        return json.dumps({"bullets": [{"lesson": "New lesson.",
                                        "applies_to": "x"}]})

    added = asyncio.run(update_playbook(
        {"run_dir": "r", "clusters": [], "facets": []}, pb, invoke,
        today="2026-06-11"))
    assert added == 1
    text = pb.read_text()
    candidates = text.split("## Candidates")[1].split("## Promoted")[0]
    promoted = text.split("## Promoted")[1]
    assert "New lesson." in candidates
    assert "New lesson." not in promoted


# --- runner: build_variant_config reconstructs run configs -----------------

def test_build_variant_config_matches_build_systems(tmp_path):
    from backtest.corpus.store import CorpusStore
    from backtest.experiments.runner import (_build_systems,
                                             build_variant_config)
    from backtest.experiments.spec import ExperimentSpec
    spec = ExperimentSpec(
        name="t", profile="smoke",
        variants=[{"name": "v1", "patch": {"n_trim": 0}}])
    store = CorpusStore(tmp_path / "corpus")
    systems = _build_systems(spec, store)
    cfg = build_variant_config(spec, "v1")
    assert systems["v1"].cfg == cfg
    with pytest.raises(ValueError, match="not in spec"):
        build_variant_config(spec, "nope")
