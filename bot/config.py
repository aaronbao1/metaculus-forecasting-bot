"""Single source of truth for every tunable. The config hash is logged with
each forecast so calibration fitting and A/B comparison can condition on the
exact configuration that produced a forecast (self-improvement hook)."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class EnsembleMember:
    model: str            # OpenRouter slug, e.g. "anthropic/claude-opus-4-6"
    runs: int = 2
    temperature: float = 0.5


def _default_ensemble() -> list[EnsembleMember]:
    # 3 families x 3 runs = 9 members (~10 independent runs is the measured
    # sweet spot — AIA found averaging "mandatory" at that n, with diminishing
    # returns after; winners' median is ~28 LLM calls/question all-in). Verify
    # slugs with scripts/check_models.py (Gate B) before first live run;
    # update freely — everything downstream is config-driven.
    return [
        EnsembleMember("openai/gpt-5.4", runs=3, temperature=0.5),
        EnsembleMember("anthropic/claude-opus-4-6", runs=3, temperature=0.5),
        EnsembleMember("google/gemini-3.1-pro", runs=3, temperature=0.5),
    ]


@dataclass
class BotConfig:
    ensemble: list[EnsembleMember] = field(default_factory=_default_ensemble)
    aggregation_method: str = "trimmed_gmo"
    min_members: int = 3              # skip question if fewer members survive
    cap_lo: float = 0.02
    cap_hi: float = 0.98
    mc_floor: float = 0.01
    # Research: ALL providers run in parallel and compose into one report —
    # distinct-source count was the #1 Fall-2025 winner predictor (r=0.42).
    research_providers: tuple = ("asknews", "search_llm", "markets")
    research_timeout_s: float = 300.0
    search_model: str = "perplexity/sonar-pro"     # via OpenRouter
    parser_model: str = "openai/gpt-5.4-mini"      # cheap reprompt/repair calls
    # Supervisor: when the binary ensemble disagrees hard, one frontier call
    # reconciles instead of blind averaging (~0.005 Brier, AIA; smingers).
    supervisor_enabled: bool = True
    supervisor_model: str = "anthropic/claude-opus-4-6"
    supervisor_trigger_spread: float = 0.35   # max(member)-min(member)
    supervisor_weight: float = 0.5            # log-odds blend vs raw aggregate
    # Triage (Tetlock commandment #1): tier is always classified and logged;
    # routing near-certain questions to a reduced ensemble ships OFF until
    # logged data shows the classifier is trustworthy.
    triage_enabled: bool = False
    triage_fast_runs: int = 1                 # runs/member when routed fast
    max_questions_per_run: int = 50                # cost brake
    calibration_dir: str = "calibration"
    log_path: str = "logs/forecasts.jsonl"

    @property
    def submit_predictions(self) -> bool:
        return os.getenv("SUBMIT_PREDICTION", "false").strip().lower() == "true"


def config_hash(cfg: BotConfig) -> str:
    blob = json.dumps(asdict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


DEFAULT_CONFIG = BotConfig()
