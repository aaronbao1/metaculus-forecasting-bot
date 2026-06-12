"""Single source of truth for every tunable. The config hash is logged with
each forecast so calibration fitting and A/B comparison can condition on the
exact configuration that produced a forecast (self-improvement hook)."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime


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
        EnsembleMember("google/gemini-3.1-pro-preview", runs=3, temperature=0.5),
    ]


@dataclass
class BotConfig:
    ensemble: list[EnsembleMember] = field(default_factory=_default_ensemble)
    aggregation_method: str = "trimmed_gmo"
    n_trim: int = 1                   # forecasts dropped per side (trimmed_gmo)
    # Fixed-α Karmarkar extremization between aggregation and calibration.
    # 1.0 = identity. Pre-registered roadmap #1 arm; fitted extremization
    # stays rejected — only fixed, pre-registered values belong here.
    extremize_a: float = 1.0
    min_members: int = 3              # skip question if fewer members survive
    cap_lo: float = 0.02
    cap_hi: float = 0.98
    mc_floor: float = 0.01
    # Per-qtype prompt template overrides ({"binary": "..."}). Empty = the
    # constants in bot/prompts.py. The optimization harness ships winners
    # through this field so candidate prompts are config, never code edits.
    prompt_overrides: dict = field(default_factory=dict)
    # Path to skills/forecasting-playbook.md. None = off; rendered prompts
    # are byte-identical to the no-playbook era (cache-hit preservation).
    playbook_path: str | None = None
    # Research: ALL providers run in parallel and compose into one report —
    # distinct-source count was the #1 Fall-2025 winner predictor (r=0.42).
    research_providers: tuple = ("asknews", "search_llm", "markets")
    research_timeout_s: float = 300.0
    search_model: str = "perplexity/sonar-pro"     # via OpenRouter
    parser_model: str = "openai/gpt-5.4-mini"      # cheap reprompt/repair calls
    max_questions_per_run: int = 50                # cost brake
    calibration_dir: str = "calibration"
    log_path: str = "logs/forecasts.jsonl"
    # Backtesting (Retrocast): pretend "today" is this instant everywhere a
    # date or day-count enters a prompt. None = live behavior, bit-identical
    # to pre-backtest code. Excluded from config_hash so live hashes are
    # unchanged and backtest variants fingerprint identically to live.
    as_of: datetime | None = None

    @property
    def submit_predictions(self) -> bool:
        return os.getenv("SUBMIT_PREDICTION", "false").strip().lower() == "true"


def config_hash(cfg: BotConfig) -> str:
    d = asdict(cfg)
    d.pop("as_of", None)     # backtest-only knob; must not perturb live hashes
    blob = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


DEFAULT_CONFIG = BotConfig()
