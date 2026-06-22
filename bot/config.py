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
    # Cap the per-call output budget. forecasting_tools' GeneralLlm defaults to
    # 65536, which OpenRouter reserves IN FULL against the key's weekly limit
    # ("requested up to 65536 tokens, can only afford ...") and 402s when the
    # cap is tight — starving the ensemble ("0 members survived"). A forecast's
    # actual output is a few thousand tokens, so 16k is ample headroom.
    max_tokens: int = 16384


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
    # Cost-brake cap on questions forecast PER RUN. Raised from 50 so realistic
    # tournament rounds are never silently truncated; if it ever binds, the
    # forecast_on_tournament override records truncated_count to logs/runs.jsonl.
    max_questions_per_run: int = 200
    # Open-question fetch target. The framework default (num_questions=None) grabs
    # only the first 100-question page; this drives full pagination so a round
    # with >100 open questions is fully fetched.
    max_questions_to_fetch: int = 500
    # Immediate in-run retries for questions that fail transiently (e.g. a 402 or
    # timeout left <min_members survivors). 0 disables; failures then wait for the
    # next cron. Publish is the last per-question step, so a retried question never
    # double-publishes.
    forecast_retry_attempts: int = 2
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
    if cfg.playbook_path:
        # Fingerprint the playbook CONTENT, not just the path: editing
        # '## Promoted' changes every rendered prompt and must change the
        # run identity (manifest-resume safety). Hash the un-filtered
        # promoted block (as_of-independent — as_of must never enter the
        # hash); Candidates appends don't churn it.
        from bot.playbook import load_playbook
        d["playbook_content_sha"] = hashlib.sha256(
            load_playbook(cfg.playbook_path).encode()).hexdigest()
    blob = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


DEFAULT_CONFIG = BotConfig()
