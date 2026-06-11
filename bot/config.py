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
    # 3 families x 2 runs = 6 members. Verify slugs with scripts/check_models.py
    # (Gate B) before first live run; update freely — everything downstream is
    # config-driven.
    return [
        EnsembleMember("openai/gpt-5.4", runs=2, temperature=0.5),
        EnsembleMember("anthropic/claude-opus-4-6", runs=2, temperature=0.5),
        EnsembleMember("google/gemini-3.1-pro", runs=2, temperature=0.5),
    ]


@dataclass
class BotConfig:
    ensemble: list[EnsembleMember] = field(default_factory=_default_ensemble)
    aggregation_method: str = "trimmed_gmo"
    min_members: int = 3              # skip question if fewer members survive
    cap_lo: float = 0.02
    cap_hi: float = 0.98
    mc_floor: float = 0.01
    research_providers: tuple = ("asknews", "search_llm")
    search_model: str = "perplexity/sonar-pro"     # via OpenRouter
    parser_model: str = "openai/gpt-5.4-mini"      # cheap reprompt/repair calls
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
