"""Budget profiles (spec §6.5).

Cost derivation: calls_per_question ≈ (Σ member.runs) ensemble calls +
~2-3 research calls. The standard ensemble is 3 models × 3 runs = 9 LLM
calls/question at roughly $0.02/call blended (mid-2026 OpenRouter rates,
long forecasting prompts), plus ~$0.05 of research calls → ≈ $0.25 per
question per horizon per variant. The smoke profile overrides the ensemble
to a single member with a single run → ~1 ensemble call + research ≈ $0.02
per question. These are deliberately rough planning numbers: the runner
prints the projection for a human gate and enforces cost_cap_usd as a hard
brake.

NOTE: the smoke ensemble_override (1 member × 1 run) cannot satisfy the
live default min_members=3 — the runner sets min_members=1 whenever it
applies an override.
"""
from __future__ import annotations

from dataclasses import dataclass

from bot.config import EnsembleMember


@dataclass(frozen=True)
class Profile:
    max_questions: int | None
    cost_cap_usd: float
    est_cost_per_question_usd: float
    ensemble_override: list[EnsembleMember] | None = None


PROFILES = {
    # CI-friendly: 30 q × 1 variant × $0.02 ≈ $0.60 projected; cap $5.
    "smoke": Profile(max_questions=30,
                     ensemble_override=[EnsembleMember("openai/gpt-5.4",
                                                       runs=1)],
                     cost_cap_usd=5.0,
                     est_cost_per_question_usd=0.02),
    # 300 q × $0.25 ≈ $75 per variant-horizon; cap $150.
    "standard": Profile(max_questions=300,
                        cost_cap_usd=150.0,
                        est_cost_per_question_usd=0.25),
    # Whole corpus, all strata; cap $500 forces deliberate spend.
    "full": Profile(max_questions=None,
                    cost_cap_usd=500.0,
                    est_cost_per_question_usd=0.25),
}
