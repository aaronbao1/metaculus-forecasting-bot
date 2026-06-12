# Forecasting Playbook

Curated heuristics injected into forecast prompts when
`BotConfig.playbook_path` points here. Only bullets under **Promoted** are
injected (see `bot/playbook.py`); **Candidates** are proposals appended by
`scripts/playbook_update.py` from failure post-mortems and are never injected.

Promotion rule (the only path): move a bullet to Promoted ONLY after a
playbook-on experiment variant beats its control through the paired-stats
gate on temporally held-out clean questions (`docs/backtesting/OPTIMIZE.md`).
Every bullet carries provenance: `<!-- id:… source:… date:… -->`.

## Promoted

<!-- none yet — promotion requires a passing experiments-gate variant -->

## Candidates
