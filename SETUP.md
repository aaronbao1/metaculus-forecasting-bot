# Setup Guide — Metaculus Ensemble Forecasting Bot

A calibrated **ensemble scaffold bot** for the Metaculus **FutureEval** AI
forecasting tournaments. It forecasts binary, multiple-choice, numeric, and
date questions by running a 3-family × 3-run LLM ensemble over parallel
multi-source research, then aggregating + calibrating into a Metaculus-valid
submission.

This guide takes you from a fresh clone to a live, prize-eligible bot, using a
five-stage **gate ladder** (A→E) that never submits anything until you
explicitly opt in.

> **How submission actually works:** your bot "competes" simply by forecasting
> tournament questions with its account token. Every question it forecasts gets
> it ranked on that tournament's **leaderboard** (where the prize money is).
> Custom scaffold bots do **not** appear on the FutureEval *model* leaderboard —
> that one is Metaculus's own house models. See `research-notes/05-metaculus-mechanics.md`.

---

## 1. Prerequisites

- **Python ≥ 3.11** (CI uses 3.12).
- **[uv](https://docs.astral.sh/uv/)** for env + dependency management (`uv 0.11+`).
- A **GitHub account** (for the every-20-min CI cron that runs the bot live).
- **`gh` CLI** authenticated with `repo` + `workflow` scopes (for secrets + go-live).
- API keys (see §3): a Metaculus **bot** token, an OpenRouter key, and (optional
  but strongly recommended) AskNews credentials.

---

## 2. Clone & install

```bash
git clone https://github.com/aaronbao1/metaculus-forecasting-bot.git
cd metaculus-forecasting-bot

uv sync                      # creates .venv from uv.lock, installs the package
# (CI equivalent: pip install -e .)
```

Run anything through `uv run`, e.g. `uv run python scripts/...` or `uv run pytest`.

Optional extras (not needed for the live bot):
```bash
uv sync --extra dev          # pytest (already used by `uv run pytest`)
uv sync --extra timeseries   # statsforecast submodels (backtesting only)
uv sync --extra tsfm         # Chronos / TimesFM (backtesting only)
```

---

## 3. Get your API keys

| Key | Where | Notes |
|---|---|---|
| `METACULUS_TOKEN` | metaculus.com → create a **bot account** → token (see [participate page](https://www.metaculus.com/futureeval/participate/)) | Must be a **bot-account** token, not your personal one. Used to read questions *and* post forecasts. |
| `OPENROUTER_API_KEY` | [openrouter.ai → Keys](https://openrouter.ai/keys) | One key covers all model families. **Set a generous weekly limit** — see §7. |
| `ASKNEWS_CLIENT_ID` / `ASKNEWS_SECRET` | [asknews.app](https://asknews.app) | OAuth pair (preferred). The bot degrades gracefully without it, but research diversity is the #1 score predictor. |
| `PERPLEXITY_API_KEY`, `EXA_API_KEY` | optional | Extra research providers / backtesting. Bot works without them. |

**Free tournament credits:** Metaculus sponsors OpenRouter-compatible LLM
credits (Anthropic/Google/OpenAI) + an AskNews tier, requested each season via
the form on the [FutureEval resources page](https://www.metaculus.com/notebooks/38928/).
If issued a Metaculus proxy key, point `OPENROUTER_API_KEY` at it.

---

## 4. Configure `.env`

```bash
cp .env.template .env
# then edit .env and paste your keys
```

`.env` is gitignored. Put values directly after `=` with nothing else on the
line (this dotenv build does **not** strip inline `# comments`). Leave the
safety flag off:

```ini
SUBMIT_PREDICTION=false       # bot only publishes when this is exactly "true"
```

---

## 5. The gate ladder (run in order)

Each gate must pass before the next. Nothing is submitted until **Gate D**.

### Gate A — offline tests (no keys, no spend)
```bash
uv run pytest -q              # expect: 508 passed, 1 skipped
```

### Gate B — model preflight (cheap live calls)
```bash
uv run python scripts/check_models.py    # must exit 0
```
One trivial call per configured slug — catches wrong/renamed model slugs, dead
keys, and quota issues before a real question does. If this 402s, your
OpenRouter weekly limit is too low (§7).

### Gate C — full-pipeline dry run (real research/ensemble spend, **no submission**)
```bash
uv run python scripts/run_testing_area.py    # SUBMIT_PREDICTION still false
```
Runs the whole bot against the **`bot-testing-area`** tournament (one question
of every type) and prints every *would-be* submission. Read them — confirm
binary / MC / numeric / date all produce sane numbers. Ends with
`submit_predictions=False (DRY RUN — nothing submitted)`.

### Gate D — real submit to the practice tournament
```bash
# edit .env: SUBMIT_PREDICTION=true   (exact lowercase "true")
uv run python scripts/run_testing_area.py
# then revert: SUBMIT_PREDICTION=false
```
This **publishes** forecasts + a private reasoning comment per question to
`bot-testing-area` (not prize-scored — it's the safe place to verify end-to-end
posting). Confirm on your bot's Metaculus profile that forecasts + comments
appear.

> ⚠️ `skip_previously_forecasted_questions=True`: once a question is forecast,
> later runs skip it — a bad first forecast is **sticky**. Verify Gate D output.

### Gate E — live seasonal tournament (local run)
```bash
# with SUBMIT_PREDICTION=true:
uv run python scripts/run_tournament.py
```
Forecasts the live season (`summer-futureeval-2026`, id 33022) **+ MiniBench**.
Only do this once Gate D looks right. For continuous operation, use CI (§6)
instead of running this by hand.

---

## 6. Going live via GitHub Actions (continuous)

Tournament questions open in short ~1–1.5 h windows and are **spot-scored at
close**, so the bot must run continuously. [`.github/workflows/run_bot.yaml`](.github/workflows/run_bot.yaml)
runs `scripts/run_tournament.py` every 20 minutes (`7,27,47 * * * *`),
skipping already-forecast questions.

**Stage the secrets + variable** (one-time):
```bash
gh secret set METACULUS_TOKEN
gh secret set OPENROUTER_API_KEY
gh secret set ASKNEWS_CLIENT_ID
gh secret set ASKNEWS_SECRET
gh variable set SUBMIT_PREDICTION --body false   # keep false until ready
```

> 🔴 **Spend gotcha:** `SUBMIT_PREDICTION=false` does **not** stop spend — a dry
> run still runs full research + the 9-member ensemble every 20 min; it only
> skips the final POST. The thing that actually gates spend is whether the
> **workflow is enabled**. Keep it disabled (`gh workflow disable run-bot`)
> until you're ready to spend.

**Flip it live** (after confirming your credit limits — §7):
```bash
gh workflow enable run-bot
gh variable set SUBMIT_PREDICTION --body true
```
The next `:07/:27/:47` cron forecasts + submits live. Watch the first run under
the repo's **Actions** tab and on your bot's Metaculus profile.

To pause: `gh workflow disable run-bot` (and/or set the variable back to `false`).

---

## 7. Operational limits to watch (these gate *quality*, not just on/off)

1. **OpenRouter weekly credit cap.** Members request up to 65536 `max_tokens`
   (the `forecasting_tools` default). If the weekly limit is low you'll see
   `402 "requires more credits, or fewer max_tokens"` and members silently drop
   — if fewer than `min_members` (3) survive, the question is skipped entirely.
   Fix: raise the key's weekly limit at openrouter.ai, **or** lower `max_tokens`
   via `GeneralLlm(..., max_tokens=…)` in [`bot/ensemble_bot.py`](bot/ensemble_bot.py).
2. **AskNews quota / rate limits.** `429` (rate) and `429001` (concurrency)
   throttling drops research to 2 of 3 sources. Renew/raise the tournament tier
   each season (AskNews Discord / `contact@asknews.app`).
3. **Cost brake.** `max_questions_per_run=50` caps spend per cron tick. With
   9 ensemble calls/question + research, a full tick is real money — confirm
   budget tolerance before enabling the cron.

**Required for prizes:** a post-season bot-maker survey (all participants,
winners or not). Skipping it forfeits *next* season's eligibility too.

---

## 8. Configuration reference — [`bot/config.py`](bot/config.py)

Everything tunable lives in `BotConfig` (the config hash is logged with each
forecast for A/B analysis). Key knobs:

| Field | Default | Meaning |
|---|---|---|
| `ensemble` | gpt-5.4 ×3, claude-opus-4-6 ×3, gemini-3.1-pro-preview ×3 | model families × runs (~9 calls/question) |
| `aggregation_method` / `n_trim` | `trimmed_gmo`, 1 | trimmed geometric-mean-of-odds |
| `cap_lo` / `cap_hi` | 0.02 / 0.98 | final binary clamp (defensive anti-extremizing) |
| `min_members` | 3 | skip the question if fewer ensemble members survive |
| `research_providers` | asknews, search_llm, markets | run in parallel, composed into one dossier |
| `max_questions_per_run` | 50 | per-run cost brake |
| `search_model` / `parser_model` | sonar-pro / gpt-5.4-mini | research + cheap repair calls |

Change models freely — then **re-run Gate B** to validate the new slugs.

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Gate B `402 requires more credits` | OpenRouter weekly cap too low (§7-1). |
| `RateLimitExceededError 429000/429001` | AskNews throttling — research degrades to 2 sources, not fatal (§7-2). |
| `member <model> failed: ... json response` | Transient LLM error; tolerated as long as ≥ `min_members` survive. |
| `NumericDistribution ... too close / too concentrated` | CDF step on a platform reject boundary — fixed in `bot/distributions.py` (enforced inside the 5e-05 / 0.2 limits). If it recurs, check `MIN_STEP`/`MAX_STEP`. |
| CI cron "ran" but nothing posted | Missing secrets, `SUBMIT_PREDICTION` variable not `true`, or workflow disabled. |
| Date question `NotImplementedError` | Stale code — pull latest (`_run_forecast_on_date` lives in `bot/ensemble_bot.py`). |

---

## 10. Layout

```
bot/            core engine: config, ensemble, research, aggregation, calibration,
                distributions (CDF construction), parsing, prompts, markets
scripts/        check_models.py (B) · run_testing_area.py (C/D) · run_tournament.py (E)
backtest/       Retrocast backtesting sidecar (offline eval; not on the live path)
tests/          pytest suite (Gate A)
.github/workflows/run_bot.yaml   the every-20-min live cron
.env.template   copy to .env and fill in keys
FORECASTING-BOT-RESEARCH.md      strategy/research synthesis
research-notes/05-metaculus-mechanics.md   exact scoring + submission mechanics
```
