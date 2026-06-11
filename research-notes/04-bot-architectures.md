# Architecture Notes: Open-Source Metaculus Forecasting Bots & the forecasting-tools Ecosystem

*Research date: 2026-06-11. Methods: shallow-cloned repos (`Metaculus/metac-bot-template`, `Metaculus/forecasting-tools`, `Metaculus/metac-bot`, `Panshul42/Forecasting_Bot_Q2`) into `/tmp/metac-research/`, plus web research including extracting the raw markdown out of Metaculus's JS-rendered notebook pages.*

---

## 1. The Official Metaculus Bot Template — `Metaculus/metac-bot-template`

**Repo:** https://github.com/Metaculus/metac-bot-template
**Current state (June 2026):** the main class is `SummerTemplateBot2026(ForecastBot)` in `main.py` — "a copy of what is used by Metaculus to run the Metac Bots in our benchmark." The docstring admits "our track record has been 1 mentionable bug per season (affecting forecasts for 1-2% of total questions)."

### Repo layout
- `main.py` (745 lines) — recommended template, built on the `forecasting-tools` package.
- `main_with_no_framework.py` (1,648 lines) — same logic with minimal deps (raw `requests` + `litellm` + numpy); controlled by constants at top (`SUBMIT_PREDICTION`, `NUM_RUNS_PER_QUESTION = 5  # The median forecast is taken between NUM_RUNS_PER_QUESTION runs`).
- `bot_helpers.py` — env validation/banners.
- `integrations/main_lightningrod_eval.py` — LightningRod SDK example (news → synthetic question generation → labeling → multi-model eval), see https://github.com/lightning-rod-labs/lightningrod-python-sdk.
- GitHub Actions workflows: `run_bot_on_tournament.yaml` (every 20 min on the live AIB tournament + MiniBench), `test_bot.yaml` (manual smoke test vs https://www.metaculus.com/tournament/bot-testing-area/), `run_bot_on_metaculus_cup.yaml` (every 2 days). Designed so a newcomer can fork, paste `METACULUS_TOKEN` + `OPENROUTER_API_KEY` secrets, and be live with zero code.

### Overall flow (from the class docstring)
> Load questions from Metaculus → for each question: execute `run_research` × `research_reports_per_question`; execute respective `run_forecast` function `predictions_per_research_report × research_reports_per_question` times; aggregate the predictions; submit prediction...

Default config: `research_reports_per_question=1`, `predictions_per_research_report=5`, i.e. **1 research pass, 5 forecasts, aggregated** (median for binary — see §2).

### Research generation
`run_research` dispatches on the configured `"researcher"` llm string:
- `"asknews/news-summaries"` or `"asknews/deep-research/{low,medium,high}-depth"` → `AskNewsSearcher().call_preconfigured_version(...)`
- `"smart-searcher/<model>"` → `SmartSearcher(model, num_searches_to_run=2, num_sites_per_search=10)` (Exa-backed, see §2)
- any `GeneralLlm` (e.g. `perplexity/sonar-pro`, `gpt-4o-search-preview`) → direct invoke
- `"None"`/`"no_research"` → empty string.

Verbatim research prompt:
```
You are an assistant to a superforecaster.
The superforecaster will give you a question they intend to forecast on.
To be a great assistant, you generate a concise but detailed rundown of the most relevant news,
including if the question would resolve Yes or No based on current information.
You do not produce forecasts yourself.

Question:
{question_text}

This question's outcome will be determined by the specific criteria below:
{resolution_criteria}

{fine_print}
```

### The forecasting prompts (verbatim, from `main.py`)

**Binary:**
```
You are a professional forecaster interviewing for a job.

Your interview question is:
{question_text}

Question background:
{background_info}

This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{resolution_criteria}

{fine_print}

Your research assistant says:
{research}

Today is {YYYY-MM-DD}.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) A brief description of a scenario that results in a No outcome.
(d) A brief description of a scenario that results in a Yes outcome.

You write your rationale remembering that good forecasters put extra weight on the status quo outcome
since the world changes slowly most of the time.

The last thing you write is your final answer as: "Probability: ZZ%", 0-100
```
The output is parsed by `structure_output(...)` (LLM-based reparser with `num_validation_samples=2`), then clamped to `[0.01, 0.99]`.

**Multiple choice** — same framing plus:
```
Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) A description of an scenario that results in an unexpected outcome.

You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome
since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability
on most options to account for unexpected outcomes.

The last thing you write is your final probabilities for the N options in this order {options} as:
Option_A: Probability_A
...
```

**Numeric** — adds units, bound messages (`"The outcome can not be higher than X"` for closed bounds vs `"The question creator thinks the number is likely not higher than X"` for open bounds), and:
```
Formatting Instructions:
- Please notice the units requested...
- Never use scientific notation.
- Always start with a smaller number (more negative if negative) and then increase from there...

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The outcome if nothing changed.
(c) The outcome if the current trend continued.
(d) The expectations of experts and markets.
(e) A brief description of an unexpected scenario that results in a low outcome.
(f) A brief description of an unexpected scenario that results in a high outcome.

You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals
to account for unknown unknowns.

The last thing you write is your final answer as:
Percentile 10: XX (lowest number value)
Percentile 20: XX
Percentile 40: XX
Percentile 60: XX
Percentile 80: XX
Percentile 90: XX (highest number value)
```
(Note: deliberately **no 0/50/100 percentiles**; 6 declared percentiles.) Summer 2026 added date questions (same structure, YYYY-MM-DD output) and conditional questions (forecast parent/child/yes/no legs separately, reaffirm pre-existing parent/child forecasts, with an explicit disclaimer that the model must "only forecast the **CHILD** question, given the parent question's resolution").

### Numeric CDF construction
The LLM's 6 percentiles become a `NumericDistribution` (`forecasting_tools/data_models/numeric_report.py`):
1. Declared percentiles are validated (strictly increasing values, minimum spacing, bound checks; a `_check_and_update_repeating_values` pass jitters duplicates).
2. Bounds are added as explicit endpoint percentiles; values are mapped to "cdf location" space ∈ [0,1] (with log-scaling support via the question's `zero_point` — `_cdf_location_to_nominal_location` handles `deriv_ratio` log scaling).
3. **Linear interpolation** between declared points is evaluated at 201 evenly spaced locations (`cdf_size`, default 201) to produce the Metaculus-API continuous CDF.
4. `_standardize_cdf` enforces Metaculus API constraints (per https://metaculus.com/api/ "CDF generation details"): zero mass outside closed bounds, minimum mass outside open bounds, minimum step `0.01/200 = 0.0005`, maximum step 0.2 (PMF cap).

### Aggregation (how the 5 runs combine)
Implemented in the data models (see §2): **binary = median**; **MC = per-option mean**; **numeric = pointwise median of the 201-point CDFs**. The no-framework file comments confirm: "The median forecast is taken between NUM_RUNS_PER_QUESTION runs."

### Default LLMs
The template ships with the `llms={...}` block commented out, deferring to `forecasting-tools` defaults, which are chosen by which env keys exist (`ForecastBot._llm_config_defaults`): default forecaster `gpt-4o` (or `claude-3-7-sonnet-latest` / `openrouter/openai/gpt-4o` / `metaculus/gpt-4o` proxy), summarizer/parser `gpt-4o-mini`, researcher preference order **AskNews → Perplexity `sonar-pro` → `openrouter/openai/gpt-4o-search-preview` → Exa SmartSearcher → `gpt-4o-search-preview`**. The commented example shows the intended pattern (`"default": openrouter/openai/gpt-4o @ temp 0.3, "researcher": "asknews/news-summaries", "summarizer"/"parser": gpt-4o-mini`).

### `Metaculus/metac-bot`
https://github.com/Metaculus/metac-bot — README: "This is the old repository used to run the official metac-bots for the Metaculus AI Benchmarking Tournament for Q3 2024 through Q1 2025... The bot is now located in forecasting-tools (see run_bot.py)". Its `main.py` is essentially the no-framework template (AskNews/Perplexity/Exa research, 5 runs median, gpt-4o default).

---

## 2. The `forecasting-tools` Python Library (Ben Wilson / Metaculus)

**Repo:** https://github.com/Metaculus/forecasting-tools (MIT; `pyproject.toml`: author "Benjamin Wilson <ben@metaculus.com>", v0.2.92; CodexVeritas was Wilson's earlier personal org — PyPI: `pip install forecasting-tools`; demo: https://forecasting-tools.streamlit.app/)

### Key abstractions

**`ForecastBot` (ABC)** — `forecasting_tools/forecast_bots/forecast_bot.py` (~1,040 lines). Owns the whole orchestration: `forecast_on_tournament(id)`, `forecast_questions(...)`, per-question concurrency, error propagation, report building, publishing, notepads (per-question state for e.g. alternating models/personas), and a named-LLM registry (`get_llm("default"/"researcher"/"summarizer"/"parser")`). Subclasses implement only `run_research` and `_run_forecast_on_{binary,multiple_choice,numeric,date,conditional}`. README: "This saves you the headache of interacting with the Metaculus API, implementing aggregation of predictions, creating benchmarking interfaces, etc."

**Aggregation** lives on the report data models:
- `BinaryReport.aggregate_predictions` → `statistics.median(predictions)` (`data_models/binary_report.py`)
- `MultipleChoiceReport.aggregate_predictions` → arithmetic **mean per option** (`data_models/multiple_choice_report.py`)
- `NumericReport.aggregate_predictions` → converts each prediction to its 201-point CDF, asserts identical x-axes, takes **`np.median` across CDFs at each x location** (`data_models/numeric_report.py`)
- `BinaryReport` also has benchmarking properties: `expected_baseline_score = 100·(c·(log2 p + 1) + (1−c)·(log2(1−p)+1))` vs community prediction `c`, plus `inversed_expected_log_score` and `deviation_points`.

**`TemplateBot` / `MainBot`** — `TemplateBot` is an alias of the latest seasonal official bot (currently `SummerTemplateBot2026`); the lineage is preserved in `forecast_bots/official_bots/`: `q3_template_bot.py` (Q3 2024: gpt-4o @ temp 0.1 + Perplexity), `q4_template_bot.py`, `q1/q2_template_bot.py`, `template_bot_2025_fall.py`, `..._2026_spring.py`, `..._2026_summer.py`, plus `research_only_bot_2025_fall.py` (model forecasts using only its own internal knowledge/search) and `uniform_probability_bot.py` (baseline). **`MainBot`** ("The verified highest accuracy bot available", `main_bot.py`) = Q2-2025 template prompts + AskNews research + **`openai/gpt-5` with `reasoning_effort="high"`, temp 0.3, 15-min timeout, 5 predictions/question**.

**`GeneralLlm`** — litellm wrapper (one interface to all providers incl. the `metaculus/` donated-credit proxy), retries, cost tracking; `invoke_and_return_verified_type` retries until the output parses to an arbitrary Python/pydantic type; even `invoke_and_unsafely_run_and_return_generated_code` for Fermi-estimate code execution. `structure_output()` (`helpers/structure_output.py`) is the LLM-reparses-LLM-output utility used everywhere, with `num_validation_samples` for consistency checks.

**Research tools** (`agents_and_tools/`):
- **`SmartSearcher`** (`research/smart_searcher.py`) — "Answers a prompt, using search results to inform its response." Pipeline: LLM generates N search queries (optionally with Exa advanced filters like date/domain) → `ExaSearcher` pulls highlights (default 2 searches × 10 sites, top 15 quotes evaluated) → LLM compiles a cited report with `[1]` citations hyperlinked to text-fragment URLs. Default model `gpt-4.1`.
- **`KeyFactorsResearcher`** (`research/key_factors_researcher.py`) — finds background + base-rate key factors for a Metaculus question, deduplicates, then **scores each factor with an LLM scorecard** (`ScoreCard` with graded criteria) and prioritizes the top-k.
- **`BaseRateResearcher`** (`base_rates/base_rate_researcher.py`) — produces base-rate reports by computing numerator/denominator reference classes; relies on **`NicheListResearcher`** (brainstorm all historical instances of a niche reference class → deduplicate → fact-check each item against generated criteria) and **`Estimator`** (Fermi estimation of reference-class sizes).
- **`QuestionDecomposer`** / `QuestionOperationalizer` (`question_generators/`) — decompose a vague topic/question into 3-5 concrete sub-questions (`decompose_into_questions_deep/fast`); used in experimental bots (`experiments/q2t_w_decomposition.py`).
- Misc: `research/computer_use.py`, `find_a_dataset.py`, `data_analyzer.py`, an AI "congress" multi-agent module (`ai_congress_v2`), and a `situation_simulator`.

**`MetaculusClient` / `MetaculusApi`** (`helpers/metaculus_client.py`, `metaculus_api.py`) — typed pydantic wrapper over the Metaculus API: `get_question_by_url/post_id`, `get_all_open_questions_from_tournament`, `get_questions_matching_filter(ApiFilter(...))` (status, type, forecaster counts, date windows; local filtering + random sampling), `get_benchmark_questions()`, posting predictions per type (binary decimal, MC PMF, numeric 201-CDF) and private comments, group-question unpacking (`group_question_mode="unpack_subquestions"`), tournament-ID constants (`CURRENT_AI_COMPETITION_ID`, `CURRENT_MINIBENCH_ID = "minibench"`, Metaculus Cup, Market Pulse), even bot-account creation endpoints.

**Benchmarking** (`cp_benchmarking/`):
- **`Benchmarker`** — "benchmark a list of forecast bots by comparing their predictions to the community prediction on a set of questions." Docstring guidance: "100-200 questions is a decent starting point, but 500+ would be ideal... even with 100 there is ~30% chance of the 'worse bot' winning if there are not large skill differences" (citing https://forum.effectivealtruism.org/posts/DzqSh7akX28JEHf9H/comparing-two-forecasters-in-an-ideal-world). Runs bot batches concurrently, snapshots bot config + git commit hash + code into **`BenchmarkForBot`** records (cost, time, per-report expected baseline scores), saved as JSON/JSONL; `benchmark_displayer.py` provides a Streamlit results browser.
- `scripts/benchmark_forecast_bot.py` is the entry point; the metac-bot-template README links this workflow as "early benchmarking" against community prediction.

**Auto prompt-optimization** (`auto_optimizers/`): `bot_optimizer.py`, `prompt_optimizer.py`, `customizable_bot.py`, `control_prompt.py` — the evolutionary prompt optimizer described in "Iterative AI-driven prompt engineering for AI forecasting" (https://www.metaculus.com/notebooks/38421/automated-prompt-engineering-for-forecasting/): seed with control prompt → Gemini 2.5 Pro + Perplexity brainstorm 25 prompts → score on 112-question train set via expected baseline score → mutate/breed top 5 → 100 prompts total → evaluate top prompts on 230-question test set. **Findings:** big gains for GPT-4.1-nano (−6.35 → +11.72 expected baseline), moderate for GPT-4.1 (15.02 → 21.53), **none for DeepSeek-R1**; optimized prompts transfer within-provider but not across (hurt Claude-Sonnet-4). Resources page adds: "The tentative conclusion is that automated prompt engineering doesn't help o3... prompt engineering has diminishing returns as models get stronger" (https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/). The product of this work is **`GPT41OptimizedBot`** (`official_bots/gpt_4_1_optimized_bot.py`) whose binary prompt is a Gemini-designed three-phase **"Weighted Critique & Sensitivity-Driven Revision"**: Phase 1 Blue Team (optimistic/pessimistic/most-likely scenarios + initial probability), Phase 2 Red Team critique with **Vulnerability Scores (1-5)** and **Plausibility-of-Alternative scores** per critique plus identification of the 2-3 most critical assumptions, Phase 3 Synthesizer revision with explicit sensitivity analysis. Code comment: "the research (and other) fields are intentionally included twice since this is how Gemini designed the prompt when the automated prompt optimization iterations were run."

**Calibration adjustment** (`calibration_adjustment/`): experimental post-hoc recalibration of bot outputs — `constant_shift_adjuster.py`, `logistic_recalibration_adjuster.py`, `decision_tree_adjuster.py`, `k_means_adjuster.py`, `step_adjuster.py`.

**`run_bots.py`** (2,053 lines) — the production harness for Metaculus's in-house fleet (the `metac-*` bots on AIB leaderboards). Each bot = one env-var-keyed `RunBotConfig` with an estimated cost/question. `create_bot()` builds a TemplateBot with `research_reports_per_question=1, predictions_per_research_report=5`, summarizer `openrouter/openai/gpt-4.1-mini`, researcher default `"asknews/news-summaries"`. The fleet (as of the June 2026 tree) spans `gpt-5.5`, `gpt-5.4(-mini/nano)`, `claude-opus-4-6/4-7/4-8`, `claude-sonnet-4-6`, `claude-fable-5`, `gemini-3.1-pro/flash`, `grok-4.3 / 4.20 / 4.20-multi-agent`, `deepseek-v4-pro`, `kimi-k2.6`, `qwen3.x`, `glm-5.1`, `minimax-m3`, `nemotron`, etc., plus research-variant bots: Gemini grounding (`tools=[{"googleSearch":{}}]`), `xai/grok-4` live search, Claude `web_search_20250305` tool (max 10 uses) with 16k thinking budget, `deepseek-r1:online` (Exa), `openai/o4-mini-deep-research`, `perplexity/sonar-deep-research` (~$0.45/call), and AskNews DeepNews (`asknews/deep-research/high-depth/claude-opus-4-6` — comment trail: "Switched to opus 4.6 Feb 23rd. Switched to high depth Feb 16th 2026... sonnet 4.5 in Nov 2025"). Cost anchors in code: gpt-4o ≈ $0.05/question, DeepSeek-R1 ≈ $0.039, sonnet-4 ≈ $0.25, gpt-5-high ≈ $0.38, opus-4.5 ≈ $1.50/question.

---

## 3. Notable Public Bots & Writeups

### 3.1 Panshul42 — Q2 2025 winner ($7,550), open-sourced
**Repo:** https://github.com/Panshul42/Forecasting_Bot_Q2 (AGPLv3) · **Winner announcement:** https://www.metaculus.com/notebooks/39140/winners-of-q2-2025-ai-benchmark-tournament/ · **Full analysis/interview:** https://www.metaculus.com/notebooks/40456/q2-ai-benchmark-results/ (mirrors: https://forum.effectivealtruism.org/posts/F2stjK9wHSy3HPEC9/q2-ai-benchmark-results-pros-maintain-clear-lead, https://www.lesswrong.com/posts/Surnjh8A4WjgtQTkZ/q2-ai-benchmark-results-pros-maintain-clear-lead)

**Architecture** (from code in `Bot/binary.py`, `Bot/prompts.py`, `Bot/llm_calls.py`):
1. **Dual research planning:** o3 is run on two planner prompts — `BINARY_PROMPT_historical` ("write a series of search queries... to find relevant historical context... generate an outside view", routes each query to Google / Google News / Perplexity with per-engine query-style guidance: "phrase the queries in a way optimal for keyword optimized search... Limit to six words" vs. Perplexity "may use a longer query with detailed instructions") and `BINARY_PROMPT_current` (inside-view news; third channel is an AskNews "Assistant").
2. **Retrieval:** `search.py` + `FastContentExtractor.py` — Serper (Google/Google News), AskNews, Perplexity sonar-reasoning-pro, BrightData scraping; multi-strategy HTML extraction (BeautifulSoup site-specific selectors, Trafilatura, Readability fallbacks).
3. **Stage-1 outside view** (`BINARY_PROMPT_1`): per-source quality analysis ("you must be able to discern factual information from opinions... strongly consider only opinions originating from identifiable experts or entities"), reference-class analysis, timeframe analysis, calibration ("Small differences in probabilities can be significant: 90% is a 9:1 odds and 99% is a 99:1 odds").
4. **Stage-2 inside view** (`BINARY_PROMPT_2`): an explicit **evidence weighing system** (Strong = "Multiple independent, reliable and identifiable sources confirming same direction / Direct causal mechanisms / Historical patterns with strong predictive power / Structural factors"; Moderate; Weak = "Anecdotal evidence; Speculative connections...") plus a six-item **FORECASTING CHECKLIST** run in-reasoning: paraphrase resolution criteria ("Bait-and-switch errors, while commonplace, are costly"), re-anchor to outside-view base rate ("Outside first, usually"), consistency line ("{prediction} out of 100 times, {resolution criteria} happens"), evidence validity check, "Blind-spot statement: Name the one scenario most likely to make your forecast look silly in hindsight", status-quo nudge.
5. **Ensemble of 5 forecasters:** 2× Claude Sonnet (3.7 → 4) + 2× o4-mini + 1× o3, with **weighted average, weights `[1, 1, 1, 2, 2]`** (o3 double-weighted; the inline comment jokes "forecaster 5 (o3) has truple weight"). Numeric: weighted average of full 201-point CDFs (`numeric.py`); also a **Monte Carlo prompt variant** that makes the LLM write NumPy simulation code (≥100k iterations, 2-4 variables, "Be careful not to add too many variables and constraints, as this might skew your final probability to 0") which is executed to print `Probability: XX.X%`.
6. **Benchmarking:** `benchmark.py` — normalized peer scoring with Monte Carlo simulated forecaster populations + community-prediction comparison; in practice ~35 questions per benchmark run + heavy manual review.

**Performance:** sum of spot peer score 5,899 vs. house bot metac-o3+asknews at 5,131 — *not* statistically separable ("error bars... overlap considerably"). Metaculus's takeaway: "the most important factor for accurate forecasting is the base model used, with additional prompting and infrastructure providing marginal gains." Panshul's advice (Bot Q in survey): "Measure performance not just by accuracy metrics, but also by how well the LLM reasons... some reasoning issues can be fixed with prompting, but systematic ones need better models." His **Fall 2025 plan** (https://www.metaculus.com/notebooks/43497/what-are-other-bots-doing/): "Agentic search (with live search tooling)... parallel one-shot forecasting, ensembling... This sounds simpler than my Q2 bot... because much of the complexity needed for google search/scraping can be easily and reliably abstracted to native LLM provider web search," with models moved to GPT-5/GPT-5-mini.

### 3.2 pgodzinai — Phil Godzin (BestWorld) — Q4 2024 winner ($9,658), Q2 2025 2nd
**Sources:** https://www.lesswrong.com/posts/P8YwCvHoF2FHQoHjF/metaculus-q4-ai-benchmarking-bots-are-closing-the-gap (full Q4 analysis: https://www.metaculus.com/notebooks/35291/q4-ai-benchmarking-results/), Q4 winners post https://www.metaculus.com/notebooks/31370/ai-forecasting-benchmark-q4-winners-q1-warmups/
- **Research:** both Perplexity (`llama-3.1-sonar-huge-128k-online`) **and** AskNews — "fetched the maximum number of latest news and historical articles from AskNews and then used an LLM to summarize only the sources relevant to the question." Specialized prompt guidance per question genre (competitions, market prices, disease).
- **Ensemble:** 3× GPT-4o + 5× Claude-3.5-Sonnet-20241022; **drop the two most extreme values, average the remaining six**. Notably did *not* use o1-preview.
- **Prompting:** "superforecaster" persona, explanation of scoring mechanics, forecasting principles, and an explicit prior: "historically it's been overconfident and the base rate for positive resolutions on Metaculus is 35%."
- **Key innovation (Q4):** "asking an LLM to group related questions together and predict them in batches to maintain internal consistency" — fixed incoherent multi-question sets (probabilities summing >150%).
- **Performance:** Q4 weighted avg peer score 13.2 [8.3, 18.1], statistically above everyone (p < .0000005); vs Pros head-to-head −8.9 (p = 0.079, n.s.). Dev time 15-40h. Q2 2025: coverage-adjusted 4,837 with 16-40h invested.

### 3.3 Metaculus house bots as de-facto winners (metac-o1, metac-o3)
- **Q1 2025:** metac-o1 (o1 + AskNews + the ~30-line template prompt, 5 runs) **won the whole tournament** and placed 25/617 in the human Quarterly Cup. "The best bot was a single-prompt bot using OpenAI's o1... Most competitors did not use o1, and opted for cheaper models." AskNews > Exa/Perplexity (not stat. sig.). https://www.metaculus.com/notebooks/38673/q1-ai-benchmarking-results/ (mirror: https://www.lesswrong.com/posts/rDy5z8ZEtMrEGnfBd/q1-ai-benchmark-results-pro-forecasters-crush-bots)
- **Q2 2025:** metac-o3+asknews 2nd overall, won the multiple-choice-only leaderboard. https://www.metaculus.com/notebooks/40456/q2-ai-benchmark-results/
- **Q3 2024:** mf-bot-1 (single-shot GPT-4o prompt) placed 4th; "We were also surprised that a single shot prompt like mf-bot-1 finished near the top compared to the many bots that used multiple agents and prompt chaining." https://www.metaculus.com/notebooks/28784/aibq3results/

### 3.4 lightningrod (Lightning Rod Labs)
- Topped the **numeric-only leaderboard in Q2 2025** (https://www.metaculus.com/notebooks/39140/...); "climbed Metaculus and market leaderboards between 2024 and early 2026, placing as high as fourth out of more than 500 entrants" (https://www.lightningrod.ai/).
- Approach: **"Future as Label" outcome-supervised fine-tuning** — "Models are trained on historical news by making predictions using only information available at prediction time. Those predictions are then scored based on what actually happened after." Qwen3-32B fine-tuned this way: Brier −27%, calibration error halved, beats Qwen3-235B on Metaculus benchmark despite 7× smaller. Paper: "Future-as-Label: Scalable Supervision from Real-World Outcomes" https://arxiv.org/abs/2601.06336; blog: https://blog.lightningrod.ai/p/using-the-future-to-train-prediction-models. Their pastcasting-dataset SDK is the one integrated in the official template (`integrations/main_lightningrod_eval.py`).
- Survey quote attributable to them (Fall advice post): "Small models can rival or beat large flagship models if they are fine-tuned via RL with the future as the label."

### 3.5 ManticAI / Mantic
- Q4 2024 4th ($3,410), Q1 2025 3rd — "o3-mini with key data sources (news, Wikipedia, economics data, external forecasts)", "base rates, status quo bias", "agentic tool calling" with incremental improvements (Q1 results post).
- 2026 (with Thinking Machines): two-phase research→prediction architecture; prediction phase has the LLM build a **mixture distribution** ("the LLM selects: the number of components in the mixture, their parameters, and their respective weights"); **RL fine-tuned gpt-oss-120b on ~10,000 binary questions** with "a standard policy gradient algorithm with GRPO-style advantage normalisation" using **Brier score as reward** ("more stable than the log score"); lifted its score 38.6 → 45.8 ("marginally above" Gemini 3 Pro), and the fine-tuned model was "the least replaceable" ensemble member after Grok 4 (40-56% ensemble weight). https://thinkingmachines.ai/news/training-llms-to-predict-world-events/

### 3.6 nostreambot — best open-source bot, Fall 2025 (9th); 7/93 Spring 2026
**Repo:** https://github.com/No-Stream/metaculus-bot · **Writeup:** "on writing a bitter-pilled metaculus bot" https://nostream.substack.com/p/on-writing-a-bitter-pilled-metaculus
- Thesis: "the biggest gains came less from clever prompting and more from boring engineering: model ensembling, continuous-question handling, reliability, and better evaluation."
- Multi-model ensemble with **median** aggregation ("the single largest contributor"); AskNews grounding + Grok 4.1 fast / Gemini 3 Flash grounding + FRED/Yahoo Finance for financial questions; numeric questions improved with **PCHIP smoothing** of the CDF (credited to Panshul42's approach) and added extreme quantiles (p2.5/p97.5) for fatter tails — "one place the template bot could be improved." Robustness lesson: missing 10 questions ≈ −150 points; a units bug cost ~80 points.

### 3.7 Jeremy Lichtman / BestWorld "Multi-AI Oracle"
- Jeremy Lichtman is CTO of BestWorld (Carolyn Meinel's org; same group as Phil Godzin/pgodzinai — "The winning bot, pgodzinai, created by BestWorld's Phillip Godzin"). His own bot, the **Multi-AI Oracle**, combines "several large language models including a news scraping and parsing capability" with daily public update posts; "Jeremy's Multi-AI Oracle was pretty good at forecasting the election." Bio: https://bestworld.net/jeremy-lichtman; archive of forecasts: https://bestworld.net/all-our-past-genai-forecasts; update example: https://bestworld.social/updates-from-bestworlds-ai-botmaster-jeremy-lichtman-of-dec-26-2024 (site intermittently unreachable during this research). Q4 survey context: top-5 bots ran primarily gpt-4o + claude-3.5-sonnet, 76% of winners aggregated repeated calls via median/mean (https://www.lesswrong.com/posts/P8YwCvHoF2FHQoHjF/...).

### 3.8 MWG, histerio, GreeneiBot2, twsummerbot, TomL2bot, CumulativeBot, smingers
- **MWG** — 2nd in Q4 2024 ($4,477) and 2nd in Q3 2024; peer score 8.8 in Q4, the only bot whose CI overlapped pgodzinai's. No public writeup found. (https://www.metaculus.com/notebooks/31370/..., 35291)
- **histerio** — 3rd Q3 2024, 5th Q4 2024 (7.9 peer). Same source.
- **GreeneiBot2** — 3rd Q4 2024 ($3,930), **top of Spring 2026 leaderboard**. Self-description (https://www.metaculus.com/notebooks/43497/what-are-other-bots-doing/): "async Python bot... parallel research before forecasting: AskNews (news context and deep research), Perplexity (sonar-reasoning-pro and sonar-deep-research), GPT-5 search via OpenRouter, and link assistants that **screenshot resolution links** and process them with GPT-5 or BrowserUse... structured CDFs (percentiles and spikes) for numeric/discrete, validation against platform rules, and multi-run aggregation."
- **twsummerbot** — Q1 2025 8th using only gpt-4o (vs metac-gpt-4o at 44th): "2-4 weeks full-time," 50-100 LLM calls/question, agentic tool calling — the Q1 counterexample showing scaffolding can rescue a weaker model. (Q1 results post)
- **TomL2bot** — Q2 2025 8th with Gemini-2.5-pro, score 3,288 vs metac-gemini-2.5-pro+asknews 1,535. Ex-Metaculus employee (wrote the original metac prompts); strategy = heavy manual output review, **prediction capping**, aggregation, final-prompt engineering; tried similar-question lookup, "found this did not help." (Q2 analysis)
- **CumulativeBot** — Q2 2025 3rd ($3,270, 41-80h, hobbyist, no AskNews). Spring 2026 description: "My strategy has been scale: I have a lot of research methods... a system that uses generated perspectives for direct prediction + monte carlo sims, a system that uses tailored prompts based on the question..., a deprecated agentic pipeline (expensive and underperformed; don't bother), and a system that takes base-rate-only research, generates a base rate, then updates it with contemporary information... I also try to veto obviously bad predictions with a set of sanity checks." Built own deep researcher on atomic-agents: "pretty much every hallucination could be tracked back to it." (43497 notebook)
- **smingers (Jordan Berman, GJ Inc. superforecaster)** — "I'm a Superforecaster. Here's what happened when I built the bot that might replace me" https://thonks.substack.com/p/im-a-superforecaster-heres-what-happened. 3-stage pipeline (research via AskNews/Serper/Wikipedia/yFinance/FRED/Google Trends → **5 outside-view forecasters across 3 models** → 5 inside-view instances that adjust the outside view), plus a **Claude Opus "Supervisor" agent invoked only on high ensemble divergence** (~1 in 6 forecasts, average adjustment 2.3pp); 70+ LLM calls per forecast; 15th place, ~$1,000 prize, beat community on ~75% of resolved questions. Lessons: "Tools are more powerful than prompts"; "No prompting technique or architecture could overcome a fundamental ignorance about the world."
- **adjacentwire** — forked the template, changed *one thing*: injected prediction-market odds from their Adjacent API into question background; placed #11 in Q2 despite a late start. (43497 notebook)
- Other Fall/Spring open-source bots (from https://www.metaculus.com/notebooks/43357/bot-advice-fall-2025/ and the resources page): geemus+bot (14th) https://github.com/geemus/metaculus-ai-forecasting, OliveTestBot (16th), BottyMcBotFace (26th), alexwcohen_bot (27th), plus a longer list (vigjibot, Lyfbot, hb_bot, 5cast, caliperbot, joy.void.joy-bot, etc.) at https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/#open-source-bots.

---

## 4. Cross-Cutting Findings (what wins, what doesn't)

**Tournament-over-tournament arc:**
- **Q3 2024 – Q2 2025: "model > scaffolding."** Repeated finding: "the most important factor for good forecasting is the base model, and additional prompting and infrastructure on top of this provide marginal gains" (Q1 + Q2 analyses). Single-prompt house bots (gpt-4o → o1-preview → o1 → o3) finished 4th, 6th, 1st, 2nd across four quarters.
- **Fall 2025: scaffolding now matters.** "Model convergence is (almost) complete" (34/39 respondents on frontier models) → "The gap between good and bad scaffolding around a frontier model is now comparable in magnitude to the gap between frontier and non-frontier models themselves" (within the GPT-5 family alone, ~27 peer points/question spread between best and worst scaffolds). https://www.metaculus.com/notebooks/43337/fall-2025-futureeval-survey/
- Pros still beat bots every quarter measured: head-to-head −8.9 (Q4'24, n.s.) → −17.7 (Q1'25) → −20.03 (Q2'25, [−28.63, −11.41]); pros' edge is **discrimination, not calibration** (44pp vs 26pp discrimination gap, Q4 analysis). Bots are worst on multiple choice (Q2: binary −14.8, numeric −23.2, MC −32.9).

**Quantified best practices:**
| Practice | Effect (Q2 2025, coverage-adjusted pts) | Source |
|---|---|---|
| Custom question creation/resolution testing | +2,216 [+912, +3,519] | Q2 analysis |
| Aggregating multiple forecasts (median/mean) | +1,799 [+1,017, +2,582] | Q2 analysis |
| Significant manual review of outputs | +1,041 [−223, +2,305] | Q2 analysis |
| AskNews usage | ~no effect | Q2 analysis |
| # distinct research sources (winners 1.75 vs 1.00) | r = 0.42, p ≈ 0.006 — strongest Fall-2025 predictor | Fall survey |
| Testing vs community median pre-submit | 41% of winners vs 0% of non-winners (p = .017) | Fall survey |
| LLM calls/question (median 28 winners vs 7) | p = .022 | Fall survey |
| Capping predictions at max/min | strongest *within-winners* differentiator, r = +0.48 | Fall survey |
| Explicit base-rate calculation | r = +0.38 within winners | Fall survey |
| Dev time | r ≈ 0.08–0.20 (negligible; Q1 even negative) | Q1/Q2/Fall |

**Costs:** Fall 2025 median winner ≈ **$0.90/question** (top-15 ≈ $1.40, bottom-half winners ≈ $0.50; no non-winner spent >$1); a season costs "$50 to $1k+" total (resources page). House-bot anchors: gpt-4o $0.05/q, DeepSeek-R1 $0.039/q, sonnet-4 $0.25/q, gpt-5-high $0.38/q, opus-class $1.50/q, sonar-deep-research ~$0.45/call (`run_bots.py`).

**Recurring engineering gotchas** (Fall advice post, https://www.metaculus.com/notebooks/43357/bot-advice-fall-2025/): bots predicting 99% because they think an unresolved question already resolved ("a good way to get a massive negative score"); naive template numeric handling (one maker succeeded by eliciting **mean + std-dev** instead of percentiles; `Percentile values must be in strictly increasing order` failures hit ~30% of numeric/MC runs for one Q2 bot); clipping/extremization underused ("The community continues to underrate the benefit of clipping their predictions"); leaky backtests → use MiniBench for iteration; one bug on a rare question format = "roughly $500 mistake."

**Consensus recipe (Metaculus's own "playbook for the next round", Fall 2025):** frontier reasoning model; 2-3 research sources (winning stacks: "AskNews + Perplexity + a custom scraper, or AskNews + Exa + OpenAI web search"); agentic/iterative search rather than one-shot ("generate queries → search → identify gaps → search again"); 10-30 LLM calls/question split research/forecast/critique; aggregate 3-10 runs (median); cap at ~[2%, 98%]; explicit base rates; check similar resolved questions/markets as priors; manual review loop; native provider web-search tools > custom pipelines ("agents are trained to use them effectively during reasoning using RL").

**Key local file paths for follow-up:** `/tmp/metac-research/metac-bot-template/main.py`, `/tmp/metac-research/forecasting-tools/forecasting_tools/{forecast_bots/forecast_bot.py, data_models/numeric_report.py, agents_and_tools/research/smart_searcher.py, cp_benchmarking/benchmarker.py}`, `/tmp/metac-research/forecasting-tools/run_bots.py`, `/tmp/metac-research/Forecasting_Bot_Q2/Bot/{prompts.py, binary.py}`, and extracted notebooks `/tmp/metac-research/nb{38673,40456,43337,43357,43497,38928,38421,35291,28784,31370,39140}.md`.
