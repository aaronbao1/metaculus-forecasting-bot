# 07 — Incorporation Research: Self-Improvement, Submodels, TimesFM, GABRIEL, parse.bot

*Compiled 2026-06-11. Sources: 105-agent adversarially-verified deep-research workflow (22 claims confirmed 3-0/2-1, 3 refuted 0-3) + 4 dedicated research agents + full repo architecture mapping. Companion to research-notes/01–06 and FORECASTING-BOT-RESEARCH.md.*

---

## 0. Executive summary and ranked incorporation table

Every item on the wishlist is incorporable; the repo's existing seams (the `ForecastSystem` protocol, the research-provider dispatch, `llm_cache`, the experiments YAML + paired-stats gate) mean almost nothing needs architectural change. The binding constraint is **not engineering or dollars — it is statistical power**: with a few-thousand-question corpus, the detectable-effect floor is ~0.005 Brier for tightly-paired variants (~780 questions at 80% power, σ_d≈0.05) and 0.02+ for loosely-coupled ones (7k–28k questions). Every self-improvement loop below must therefore *generate candidates cheaply and promote rarely*, with the existing paired-bootstrap gate on temporally held-out questions as the only promotion path.

| Rank | What | Value (vs. documented gaps) | Eng. time | $ cost | Plugs into |
|---|---|---|---|---|---|
| 1 | Calibration flywheel on backtest artifacts (already designed as Retrocast ext. 3) | Platt fitting worth +0.006–0.016 Brier (research-notes/06); unblocks the ≥100-resolved-forecast requirement *now* | 1–2 days | ~$0 (reuses artifacts) | `scripts/fit_calibration.py`, `allow_calibration` guard |
| 2 | Open-source submodels (statsforecast, Chronos-2, TimesFM 2.5 zero-shot) as `ForecastSystem`s + numeric CDF anchor | Attacks the worst documented weakness: numeric −23.2 / MC −32.9 vs binary −14.8 | 3–5 days | ~$0 (CPU inference) | `backtest/systems/` protocol; `ctx.store.value_at()`; ForecastBench dataset questions already in corpus |
| 3 | GABRIEL-style evidence rating + corpus annotation + Clio-style failure clustering | Improves the #1 winner predictor (research source quality, r=0.42); failure taxonomy at scale | 3–5 days | $10–50 per corpus pass (nano models) | `bot/research.py` relevance filter; post-mortems over `artifacts.jsonl` |
| 4 | GEPA (DSPy) prompt optimization, binary first, on cached research | Modest expected gains (frontier-model caveat); near-free rollouts via cache | 2–4 days | $20–200/run | `bot/prompts.py` constants; `from_research_text()` stage entry; experiments gate |
| 5 | ACE-style self-improving playbook ("skills") + resolution post-mortem memory | Cheapest closed loop; low Goodhart risk (human-readable deltas) | 3–5 days | $5–20/cycle | New `skills/` dir; injected into prompts; gated by experiments |
| 6 | Evolutionary code search (ShinkaEvolve/OpenEvolve) on **cached-artifact** components only | Bespoke aggregation/calibration/feature discovery | 1 week | $1–120/run (≈$0 if fitness uses cache) | `bot/aggregation.py`, `bot/calibration.py` genomes; backtest = fitness fn |
| 7 | TimesFM fine-tuned collection (LoRA adapters per domain) + Chronos-Bolt/TTM | Upgrade path for #2 once zero-shot is plumbed | 1 week initial, then maintenance | <$5/fine-tune run; <$50/mo for 5–10 adapters | Same `ForecastSystem` seam; HF PEFT path official since Apr 2026 |
| 8 | parse.bot for live ingestion | Marginal: live-only, no archive ⇒ useless for Retrocast; OK for live bot on API-less sites | 1–2 days | $0 trial (200 credits); $30/mo Hobby | New provider in `bot/research.py` (live only) |
| 9 | RLVR fine-tuned open-weights ensemble member (Turtel-style, via Tinker/Together) | Decorrelation value (Mantic earned 40–56% ensemble weight) | 2–4 weeks | $300–3,000 | New `EnsembleMember`; phase-2+ per existing doctrine |
| 10 | Persona simulation / silicon sampling | **Hold.** Headline validation numbers (91% / 85%) refuted 0-3 in adversarial verification | — | — | Experimental variant only, never default |

Realistic program through rank 6: **~6–8 part-time engineering weeks, $300–800 of one-off API/compute, ~$50–100/mo ongoing.**

---

## 1. The foundation: backtest-as-fitness-function (item 6 — "train/tweak by backtesting, start with binary")

This is the keystone — everything else is a candidate generator that feeds it.

**What the repo already has** (no new work): YAML variants with config patches, paired bootstrap + paired t in `backtest/scoring/stats.py`, manifest-resumable runs, `llm_cache` giving $0 replays, contamination strata, profiles with hard cost caps ($5/$150/$500).

**What to add:**

1. **Calibration flywheel (rank 1).** Run `current-bot-clean.yaml` (120 binary q, ~$30), then fit Platt on the resulting clean artifacts instead of waiting months for live resolutions. The `MIN_FIT_N=100` requirement is satisfiable immediately. Keep the `allow_calibration=False` leakage guard for any backtest that *evaluates* the calibrated config — fit on one temporal window, evaluate on a later one.
2. **Promotion gate codification.** A small `scripts/promote.py`: a variant is promoted to live config only if (a) paired stats significant on a **temporally held-out window** (train on questions closing before date D, validate after D), (b) clean stratum only, (c) improvement ≥ the noise floor for that comparison type. This single script is what makes every automated loop below safe.
3. **Stage-level replay for cheap optimization.** `from_research_text()` and `from_member_forecasts()` already exist as stage entry points. Optimizing aggregation/calibration against cached member outputs costs **$0 per fitness call**; optimizing prompts against cached research costs only the member LLM calls (~$0.18/q at 9 × $0.02).

**Statistical power (cross-cutting; from ForecastBench/Wilks/Metaculus variance data):**

| Comparison type | σ of per-question Brier diff | n for δ=0.005 @ 80% power |
|---|---|---|
| Tightly paired (same research+members, one component changed) | ~0.05 | ~780 |
| Loosely paired (different prompts, shared questions) | ~0.10–0.15 | 3k–7k |
| Different models | ~0.15–0.30 | 7k–28k |

ForecastBench treats a 0.026 Brier gap as significant at N=1,000; the current superforecaster-vs-best-LLM gap is 0.017. Implication: **the corpus build-out (Metaculus Data Needs Form, ForecastBench's 34 question sets, Halawi's 5.5k, market crosswalks) is itself the highest-leverage "self-improvement" investment** — it raises the ceiling on every loop.

Cost/time: 2–4 days; ~$30–150 per standard backtest; flywheel itself ~$0.

---

## 2. Self-improving skills (item: "automate creation of self-improving skills")

**What survived scrutiny in the literature** (self-evolving-agents survey [arXiv:2507.21046](https://arxiv.org/abs/2507.21046) taxonomizes by component: models / memory / tools / architecture):

- **ACE — Agentic Context Engineering** ([arXiv:2510.04618](https://arxiv.org/abs/2510.04618)): contexts as evolving *playbooks* updated by Generator/Reflector/Curator with **delta updates** (avoids context collapse from full rewrites); +10.6% agent benchmarks, +8.6% finance, far cheaper than re-optimization. This is the right template for forecasting skills.
- **Voyager/SkillWeaver** ([2305.16291](https://arxiv.org/abs/2305.16291), [2504.07079](https://arxiv.org/abs/2504.07079)): skills as validated code + embedding retrieval; SkillWeaver's skills transferred (+up to 54% for weak agents). Anthropic's SKILL.md format ([docs](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)) is free packaging.
- **ADAS/AFlow architecture search: skip.** A critical review documents contaminated benchmarks, ~9% nondeterminism, reward hacking; Google's MASS follow-up ([2502.02533](https://arxiv.org/html/2502.02533v1)) found **prompts matter more than topology**. With our noisy metric this is Goodhart bait.

**Concrete design (rank 5):** after each resolution batch (live or backtest), an automated post-mortem agent reviews the worst Brier contributors → proposes *delta* additions to a `skills/forecasting-playbook.md` (e.g., "for sanctions questions, check OFAC press page; base rate of X is ~Y") → deltas accumulate in a candidate playbook → playbook variant runs through the experiments gate before promotion. Pair with **retrieval of similar past questions + their outcomes + post-mortems** at forecast time (the corpus store already holds everything needed; add an embedding index over resolved questions). No published forecasting system does this yet — genuinely open niche.

Cost/time: 3–5 days; $5–20 per nightly/weekly cycle (cheap model for reflection); zero risk to live bot because promotion is gated.

---

## 3. Submodels: pull from open source + create our own (items: submodels, TimesFM collection, auto-research for submodels)

### 3a. Open source first (rank 2 — do this before any fine-tuning)

The `ForecastSystem` protocol is the seam: a submodel is just another system returning a `ForecastArtifact`, with `ctx.store.value_at(qid, "market_price"/"cp", ctx.as_of)` and series blobs as features. **Fitness calls are free** (no LLM).

- **statsforecast (AutoARIMA/AutoETS/Theta) + LightGBM**: M-competition evidence says these still win on short, low-frequency series — exactly FRED-style monthly/quarterly indicators. Hours to integrate.
- **Chronos-2 / Chronos-Bolt** (Amazon, Apache-2.0): best pure pretrained model on GIFT-Eval (rank ~22 vs TimesFM-2.5 at ~29 as of 2026-06-11, [tsfm.ai](https://tsfm.ai/benchmarks/gift-eval)); quantile-grid outputs; CPU inference; easiest fine-tuning in class (`fine_tune=True` in AutoGluon).
- **TimesFM 2.5** (Google, Apache-2.0, 200M params, 16k context): mean + q10–q90 deciles, optional 30M continuous-quantile head, `fix_quantile_crossing=True`. Official **HF Transformers + PEFT/LoRA fine-tuning example since Apr 2026** (`timesfm-forecasting/examples/finetuning/finetune_lora.py`; LoRA r=4, ~0.6% trainable params).
- **IBM Granite TTM** (1–5M params, Apache-2.0): per-domain micro-fine-tunes "in minutes" on one GPU — the cheapest "many small specialists" play.
- **Avoid as collection members:** Moirai-2 (CC-BY-NC weights — a problem if the bot wins money), TimeGPT (closed API, can't own weights), TiRex (restrictive license).

**Where the value is:** ForecastBench-style *dataset questions* (FRED/ACLED/yfinance/Wikipedia series — 34 question sets already in `data/backtest/datasets/`) and Metaculus numeric questions, where the bot's documented deficit is largest. Pipeline: TSFM extrapolates the series quantitatively → quantiles anchor the CDF body → fit parametric tails below q10/above q90 (TimesFM's deciles alone can't fill a 201-point Metaculus CDF) → LLM adjusts for resolution-criteria quirks and regime changes. This also slots directly into **Retrocast extension 4 (auto-generated dataset questions)** — already flagged in the plans as "biggest post-MVP win." Published hybrid precedent: "From News to Forecast" ([NeurIPS 2024](https://arxiv.org/abs/2409.17515)); **no published work applies TSFMs to prediction-market or geopolitical questions — open niche.**

**Caveat (verified finance literature, [arXiv:2511.18578](https://arxiv.org/abs/2511.18578)):** TSFMs perform poorly zero-shot *and* fine-tuned on financial/martingale-like series. Expect ~no alpha over "last price" on market-price series; point submodels at economic indicators, counts, and trends instead.

Cost/time: 3–5 days for statsforecast + one TSFM as systems with CDF anchoring; $0 (CPU).

### 3b. Fine-tuned TimesFM collection (rank 7 — only after 3a shows signal)

- Hardware: 200M params LoRA-fine-tunes on a single T4/4090/A10G; no A100 needed. RunPod A100 $1.19–1.39/hr if wanted; Colab Pro $9.99/mo ≈ 57 T4-hours.
- Cost: **low single-digit dollars per LoRA run**; a maintained collection of 5–10 domain adapters (econ indicators, conflict counts, epidemiology, energy, tech-adoption) is **<$50/month** including periodic refresh.
- Maintenance loop: monthly — re-pull fresh series → re-run LoRA per domain → backtest each adapter vs its zero-shot parent on held-out dataset questions → promote only gate-passing adapters. This *is* the "auto research for submodels" loop, and it's fully automatable because fitness is free and leakage-free (series are dated).
- Evidence on gains: fine-tuning usually helps but is domain-dependent (Lag-Llama mid-pack→best; process-model forecasting "gains often small"); zero-shot is a strong default — hence 3a first.

### 3c. Automated submodel/algorithm discovery (rank 6)

- **OpenEvolve** (Apache-2.0 AlphaEvolve reimplementation, `pip install openevolve`, works with the existing OpenRouter endpoint): verified per-iteration costs $0.01–0.60 depending on model ⇒ **$1–120 per 100–200-iteration run**. ⚠️ Its self-reported benchmark wins (HotpotQA +23%, kernel speedups) were **refuted 0-3** — treat as usable infrastructure, not proven-effective.
- **ShinkaEvolve** (Sakana, Apache-2.0): the sample-efficient variant — SOTA results in ~150 evaluations via novelty rejection-sampling; better fit when fitness costs money.
- **CAAFE / LLM-FE** ([OpenReview](https://openreview.net/forum?id=9WSxQZ9mG7), [2503.14434](https://arxiv.org/html/2503.14434v1)): LLM-written semantic features evolved against eval — the template for feature extractors over the corpus (e.g., market-microstructure features, news-velocity features).
- **AutoGluon 1.5** for the boring 90%: near-SOTA tabular/TS ensembles in 3 lines, includes Chronos-2.

**Rule: only evolve components whose fitness is computable from cached artifacts** — aggregation functions, calibration maps, CDF-tail shapes, feature extractors, hazard-rate schedules. Never the full LLM pipeline (each fitness call would cost dollars and the eval noise swamps the signal). Cap eval budgets (150–500), always confirm winners on a temporally held-out window through the promotion gate.

Cost/time: ~1 week for the evaluator harness + first runs; $1–120/run, ≈$0 if fitness is fully cached.

---

## 4. GABRIEL, OpenAI "Scaling social science," and the Anthropic analogs (rank 3)

**Resolved:** "Gabriel" = **GABRIEL** (Generalized Attribute Based Ratings Information Extraction Library) — official OpenAI Economic Research toolkit, announced 2026-02-13 in ["Scaling social science research"](https://openai.com/index/scaling-social-science-research/), co-authored with Harvard (Asirvatham, Mokski, **Shleifer**), companion paper NBER WP 34834 "GPT as a Measurement Tool." Apache-2.0, `pip install openai-gabriel` (v1.1.8, May 2026), [github.com/openai/GABRIEL](https://github.com/openai/GABRIEL). All verified 3-0. Caveat: OpenAI-API-only client (route through OpenRouter-compatible base URL or keep a small OpenAI key); validation is self-assessed (1,000+ human-annotated tasks, "generally indistinguishable from human evaluators").

**Primitives → forecasting mappings:**

| GABRIEL primitive | Use here |
|---|---|
| `rate` (0–100 attributes, parallelized) | Article relevance/reliability/recency-sensitivity scoring in the retrieval pipeline — replaces/augments the Halawi 1–6 relevance filter |
| `rank` (pairwise ELO → z-scores) | Ordering evidence snippets or candidate rationales (pairwise beats absolute scoring for reliability) |
| `classify` | Evidence-type tagging (primary source / market signal / speculation) |
| `extract` | Structured fact extraction (dates, counts, named commitments) from news |
| `discover` | **Contrast resolved-YES vs resolved-NO question corpora in the backtest set to find discriminating features** — automated hypothesis generation for the playbook |

**Verified economics:** ~17,500× cheaper than $15/hr human annotation with nano-class models; 100k full-text docs = $43–1,083. A full pass over the existing news corpus on several attributes ≈ **tens of dollars**. Route all calls through `llm_cache` for reproducibility.

**Methodology worth stealing regardless of the library:**
1. **Contamination test design** (paper §4.3.1): exploit staggered training cutoffs across models — no significant accuracy difference pre/post-cutoff ⇒ measurement not contaminated. Directly reusable to validate Retrocast's leakage controls, complementing `model_releases.py`.
2. **Prompt-robustness sweeps**: vary phrasings, confirm score stability — apply to our prompts before trusting any A/B.
3. **Consensus columns**: N independent samples per judgment with agreement stats — cheap reliability signal for any LLM-rater step.

**Anthropic analogs (method-level transfers, no library):**
- **Clio** ([arXiv:2412.13678](https://arxiv.org/abs/2412.13678); open reimplementation [OpenClio](https://github.com/Phylliida/OpenClio)): facet extraction → embed → cluster → LLM-titled hierarchy. Use for: (a) clustering retrieved news per question, (b) **failure-mode taxonomy over losing forecasts/rationales** in `artifacts.jsonl` — feeds the playbook loop. ~2–3 days, one cheap-model call per item.
- **Economic Index "primitives" pattern** ([anthropic.com](https://www.anthropic.com/research/economic-index-primitives)): fixed rubric of questions applied uniformly by an LLM — structurally identical to what the relevance filter already does; extend to a standard per-question annotation rubric (question class, data-richness, base-rate availability) that drives **triage routing**. Anthropic validated for *directional* accuracy only.

**Adjacent social-science methods — calibrated take:**
- **Anchoring on crowd/market medians** (Schoenegger et al., Science Advances): giving LLMs the human median improved accuracy 17–28% — already aligned with repo doctrine (CP as anchor input, not target).
- **SCM-conditioned forecasting** (Manning/Horton, [2404.11794](https://arxiv.org/abs/2404.11794)): LLMs get causal *direction* right but *magnitude* wrong unless conditioned on an explicit causal model → a "draft causal graph → estimate edges → derive probability" prompt stage is a cheap experiment variant.
- **Persona simulation / silicon sampling: HOLD.** The headline numbers (Hewitt 91% treatment-effect variance; Park 85% individual replication) were **refuted 0-3** in adversarial verification (claims traced to a secondary source, [2504.02234](https://arxiv.org/abs/2504.02234)). The technique may still have value for poll/referendum questions, but build it only as a gated experimental variant, never as a default signal.

Cost/time: 3–5 days; $10–50 per corpus pass; failure clustering $5–20.

---

## 5. parse.bot and ingestion (rank 8)

**Verified facts (3-0 via live `parse.bot/api/products` probe):** YC F25, 2-person team, launched Aug 2025. Reverse-engineers sites' internal XHR/JSON calls into hosted typed REST endpoints (no headless browser); ~87–241ms calls; marketplace of community APIs; hosted MCP server. Pricing: **Free $0/200 credits/5 rpm; Hobby $30/mo/1,000 credits/20 rpm; Developer $100/mo/5,000 credits/250 rpm**; new API build = 75 credits, revision = 50. No SLA anywhere on the site; breakage repair = manual marketplace merge; HN reception skeptical (fragility, selector churn); documented `blocked` failures on bot-protected sites.

**Decisive limitation: live-only.** Zero archive/point-in-time capability ⇒ **unusable for Retrocast** (as-of-T discipline impossible). Any parse.bot-derived signal is also un-backtestable until the snapshot accumulator has archived months of its output — which weakens the case further.

**For policy questions specifically, mostly unnecessary:** Federal Register, congress.gov, CourtListener/RECAP have free official APIs with **natively dated** records (backtest-compatible!); central banks have RSS; FRED/ALFRED gives vintage data. **Exa** is the only commercial option with native dated search (`start/end_published_date`, $1/1k pages) — and `exa_historical` is already a planned Retrocast provider.

**Verdict:** run the $0 free-tier trial (200 credits) on 2–3 specific API-less sites you actually want (e.g., a niche government stats page); wire as a live-only research provider (~1–2 days) behind the never-raise contract; record everything via the snapshot accumulator so it *becomes* backtestable later. Don't put it on the critical path. Priority order for new ingestion: official policy APIs (free, dated) > Exa dated search > Firecrawl ($16–83/mo, has change-monitoring) > parse.bot.

---

## 6. The composed architecture: how it all fits

```
                        ┌────────────────────────────────────────────┐
                        │  CANDIDATE GENERATORS (cheap, automated)    │
                        │  • GEPA prompt variants (cached research)   │
                        │  • OpenEvolve/ShinkaEvolve code variants    │
                        │    (aggregation, calibration, features)     │
                        │  • Playbook deltas (post-mortem reflection) │
                        │  • New submodels (statsforecast/TSFM/LoRA)  │
                        │  • GABRIEL `discover` hypotheses            │
                        └────────────────┬───────────────────────────┘
                                         ▼
                        ┌────────────────────────────────────────────┐
                        │  RETROCAST = FITNESS FUNCTION               │
                        │  experiments YAML variant → clean stratum,  │
                        │  temporally held-out window, paired stats   │
                        └────────────────┬───────────────────────────┘
                                         ▼
                        ┌────────────────────────────────────────────┐
                        │  PROMOTION GATE (scripts/promote.py)        │
                        │  significant + ≥ noise floor + clean only   │
                        └────────────────┬───────────────────────────┘
                                         ▼
                live bot config / skills/ playbook / calibration/binary.json
                                         │
            resolutions ─────────────────┘  (post-mortems → new candidates)
```

Key design rules (each backed by verified evidence above):
1. **One gate.** Nothing self-promotes; the experiments paired-stats gate on temporally held-out clean questions is the only path to live config. (Every documented production flywheel keeps a statistical/human gate.)
2. **Optimize against caches.** Stage-level replay makes fitness calls free for everything downstream of LLM members; only prompt changes re-spend, and only member-call cost.
3. **Respect the noise floor.** Don't run loops whose expected effect size is below detectability for the available n — that's selecting on noise.
4. **Leakage discipline everywhere.** All loops inherit as-of/contamination machinery; add the GABRIEL staggered-cutoff test as a periodic audit.

---

## 7. Cost & time budget (phased)

| Phase | Work | Eng. time | One-off $ | Ongoing $ |
|---|---|---|---|---|
| 1 | Run current-bot-clean backtest; calibration flywheel; promotion gate script | 2–4 days | $30–60 | — |
| 2 | statsforecast + Chronos-2/TimesFM-2.5 zero-shot as ForecastSystems; CDF anchoring; dataset-question generator (ext. 4) | 1 week | ~$0 | — |
| 3 | GABRIEL evidence-rater into retrieval + Clio failure clustering + annotation rubric | 1 week | $20–80 | $10–30/mo |
| 4 | GEPA adapter (binary prompts, cached research) + playbook/skills loop | 1–1.5 weeks | $50–300 | $20–50/mo |
| 5 | OpenEvolve/ShinkaEvolve evaluator over cached artifacts (aggregation/calibration/features) | 1 week | $20–150 | per-run $1–120 |
| 6 (stretch) | TimesFM/TTM LoRA adapter collection + monthly refresh loop | 1 week | $10–30 | <$50/mo |
| 7 (stretch) | parse.bot trial + 1 live provider; Exa dated provider | 2–3 days | $0 | $0–30/mo |
| 8 (later) | RLVR open-weights ensemble member (Tinker/Together) | 2–4 weeks | $300–3,000 | serving costs |

**Total through phase 5: ~6–8 part-time weeks, $150–600 one-off, ~$50–100/mo.** All within the existing profile caps; one standard backtest ($150 cap) is the dominant recurring research cost.

---

## 8. Negative results & risks (verified — do not relearn these)

1. **Persona-simulation headline accuracy refuted 0-3** — don't build policy forecasting on silicon sampling yet (§4).
2. **OpenEvolve's self-reported benchmark wins refuted 0-3** — infrastructure yes, expected-gains no (§3c).
3. **parse.bot is live-only, no SLA, vendor-asserted reliability** — never on the backtest critical path (§5).
4. **Prompt optimization has diminishing returns on frontier models** (Metaculus's own optimizer: big gains GPT-4.1-nano, nothing for o3, *negative* for DeepSeek-R1; poor cross-provider transfer) — point GEPA at cheap members and research/summarization prompts, not at frontier reasoning members; consistent with repo doctrine "model >> prompting."
5. **GEPA overfits with large train sets** (Decagon: 500 examples underperformed 20–100) — small train sets, temporal validation.
6. **TSFMs add ~nothing on martingale market-price series** — aim submodels at indicators/counts, not at out-predicting Polymarket.
7. **Eval noise is the master constraint** — corpus growth beats loop sophistication; a loop making dozens of sequential selections on a few hundred questions mostly selects noise.
8. **Backtests remain relative-comparison tools** (repo doctrine, reaffirmed): promotion gates + prospective validation (MiniBench/bot-testing-area) stay mandatory.

---

## 9. Primary sources

- GABRIEL: [repo](https://github.com/openai/GABRIEL) · [announcement](https://openai.com/index/scaling-social-science-research/) · [NBER WP 34834 PDF](https://cdn.openai.com/pdf/7517a586-5bfa-4b87-bd3d-6ea0e9e844c7/GPT-as-a-measurement-tool.pdf) · [PyPI](https://pypi.org/project/openai-gabriel/)
- Anthropic: [Clio](https://arxiv.org/abs/2412.13678) · [OpenClio](https://github.com/Phylliida/OpenClio) · [Economic Index primitives](https://www.anthropic.com/research/economic-index-primitives) · [Anthropic Interviewer](https://www.anthropic.com/research/anthropic-interviewer) · [Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- Optimization/self-improvement: [GEPA](https://arxiv.org/abs/2507.19457) (ICLR 2026 oral) · [DSPy GEPA docs](https://dspy.ai/api/optimizers/GEPA/overview/) · [ACE](https://arxiv.org/abs/2510.04618) · [self-evolving-agents survey](https://arxiv.org/abs/2507.21046) · [SkillWeaver](https://arxiv.org/abs/2504.07079) · [MASS (prompts>topology)](https://arxiv.org/html/2502.02533v1) · [Metaculus prompt-opt experiment](https://www.lesswrong.com/posts/BStj6fgqxmrPqgPtw/analysis-of-automated-prompt-engineering-for-forecasting) · [Decagon GEPA production](https://decagon.ai/blog/optimizing-gepa-for-production)
- Algorithm discovery: [AlphaEvolve](https://arxiv.org/abs/2506.13131) · [FunSearch](https://www.nature.com/articles/s41586-023-06924-6) · [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) · [ShinkaEvolve](https://sakana.ai/shinka-evolve/) · [LLM-FE](https://arxiv.org/html/2503.14434v1) · [AutoGluon](https://github.com/autogluon/autogluon)
- Time series: [TimesFM repo](https://github.com/google-research/timesfm) · [TimesFM 2.5 card](https://huggingface.co/google/timesfm-2.5-200m-pytorch) · [LoRA example](https://github.com/google-research/timesfm/tree/master/timesfm-forecasting/examples/finetuning) · [Chronos-2](https://huggingface.co/amazon/chronos-2) · [Granite TTM](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) · [GIFT-Eval leaderboard](https://tsfm.ai/benchmarks/gift-eval) · [TSFMs-in-finance warning](https://arxiv.org/abs/2511.18578) · [In-Context Fine-Tuning](https://arxiv.org/pdf/2410.24087) · [From News to Forecast](https://arxiv.org/abs/2409.17515)
- Fine-tuning access (mid-2026): OpenAI FT API winding down (no new orgs since 2026-05-07, shutdown 2027-01-06, [deprecations](https://developers.openai.com/api/docs/deprecations)) · [Together pricing](https://www.together.ai/pricing) · [Tinker (hosted LoRA RL)](https://thinkingmachines.ai/tinker/) · [Turtel RLVR](https://arxiv.org/abs/2505.17989) · [Future-as-Label](https://arxiv.org/pdf/2601.06336)
- parse.bot: [site](https://parse.bot/) · [docs](https://docs.parse.bot/introduction) · [YC page](https://www.ycombinator.com/companies/parse-bot) · [HN thread](https://news.ycombinator.com/item?id=44833655) · [hermes-agent integration report](https://github.com/NousResearch/hermes-agent/issues/494) · alternatives: [Firecrawl](https://www.firecrawl.dev/pricing) · [Exa](https://exa.ai/pricing) · [Federal Register API](https://www.federalregister.gov/developers/documentation/api/v1)
- Power analysis inputs: [ForecastBench](https://arxiv.org/pdf/2409.19839) · [FRI Brier Index](https://forecastingresearch.substack.com/p/introducing-the-brier-index) · Wilks 2007 ([AMS](https://journals.ametsoc.org/view/journals/wefo/22/5/waf1034_1.xml))
