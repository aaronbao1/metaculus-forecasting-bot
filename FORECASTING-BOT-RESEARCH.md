# Building a Tournament-Winning Metaculus Forecasting Bot
## A Comprehensive Research Document: Architectures, Approaches & Methodologies

*Compiled June 11, 2026. Synthesized from peer-reviewed/arXiv literature, Metaculus's own tournament analyses and surveys, open-source bot codebases, the classic forecasting-science literature, and practitioner writeups. Every major claim is cited; deep-dive notes with full details live in [`research-notes/`](research-notes/).*

**Companion deep-dive notes:**
- [01 — Halawi et al. deep read](research-notes/01-halawi-deep-read.md) (the canonical LLM forecasting pipeline, full configs & ablations)
- [02 — LLM forecasting literature survey](research-notes/02-llm-forecasting-literature.md) (ForecastBench, RL fine-tuning, AIA Forecaster, failure modes)
- [03 — Superforecasting science](research-notes/03-superforecasting-science.md) (Tetlock/GJP, aggregation math, scoring theory, practitioner methods)
- [04 — Bot architectures](research-notes/04-bot-architectures.md) (template + winner codebases, verbatim prompts, survey data)
- [05 — Metaculus mechanics](research-notes/05-metaculus-mechanics.md) (exact scoring formulas, tournament history, API, rules)
- [06 — Adversarially verified findings](research-notes/06-deep-research-verified-findings.md) (104-agent verification sweep)

---

## Table of Contents

1. [Executive Summary: The State of Play (June 2026)](#1-executive-summary)
2. [The Arena: Metaculus FutureEval Tournaments & Scoring](#2-the-arena)
3. [Scoring Math → Strategy](#3-scoring-math--strategy)
4. [Foundations: The Science of Human Forecasting](#4-foundations)
5. [Academic Research on LLM Forecasting](#5-academic-research)
6. [Reference Architectures: What Winning Bots Actually Look Like](#6-reference-architectures)
7. [Methodology Component Deep-Dives](#7-methodology-components)
8. [Catalog of LLM Forecasting Failure Modes](#8-failure-modes)
9. [Evaluation & Benchmarking Methodology](#9-evaluation)
10. [Fine-Tuning and RL (the advanced lever)](#10-fine-tuning)
11. [Synthesis: An Evidence-Weighted Blueprint](#11-blueprint)
12. [Key Resources](#12-resources)

---

<a name="1-executive-summary"></a>
## 1. Executive Summary: The State of Play (June 2026)

**Where the field is.** No bot has beaten Metaculus Pro forecasters in a live tournament. The Pros won every measured season: head-to-head spot peer score (bot team minus pro team) went −11.3 (Q3 2024) → −8.9 (Q4 2024, not significant) → −17.7 (Q1 2025) → −20.03 (Q2 2025). On the FutureEval Model Leaderboard (May 2026 snapshot), Pros sit at 36.0 and the Metaculus Community Prediction at 22.4, vs. the best bare model (Claude Opus 4.5 High) at 19.5. Metaculus's own trendlines project bots passing the Community Prediction ~April 2026 (i.e., roughly now) and the Pros ~mid-2027. On ForecastBench, engineered systems reached statistical superforecaster parity in late 2025 (Bridgewater's AIA Forecaster: Brier 0.1076 vs supers' 0.1110), but that's a backtest-style benchmark; the live-tournament gap persists. **Translation: the window where a well-engineered bot can win bot tournaments and plausibly become the first to beat the human crowd is open right now.**

**What wins (the five-line version):**
1. **Frontier reasoning model** — base model choice dominated everything in 2024–mid-2025; by Fall 2025 frontier models became table stakes and **scaffolding now differentiates** (~27 peer points/question spread within the GPT-5 family from scaffolding alone; "good scaffolding ≈ 9 months of base-model progress").
2. **Multiple research sources, agentically searched** — number of distinct research sources was the #1 predictor of Fall 2025 success (r=0.42, p=0.006); iterative search beats one-shot retrieval; raw news dumps can *hurt*.
3. **Ensembling with robust aggregation** — many independent runs (winners' median: ~28 LLM calls/question), across 2+ model families, aggregated by median/trimmed mean/geometric-mean-of-odds. Worth +1,799 coverage-adjusted points in Q2 2025.
4. **Calibration discipline** — cap outputs (~[2%, 98%]; strongest within-winner score correlate, r=+0.48), correct the documented yes-bias, fit Platt scaling on your own resolved history (+0.016 Brier on Metaculus data).
5. **Rigorous self-evaluation** — testing on custom-created resolution questions was the single practice most associated with winning (+2,216 points); never trust pre-cutoff backtests.

**The open frontier.** Bots lose to Pros on **discrimination, not calibration** (Pros' yes-vs-no forecast gap: 44pp vs bots' 26pp). The biggest known headroom: (a) gathering decisive evidence and committing to justified extremes; (b) numeric and multiple-choice questions, where bots' deficit is 1.5–2× worse than binary; (c) decorrelated ensemble members (possibly including an outcome-fine-tuned open model).

---

<a name="2-the-arena"></a>
## 2. The Arena: Metaculus FutureEval Tournaments & Scoring

*(Full details: [research-notes/05](research-notes/05-metaculus-mechanics.md))*

### 2.1 Tournament structure (Year 2, current)
- **$175k/year**: three seasonal $50k tournaments (Fall / Spring / **Summer 2026 — live now, started May 18, 2026**, ID `summer-futureeval-2026`), 300–500 questions each, plus bi-weekly $1k **MiniBench** tournaments (~60 auto-generated, auto-resolving questions — ideal for iteration).
- Question types: **binary, numeric (continuous), discrete, multiple choice**. Questions launch in batches (up to ~5–10 at random times), stay open **~1–1.5 hours**, and are **spot-scored at close** — your bot must run on a schedule (the template polls every 20 minutes via GitHub Actions) and the last forecast before close is what counts. No updating needed after submission.
- The human Community Prediction and Pro forecasts are **hidden** on bot-tournament questions (no copying). Main-site CP on synced questions is visible and legal to use.
- **Rules:** no human in the loop (no manual forecasts, no previewing open questions; scheduled triggering is fine); teams allowed (one prize-eligible bot); winners must show/describe code; post-season survey mandatory for prize eligibility. Free credits: Metaculus-managed OpenRouter keys (Anthropic/Google/OpenAI) + AskNews (1,000 calls/month, 4,000/tournament).
- Bot registration: account → Settings → "My Forecasting Bots" → token; the [metac-bot-template](https://github.com/Metaculus/metac-bot-template) is fork-and-go.

### 2.2 The scoring system (exact)
Everything is built on the **log score**:
- Binary/MC: `ln(P(outcome))`. Continuous: `ln(pdf(resolution value))`, where every pdf gets a **uniform 0.01 floor added** and pdf values are capped ~35.
- **Baseline score** (vs. an ignorant prior): binary `100 × (log₂ P + 1)`; MC scaled by `ln N`; continuous `100 × (ln pdf − ln baseline)/2`.
- **Peer score** (what tournaments use): `100 × [ln p − ln(geometric mean of all other forecasters' p)]`. Zero-sum across participants. Continuous/discrete peer scores divided by 2.
- **Spot peer score** = peer score evaluated at one instant (question close in bot tournaments). No time-averaging, no coverage decay — only your standing forecast at close matters.
- **Tournament ranking:** `total = Σ (spot peer × question weight)`, `take = max(total, 0)²`, prizes ∝ take. Squaring rewards both accuracy and volume — forecast everything unless you're below field-average on a question type.
- Platform clamps: binary forecasts ∈ **[0.1%, 99.9%]**; numeric CDF = 201 points, each step ≥ 5e-05, max per-bin mass 0.2 (tightened from 0.59 in Spring 2026).

---

<a name="3-scoring-math--strategy"></a>
## 3. Scoring Math → Strategy

The scoring rules have direct strategic consequences:

1. **Properness per question:** spot peer is proper — report your true belief on each question. There is no per-question gaming.
2. **Log-score asymmetry near extremes:** going 99% → 99.9% gains +0.009 log points if right but loses −2.3 if wrong. One hallucinated "this already resolved" forecast at 99% can erase a season (observed failure mode). Hence the near-universal winner practice of **capping at ~[2%, 98%]** — the strongest within-winner score correlate in Fall 2025 (r = +0.48, p = 0.005). The cap is *defensive*: it bounds tail risk from model errors, not a hedge on honest beliefs.
3. **But discrimination wins:** peer score is zero-sum against other bots. Pros crush bots by being *justifiably* extreme (going <2% / >90% when evidence warrants). Clustering near 50% bleeds points to bolder competitors. The optimal posture: **be as extreme as your evidence justifies, inside a hard safety cap**, and invest in the research that justifies extremity.
4. **Skipping is sum-positive when you're below-average:** unanswered questions score 0 while a below-field-average forecast has negative expected spot peer. If your numeric pipeline is weak, fixing it beats forecasting badly — but for an above-average bot, every missed question is forfeited points (and take² rewards volume). Also: withdrawal via API exists; auto-withdrawal defaults exist for non-tournament use.
5. **Don't copy the crowd:** the CP is hidden in bot tournaments anyway; on visible-CP venues, CP-mimicking prompts that won offline evals *underperformed live* (+2.76 vs +4.71 per question). Use external market prices/CP as one anchor input, not the target. (Showing a model the crowd median and averaging *does* help accuracy in research settings — Schoenegger et al. found +17–28% — the failure mode is *optimizing to match* the crowd rather than using it as evidence.)
6. **Time-averaged venues differ:** in human tournaments (Metaculus Cup etc.) scores are time-averaged with truncation at early resolution — forecast early, update often, and burn down deadline-question probabilities (see §7.6). In spot-scored bot tournaments, only the close-time forecast matters.

---

<a name="4-foundations"></a>
## 4. Foundations: The Science of Human Forecasting

*(Full details: [research-notes/03](research-notes/03-superforecasting-science.md). This is the intellectual bedrock — nearly every winning bot technique is a mechanization of something here.)*

### 4.1 What makes superforecasters super (Tetlock/GJP, quantified)
- GJP won IARPA's ACE tournament by 35–72% over rival academic teams. Superforecasters vs. regulars: calibration error 0.01 vs 0.04, AUC 96% vs 75%; their forecasts 300 days out matched regulars' at 60 days out.
- **Granularity carries information:** supers used 57 distinct probability values (multiples of 1%); rounding their forecasts to the nearest 10% measurably *worsened* Brier. → A bot should output 1%-granular, non-round probabilities.
- **Frequent small updates:** supers made ~6.7 forecasts/question vs ~2.9 for others.
- **Scope sensitivity:** supers' probabilities responded ~2× more appropriately to changes in deadlines/thresholds. (Bots tested in Q3 2024 were badly scope-insensitive: P(B)+P(C) = 1.24×P(A) vs Pros' 1.02×.)
- **Fox > hedgehog:** many small models, eclectic evidence, self-critical updating. Hedgehogs' decomposed scenario-tree probabilities summed to 1.58 (incoherent!) — a bot can enforce coherence mechanically where humans can't.
- The **Ten Commandments** (triage; Fermi-decompose; outside view first; balance under/overreaction to evidence; clashing causal forces; granular doubt; balanced confidence; error post-mortems; team perspective-taking; deliberate practice) are directly prompt-encodable.

### 4.2 The single most valuable trainable skill: comparison classes
GJP's one-hour CHAMPS-KNOW training improved Brier 6–12% season-long. Decomposing which principle mattered: forecasts explicitly invoking **comparison classes (base rates / outside view)** scored Brier 0.17 vs 0.49 for control — by far the largest effect of any principle. Kahneman's outside-view-first discipline (find the reference class, take its base rate, then adjust with case specifics) is the load-bearing wall of forecasting methodology.

### 4.3 Aggregation science (the math that won ACE)
- GJP's winning aggregate = recency/track-record-weighted forecasts + **extremized logit aggregation** (Satopää et al. 2014): transform `t(p) = p^a / (p^a + (1−p)^a)` with a ≈ 2.5 (optimal range 1.16–3.92).
- **Two reasons aggregates need extremizing** (Baron et al. 2014): (i) means are compressed toward 0.5 near the ends of the scale (medians immune); (ii) each individual rationally shades toward 50% for privately-missing information, but the crowd collectively holds more info — so the aggregate should be more confident than its average member. More expert/correlated crowds need *less* extremizing (a ≈ 1.6–2.4); Neyman & Roughgarden's principled n-dependent factor → √3 ≈ 1.73.
- **Pooling functions:** geometric mean of odds (= mean of log-odds) beat arithmetic mean of probabilities and median on ~850 resolved Metaculus questions (Brier 0.116 vs 0.122 vs 0.134); it's "externally Bayesian" and respects information in extreme forecasts. **Trimmed pools** (drop extremes, then pool) fix the diversity-induced underconfidence of linear pools (Jose et al. 2014).
- **Overfitting warning:** you can *always* find an extremization parameter that improves historical forecasts; on recent Metaculus data extremized variants underperformed plain GMO because the crowd was already confident. Extremize only with independent evidence of underconfidence — or fit on truly held-out data.
- **Samotsvety's process** (arguably the world's best forecasting team): independent forecasts first (no anchoring) → optional different decompositions per member → discussion → **drop one most-extreme forecast per side → geometric mean of odds**.

### 4.4 Scoring-rule theory
Both Brier and log are strictly proper: truthful reporting maximizes expected score; "hedging toward 50%" loses EV under both. The difference is tail behavior: log score → −∞ as p(outcome) → 0, so **overconfidence near 0/1 is survivable under Brier and fatal under log** — and Metaculus is log-based. Brier decomposes into calibration + resolution (+ irreducible noise): track both for your bot; the resolution (discrimination) term is where bots currently lose.

### 4.5 Time decay and hazard rates
For "Will X happen by date Y?" questions, absent news the correct probability **decays toward 0 as the window shrinks** (constant-hazard model: solve `(1−r)^T = 1−P` for per-period hazard r, then `P_now = 1−(1−r)^t_remaining`). Use Laplace's rule of succession `(k+1)/(n+2)` when the hazard itself is unknown. Real questions have *lumpy* hazard (elections, rulings, scheduled launches): probability should step down as catalysts pass uneventfully, not glide. A documented community failure is forecasts staying too high too long, then crashing late. (In spot-scored bot tournaments this matters less since you forecast near close; it matters greatly in time-averaged venues.)

---

<a name="5-academic-research"></a>
## 5. Academic Research on LLM Forecasting

*(Full details: [research-notes/01](research-notes/01-halawi-deep-read.md) and [02](research-notes/02-llm-forecasting-literature.md))*

### 5.1 The canonical pipeline: Halawi, Zhang, Yueh-Han & Steinhardt (NeurIPS 2024)
"Approaching Human-Level Forecasting with Language Models" ([arXiv 2402.18563](https://arxiv.org/abs/2402.18563)) built the reference retrieval-augmented architecture:
**query generation** (2 prompts: direct expansion + sub-question decomposition, ~12 queries) → **news APIs** (NewsCatcher + Google News, top 10/query, date-bounded) → **LLM relevance rating** (1–6 scale on title+first 250 words; keep ≥4) → **question-conditioned ≤100-word summaries** → **top-15 by relevance into context** → **reasoning** (3 best scratchpad prompts on GPT-4 + 3 samples from a fine-tuned model) → **trimmed-mean ensemble of 6**.

Key numbers and lessons:
- System Brier **.179** vs crowd **.149** (914 post-cutoff binary questions). **Beat the crowd where the crowd was uncertain** (p∈[.3,.7]: .238 vs .240) and when retrieval was rich; system+crowd average beat the crowd everywhere (.149→.146).
- **Ablations:** retrieval is the biggest component (no-retrieval = .206 vs .186); fine-tuning +.007; ensembling +.003; prompt optimization alone ≈ nothing. The best scratchpad: rephrase → reasons-no → reasons-yes → "aggregate like a superforecaster" → initial probability → over/underconfidence + base-rate check → final.
- Fine-tuning trick: self-supervised on outcomes — keep only reasoning traces that beat the crowd, discard predictions >0.15 from crowd, average targets with the crowd. (Anti-overconfidence by construction.)
- **Dominant failure mode: hedging away from 0/1** (attributed to safety training) — worst exactly where log-based Metaculus scoring punishes.
- Off-the-shelf 2023-era models without retrieval were near-random (.208–.35 vs crowd .149); Schoenegger & Park's GPT-4 entered a real 2023 Metaculus tournament and failed to beat always-50%.

### 5.2 ForecastBench: the contamination-free yardstick
([arXiv 2409.19839](https://arxiv.org/abs/2409.19839), ICLR 2025; [forecastbench.org](https://www.forecastbench.org)) — ~1,000 questions/round about genuinely future events, market-sourced + time-series-derived. Findings: superforecasters 0.093 vs best LLM 0.111–0.122 (2024); scratchpad prompting + showing market "freeze values" helped most; naive news addition sometimes hurt. By mid-2026: supers' Brier Index 70.6% vs best AI 67.9% (CassiAI ensemble, Grok 4.20); SOTA improves ~0.016 Brier/year; extrapolated parity **Nov 2026** (95% CI Dec 2025–Jan 2028). Caution: on market questions LLMs' outputs correlate 0.994 with shown market prices — anchor-copying masquerading as skill.

### 5.3 The strongest published system: AIA Forecaster (Bridgewater, Nov 2025)
([arXiv 2511.07678](https://arxiv.org/html/2511.07678v1)) — first credible superforecaster-parity system: **agentic iterative search** per agent → **~10 independent agent forecasts** → **supervisor agent** that investigates disagreements with targeted follow-up search (not naive averaging) → **Platt-scaling calibration layer** fitted on resolved questions. Brier 0.1076 vs supers' 0.1110. Ablations: agentic search worth ~0.009; Platt scaling worth ~0.0064 (as much as the supervisor); ~10-run averaging "mandatory" (mean ≈ median ≈ trimmed mean at that n). Honest leakage accounting (~1.65% of searches leaked; bounded ≤0.6% effect).

### 5.4 Other load-bearing results
- **Wisdom of the Silicon Crowd** (Schoenegger et al., Science Advances 2024): 12-model LLM ensemble matched a 925-human crowd (Brier 0.20 vs 0.19); showing models the human median improved them 17–28%; documented **acquiescence bias** (mean prediction 57.4% when only ~45% resolved Yes) and round-number clumping.
- **Prompt micro-optimization doesn't save you** (Schoenegger, Jones, Tetlock, Mellers, [arXiv 2506.01578](https://arxiv.org/abs/2506.01578)): 38 prompts × 4 models → negligible gains; small benefit from base-rate references; "think Bayesianly" prompts *hurt*. (Metaculus's own automated prompt-optimization replicated this: big gains for weak models, ~nothing for o3/R1, no cross-provider transfer.)
- **Consistency checks** (Paleka et al., [arXiv 2412.18544](https://arxiv.org/abs/2412.18544), ICLR 2025 Oral): frontier forecasters violate negation/partition/monotonicity coherence enough to be arbitraged; violation metrics — computable instantly — predict future Brier. Use as free CI.
- **Evaluation pitfalls** (Paleka et al., [arXiv 2506.00723](https://arxiv.org/html/2506.00723v1); "Simulated Ignorance Fails," arXiv 2601.13717): backtests on pre-cutoff questions are unfixably leaky (prompting a model to "pretend it's 2023" leaves a 52% performance gap vs true ignorance); date-restricted search still leaks via re-dated articles. → Evaluate prospectively or on frozen-corpus pastcasting (FutureSearch's Bench-to-the-Future).
- **FutureSearch BTF-2** (arXiv 2604.26106): frozen 15M-doc corpus pastcasting; best frontier agent (Opus 4.6, 0.130) still beaten by a composite forecaster (0.119); qualitative edge of the best forecasters = **pre-mortem/black-swan analysis**; documented frontier-agent weak spots: leaders' incentives, follow-through on plans, institutional processes.

---

<a name="6-reference-architectures"></a>
## 6. Reference Architectures: What Winning Bots Actually Look Like

*(Full details with verbatim prompts: [research-notes/04](research-notes/04-bot-architectures.md))*

### 6.1 The baseline: official metac-bot template
The [metac-bot-template](https://github.com/Metaculus/metac-bot-template) + [forecasting-tools](https://github.com/Metaculus/forecasting-tools) library define the floor (and the floor is high — the template placed 13th of 123 in Fall 2025; its o1 version *won* Q1 2025):
- **Flow:** poll tournament every 20 min → per question: 1 research pass (AskNews summaries by default; Perplexity/Exa/search-enabled-LLM alternatives) → **5 independent forecasts** with a single structured prompt → **median** (binary), per-option mean (MC), pointwise-median of 201-point CDFs (numeric) → clamp [1%, 99%] → submit + private reasoning comment.
- **The prompt** (binary): "professional forecaster interviewing for a job" persona; writes (a) time to resolution, (b) **status-quo outcome**, (c) a No scenario, (d) a Yes scenario; "good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time"; final answer "Probability: ZZ%".
- **Numeric:** elicits 6 percentiles (10/20/40/60/80/90), reminds the model to set "wide 90/10 confidence intervals to account for unknown unknowns"; builds the 201-point CDF by linear interpolation + constraint standardization.
- `forecasting-tools` also ships: SmartSearcher (Exa-backed cited research), BaseRateResearcher/NicheListResearcher (reference-class construction), KeyFactorsResearcher, QuestionDecomposer, Benchmarker (vs community prediction), calibration adjusters (Platt/logistic etc.), and MetaculusApi (incl. withdrawal, tournament constants).

### 6.2 Winner blueprints (chronological)
- **pgodzinai (Q4 2024 winner, $9,658; 15–40 hours of work):** Perplexity + AskNews research with per-genre prompt guidance; 3× GPT-4o + 5× Claude-3.5-Sonnet; **drop two most extreme, average the six remaining**; superforecaster persona + scoring explanation + explicit prior correction ("the base rate for positive resolutions on Metaculus is 35%"); **batch-predicted related questions together for internal consistency**.
- **metac-o1 (Q1 2025 de-facto winner) / metac-o3 (Q2 2025 #2):** the plain 5-run template on the best available reasoning model + AskNews. The era's lesson: **model > scaffolding**.
- **Panshul42 (Q2 2025 winner, $7,550, open source):** o3 plans **dual research tracks** — historical/outside-view queries and current/inside-view news — routed per-engine (Serper Google/News, AskNews, Perplexity sonar-reasoning-pro, BrightData scraping); stage-1 outside-view report → stage-2 inside-view with an **evidence-strength rubric** (strong/moderate/weak criteria) and a 6-item checklist (paraphrase resolution criteria; outside view first; "X out of 100 times" consistency line; evidence validity; **blind-spot statement** — "name the scenario most likely to make your forecast look silly"; status-quo nudge); ensemble: 2× Sonnet + 2× o4-mini + 1× o3, weighted [1,1,1,2,2]; numeric = weighted average of full CDFs; a Monte-Carlo variant has the LLM write and execute NumPy simulations.
- **smingers (Jordan Berman — an actual GJ superforecaster automating himself):** research (AskNews/Serper/Wikipedia/yFinance/FRED/Trends) → **5 outside-view forecasters across 3 models** → **5 inside-view instances adjusting the outside view** → **Claude Opus "Supervisor" invoked only when ensemble divergence is high** (~1 in 6 questions). 70+ LLM calls/question. "Tools are more powerful than prompts."
- **nostreambot (best open-source, Fall 2025, 9th):** "biggest gains came less from clever prompting and more from boring engineering: model ensembling, continuous-question handling, reliability, and better evaluation." Multi-model median ensemble (single largest contributor); **PCHIP smoothing** of numeric CDFs + added p2.5/p97.5 extreme quantiles for fatter tails; FRED/Yahoo for financial questions. Reliability math: missing 10 questions ≈ −150 points; one units bug ≈ −80.
- **GreeneiBot2 (top of Spring 2026):** parallel research across AskNews + Perplexity (2 modes) + GPT-5 search + **link assistants that screenshot the resolution-source URLs** and read them with vision/BrowserUse; structured CDFs with spikes; validation against platform rules; multi-run aggregation.
- **Mantic (first bot in a human-tournament top-10, Fall 2025 Metaculus Cup 8/549):** two-phase research→prediction; numeric questions answered as **LLM-specified mixture distributions**; **RL-fine-tuned gpt-oss-120b** (GRPO-style, Brier reward, ~10k resolved questions) that earned 40–56% of ensemble weight by being decorrelated with the frontier members.
- **lightningrod (Q2 2025 numeric leaderboard winner):** "Future-as-Label" outcome-supervised fine-tuning ([arXiv 2601.06336](https://arxiv.org/abs/2601.06336)); fine-tuned Qwen3-32B: Brier −27%, calibration error halved, beats Qwen3-235B.

### 6.3 The meta-lesson across seasons
2024–mid-2025: single-prompt bots on the best reasoning model beat elaborate multi-agent systems on weaker models. **Fall 2025 onward: frontier models converged ("table stakes") and scaffolding became the differentiator** — top-5 scaffolded bots beat their same-model house baseline by 5–11 peer points/question; the spread *within* the GPT-5 family from scaffolding alone (~27 points/question) now exceeds the frontier-vs-non-frontier model gap. Costs are modest: Fall 2025 winners' median ≈ **$0.90/question** (top-15 ≈ $1.40).

---

<a name="7-methodology-components"></a>
## 7. Methodology Component Deep-Dives

### 7.1 Research / retrieval pipeline
**Evidence:** retrieval was Halawi's biggest ablation (~0.02 Brier); agentic search worth ~0.009 Brier for AIA; *number of distinct research sources* was the #1 Fall 2025 winner predictor; but ForecastBench showed naive news-dumping can hurt.
**Design principles:**
- **Multiple heterogeneous providers** (winning stacks: AskNews + Perplexity + custom scraper; AskNews + Exa + OpenAI web search). Providers have different blind spots; per-provider query styles matter (keyword-short for Google, instruction-long for Perplexity).
- **Agentic/iterative, not one-shot:** generate queries → search → identify gaps → search again. Native provider web-search tools (OpenAI/Anthropic/Gemini/Grok search) increasingly beat custom pipelines because models are RL-trained to use them.
- **Two-track research:** explicitly separate **outside-view/historical** retrieval (base rates, reference classes, similar past events) from **inside-view/current** retrieval (latest news, expert statements). Both Panshul42 and smingers structure this way.
- **Relevance-filter before context:** rate article relevance cheaply, discard low scores, summarize question-conditioned, order by relevance (Halawi's recipe). Quality over volume.
- **Specialized data for specialized questions:** FRED/Yahoo Finance for econ/markets, structured datasets for time-series questions, screenshots of the actual resolution-source page (GreeneiBot2) to ground "what would this resolve right now."
- **Date hygiene:** always inject today's date; bound searches to the question window during backtests.

### 7.2 Reasoning & prompting
**Evidence:** structured scratchpads help; micro-optimization beyond that doesn't (for frontier models); specific content elements with measured support: status-quo weighting, base-rate references, premortem/blind-spot steps, evidence-strength rubrics, scope-sensitivity checks.
**A consensus scratchpad** (synthesis of template + Halawi + Panshul42 + GJP):
1. Restate the question and **resolution criteria literally** (forecast the criteria, not the title's vibe); note dates/thresholds and time remaining.
2. **Outside view first:** name 2–3 candidate reference classes with counts where possible; state a base rate.
3. Status quo: what happens if nothing changes? (Weight it — the world changes slowly.)
4. Inside view: strongest evidence/scenarios for each outcome, graded by an evidence-strength rubric (independent confirmation > causal mechanism > expert opinion > anecdote/speculation).
5. Clashing forces + **premortem**: "it's resolution day and I was badly wrong — why?" / blind-spot statement.
6. Hazard/time check for deadline questions; scope-sensitivity self-test (would my number move appropriately if the deadline doubled?).
7. Initial probability → over/underconfidence + base-rate re-check → **final probability at 1% granularity** (sanity line: "X out of 100 times…").
**Anti-patterns:** "think Bayesianly" instructions (measured harm); persona-stacking and elaborate role-play beyond one clean persona; letting the model see only pro-Yes news (acquiescence amplification).

### 7.3 Ensembling & aggregation
**Evidence:** the single most consistently validated technique (Q2 2025: +1,799 points; AIA: ~10 runs "mandatory"; silicon-crowd parity).
- **Within-model:** 5–10 independent runs minimum (template uses 5; AIA 10). Diminishing returns after ~10.
- **Across models:** 2–3 decorrelated families (e.g., GPT-5.x + Claude Opus/Sonnet + Gemini/Grok/R1). Decorrelation is the value (Mantic's fine-tune earned its weight by disagreeing usefully).
- **Pooling function:** median (template default, robust), trimmed mean (drop extremes then average — pgodzinai, Samotsvety, Jose et al.), or geometric mean of odds (best on historical Metaculus data, respects extreme information). At n≈10 the choice is second-order; *doing* aggregation is first-order. For numeric: pointwise median/weighted-average of full CDFs.
- **Supervisor reconciliation:** when ensemble variance is high, a strong model investigates the disagreement with follow-up research instead of blind averaging (AIA ~0.005 Brier; smingers' divergence-triggered Opus supervisor).
- **Extremizing:** theoretically justified for diverse crowds shading toward 50 (a ≈ 1.5–2.5); empirically helps when members hedge, hurts when members are already confident/correlated; fitted extremization overfits in-sample. Prefer **fitted Platt scaling on your own resolved-question history** (+0.016 Brier on Metaculus bot data, p=0.0005) over blind extremization, with separate maps per question type.

### 7.4 Calibration layer (post-processing)
Ordered pipeline applied to the aggregate:
1. **Yes-bias correction** — LLMs say yes too often (57% mean prediction vs 45% base rate; "when bots say 29% it happens 12% of the time"; Metaculus's suggested 30%→20% style adjustment; Metaculus event-question resolution base rate ≈ 35%).
2. **Platt/logistic recalibration** fitted on your bot's resolved history (worth ~0.006–0.016 Brier; refit per season and per question type; don't fit on <100 resolutions).
3. **Hard caps ~[2%, 98%]** (or [1%, 99%]) as the last step — tail-risk insurance against hallucinated certainty under log scoring.
4. **Coherence enforcement (mechanical):** MC options sum to 1 with a floor on every option ("leave moderate probability on most options"); date-ladder monotonicity (P(by 2026) ≥ P(by 2025)); related-question batch consistency (pgodzinai's innovation); negation consistency spot-checks.

### 7.5 Numeric, discrete & multiple-choice questions (the open weakness)
Bots' head-to-head deficit vs Pros: binary −14.8, numeric −23.2, MC −32.9 (Q2 2025) — **the highest-leverage question types to be good at**, and where leaderboards were won (lightningrod took numeric-only; "almost all numeric questions are easy" per MiniBench analysis — for a bot with solid plumbing).
- Elicit percentiles (template: 10/20/40/60/80/90) — or **mean + std-dev** (one Fall winner found it more reliable), or **mixture distributions** (Mantic: LLM picks components, parameters, weights).
- **Fatten the tails deliberately:** add p2.5/p97.5 elicitation (nostreambot); "tails systematically not fat enough" is also the top human-expert error (Lifland). Smooth with **PCHIP** rather than linear interpolation; respect the 201-point CDF constraints (min step 5e-05, max bin 0.2 — validation failures hit ~30% of numeric runs for one bot; build a standardizer + retry loop).
- For closed bounds put zero mass outside; for open bounds keep the minimum mass outside; never let percentile parsing produce non-monotonic values (jitter duplicates).
- Use **code execution** where the question is computational: LLM-written Monte Carlo (Panshul42's ≥100k-iteration NumPy sims), trend extrapolation from FRED/Yahoo data, Laplace rules on count data.
- MC: forecast as a distribution (not the argmax), renormalize after flooring each option (~1–2%), and remember "good forecasters leave some moderate probability on most options."

### 7.6 Updating & time decay
In spot-scored bot tournaments, forecast near the close of the (short) window, so freshness is automatic. In time-averaged venues (Metaculus Cup, Market Pulse — also bot-eligible): forecast early (coverage), update on news (supers: many small updates), and **re-emit decayed probabilities on a schedule even without news** for deadline questions (constant-hazard decay or catalyst-schedule steps; Laplace `(k+1)/(n+2)` for unknown hazards). Failing to burn down is a continuous score leak.

### 7.7 Question triage
Tetlock's first commandment, encoded: classify each question — (a) near-certain (forecast the obvious extreme, capped; don't overthink), (b) genuinely uncertain Goldilocks (spend the full pipeline; this is where bots beat crowds — Halawi's selective-setting result), (c) noise/unknowable (output base rate / wide distribution cheaply). Allocate research depth and ensemble size by tier; route weak question types (your bot's, per benchmark data) to fallback strategies or skip.

---

<a name="8-failure-modes"></a>
## 8. Catalog of LLM Forecasting Failure Modes

| # | Failure mode | Evidence | Mitigation |
|---|---|---|---|
| 1 | **Hedging toward 0.5** (underconfidence at extremes; safety-training artifact) | Halawi's dominant failure; AIA's Platt layer worth 0.0064 | Fitted recalibration; supervisor pass; decisive-evidence prompts |
| 2 | **Yes/acquiescence bias** | Mean prediction 57.4% vs 45% Yes rate; bots' 29% → 12% reality | Systematic downward adjustment; status-quo weighting; negation checks |
| 3 | **Believing the event already happened** → 99% on unresolved questions | Fall 2025 advice post ("massive negative score") | Resolution-source screenshot check; cap; "has this actually resolved?" verifier |
| 4 | **Round-number clumping & run-to-run variance** (30/30/40/40/60 on the same question) | Silicon Crowd; titotal's 539 audit | Ensembling (variance); 1%-granularity instruction |
| 5 | **Scope insensitivity** (deadline/threshold changes don't move the number) | Bots: P(B)+P(C)=1.24×P(A) vs Pros 1.02× | Scope self-test step; date-ladder coherence enforcement |
| 6 | **Logical incoherence** (negations, partitions, monotonicity) | Paleka et al. — arbitrageable; predicts bad Brier | Mechanical coherence layer; batch related questions |
| 7 | **Market/CP parroting** (0.994 correlation with shown prices) | ForecastBench imputed-forecaster result | Use markets as one input; require reasons to deviate or concur |
| 8 | **Poor numeric distributions** (thin tails, non-monotonic percentiles, units errors) | −23.2 numeric deficit; nostreambot's −80-point units bug | §7.5 machinery + validation/retry + fat-tail elicitation |
| 9 | **Knowledge-cutoff confabulation / leaky backtests** | 52% gap for "simulated ignorance"; Pitfalls paper | Prospective evals; MiniBench; frozen-corpus pastcasting only |
| 10 | **Hallucinating research** (deep-researcher agents fabricating) | CumulativeBot: "every hallucination tracked back to it" | Cited retrieval; relevance filtering; cross-source verification |
| 11 | **Recurrence underestimation & equilibrium thinking** | Lifland retro (COVID waves) | Historical-pattern prompts; hazard-rate framing |
| 12 | **Weak on incentives/institutions** | FutureSearch expert review | Targeted "power players & incentives" prompt section (CHAMPS-KNOW "K/N") |
| 13 | **Engineering failures** (missed questions, API rejections, format bugs) | −150 pts for 10 missed questions; "1 mentionable bug per season" even for Metaculus | Retries, validation, alerting, idempotent scheduling, format fuzz tests |

---

<a name="9-evaluation"></a>
## 9. Evaluation & Benchmarking Methodology

The practice most strongly associated with winning (+2,216 points, Q2 2025) is **building your own eval harness**:
1. **Custom resolution questions:** write questions that resolve soon (days/weeks), run the bot, score against reality. Catches reasoning *and* plumbing bugs.
2. **MiniBench** (bi-weekly, $1k, auto-resolving): the official iteration sandbox; plus the **bot-testing-area** tournament (one practice question per type) for format testing.
3. **Community-prediction benchmarking:** `forecasting-tools` Benchmarker scores expected baseline score vs CP on hundreds of open questions. Cheap and immediate but capped at "match the crowd" — top bots profit by *disagreeing well*; within winners, CP-checking correlates slightly negatively with score. Use it as a regression test, not an objective.
4. **Consistency-check CI** (free, instant): negation pairs, MC sum-to-one, date-ladder monotonicity — violations predict bad Brier (Paleka).
5. **Statistical power discipline:** ~100 questions gives a ~30% chance the worse bot wins a comparison; aim for 300–500+ before believing an architecture change. Log everything (config, commit hash, cost, per-question scores) — Benchmarker does this natively.
6. **Never backtest on pre-cutoff questions** (§8.9). If you must pastcast, use frozen-corpus setups (BTF-style) and treat results as upper bounds.
7. **Track the Brier decomposition** (calibration vs resolution) per question type — it tells you whether to invest in recalibration (calibration deficit) or research/decisiveness (resolution deficit). Current field-wide answer: resolution.

---

<a name="10-fine-tuning"></a>
## 10. Fine-Tuning and RL (the advanced lever)

Almost no tournament winner has fine-tuned (Q4 2024: one of the top bots) — but the research results are now strong and two production systems (Mantic, lightningrod) prove it out:
- **Outcome-supervised SFT** (Halawi): generate many reasoning traces on resolved questions, keep only those beating the crowd & near the crowd, fine-tune. +0.007 Brier in 2024.
- **Self-play DPO** (Turtel et al. [2502.05253](https://arxiv.org/abs/2502.05253)): rank trace pairs by outcome distance; +7–10% accuracy at 14B scale.
- **RLVR with Brier rewards** (Turtel et al. [2505.17989](https://arxiv.org/abs/2505.17989); Mantic/Thinking Machines): 14B RL-tuned model beat o1 on accuracy *and* calibration (ECE 0.062 vs 0.093); Mantic's RL'd gpt-oss-120b reached Gemini-3-Pro parity on forecasting. **Critical gotcha:** vanilla GRPO's advantage normalization causes overconfidence collapse (39% of predictions in extreme bins) — use ReMax/modified advantages + guardrails. Brier reward is more stable than log.
- **Future-as-Label** (lightningrod, [2601.06336](https://arxiv.org/abs/2601.06336)): train on historical news snapshots, score against what happened; Qwen3-32B → Brier −27%, calibration error halved.
- **Strategic role:** the fine-tuned model's main value in 2026 is as a **decorrelated ensemble member** (Mantic: 40–56% ensemble weight, "least replaceable" after Grok 4), not as a sole forecaster. Data is available (resolved Metaculus/Polymarket/Manifold questions; Halawi's HuggingFace datasets; lightningrod SDK). This is a Phase-2+ investment after the scaffold is competitive.

---

<a name="11-blueprint"></a>
## 11. Synthesis: An Evidence-Weighted Blueprint

What the totality of evidence says the best 2026 bot looks like, by layer, with the measured value of each decision:

```
┌──────────────────────────────────────────────────────────────────────┐
│ SCHEDULER  (GitHub Actions / cron, every 20 min; idempotent;         │
│ retries; alerting; never miss a question — 10 misses ≈ −150 pts)     │
├──────────────────────────────────────────────────────────────────────┤
│ TRIAGE  (classify: near-certain / Goldilocks / noise; route effort;  │
│ detect question type → type-specific pipeline; "already resolved?"   │
│ verifier against resolution source)                                  │
├──────────────────────────────────────────────────────────────────────┤
│ RESEARCH  (the #1 differentiator: ≥2-3 heterogeneous sources)        │
│ • Track A outside view: historical base rates, reference classes,    │
│   similar resolved questions, market odds where visible              │
│ • Track B inside view: agentic iterative news search (native         │
│   provider search tools + AskNews + Perplexity), gap-driven re-query │
│ • Specialized: FRED/Yahoo for econ; resolution-page screenshot;      │
│   relevance-filter + question-conditioned summaries before context   │
├──────────────────────────────────────────────────────────────────────┤
│ REASONING  (structured scratchpad on 2-3 frontier model families)    │
│ restate criteria literally → reference classes & base rate →         │
│ status quo → evidence rubric (strong/moderate/weak) → clashing       │
│ forces → premortem/blind-spot → hazard & scope check → 1%-granular   │
│ probability  |  numeric: percentiles incl. p2.5/p97.5 (or mixtures   │
│ / Monte-Carlo code) → PCHIP CDF + constraint standardizer            │
├──────────────────────────────────────────────────────────────────────┤
│ ENSEMBLE  (5-10 runs across families; median / trimmed GMO;          │
│ supervisor agent investigates high-divergence questions with         │
│ follow-up research; optional decorrelated fine-tuned member)         │
├──────────────────────────────────────────────────────────────────────┤
│ CALIBRATION & COHERENCE  (yes-bias correction → fitted Platt per     │
│ question type → hard cap [2%,98%] → MC renormalize+floor →           │
│ date-ladder & related-question consistency → format validation)      │
├──────────────────────────────────────────────────────────────────────┤
│ SUBMIT + LOG  (forecast + reasoning comment via API; full config/    │
│ cost/score logging for the eval loop)                                │
├──────────────────────────────────────────────────────────────────────┤
│ EVAL LOOP  (custom resolvable questions; MiniBench; CP regression    │
│ tests; consistency-check CI; Brier decomposition per type;           │
│ 300+ questions before believing any change)                          │
└──────────────────────────────────────────────────────────────────────┘
```

**Where to spend marginal effort, ranked by measured ROI:**
1. Reliability + question-type coverage (table stakes; cheap; bugs are the most expensive failure).
2. Research breadth/agentic depth (#1 winner predictor).
3. Ensembling across model families (+1,799 pts; most validated).
4. Numeric/MC machinery (biggest per-type deficit = biggest headroom).
5. Calibration layer fitted on own history (+0.016 Brier equivalent; capping r=+0.48).
6. Supervisor/divergence reconciliation (~0.005 Brier).
7. Custom-question eval harness (+2,216 pts — technically rank #1 in the survey, but it pays through everything above).
8. Fine-tuned decorrelated ensemble member (frontier-level, unproven in tournaments, real research support).

**Strategic posture:** maximize discrimination subject to calibration — be boldly extreme only when the research justifies it, capped for tail safety; forecast every question your pipeline handles at field-average or better; treat the community/market signal as evidence, never as the target.

---

<a name="12-resources"></a>
## 12. Key Resources

**Infrastructure**
- Bot template: https://github.com/Metaculus/metac-bot-template · forecasting-tools: https://github.com/Metaculus/forecasting-tools · API: https://www.metaculus.com/api/
- FutureEval hub: https://www.metaculus.com/futureeval/ · Resources notebook: https://www.metaculus.com/notebooks/38928/ · Scores FAQ: https://www.metaculus.com/help/scores-faq/
- Summer 2026 tournament: https://forum.effectivealtruism.org/posts/ZfLAN557rGWACKtmc/announcing-metaculus-summer-2026-futureeval-bot-tournament

**Open-source bots**
- Panshul42 (Q2'25 winner): https://github.com/Panshul42/Forecasting_Bot_Q2 · nostreambot: https://github.com/No-Stream/metaculus-bot · Halawi pipeline: https://github.com/dannyallover/llm_forecasting · more: resources notebook §open-source-bots

**Must-read papers** (details in notes 01–02)
- Halawi et al. 2402.18563 (NeurIPS 2024) · ForecastBench 2409.19839 (ICLR 2025) · AIA Forecaster 2511.07678 · Consistency Checks 2412.18544 (ICLR 2025) · Pitfalls 2506.00723 · Silicon Crowd 2402.19379 (Sci. Adv.) · Prompt-engineering null result 2506.01578 · RLVR forecasting 2505.17989 · Future-as-Label 2601.06336 · BTF-2 2604.26106

**Tournament analyses & surveys** (details in notes 04–05)
- Q3'24 28784 · Q4'24 35291 · Q1'25 38673 · Q2'25 40456 · Fall'25 survey 43337 · Fall'25 advice 43357 · "What are other bots doing" 43497 · Calibration-adjustment analysis 43356 · Prompt-optimization 38421/43362 (all at metaculus.com/notebooks/)

**Forecasting science** (details in notes 03)
- Tetlock & Gardner *Superforecasting* · Mellers et al. 2014/2015 · Chang et al. 2016 (training) · Baron et al. 2014 (extremizing) · Satopää et al. 2014 · Jose et al. 2014 (trimming) · Sevilla's aggregation guides (EA Forum) · Samotsvety methodology posts · Sempere on time decay (LessWrong)
