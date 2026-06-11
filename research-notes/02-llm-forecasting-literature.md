# LLM-Based Forecasting Literature Survey (beyond Halawi et al. 2402.18563)
*Compiled June 2026. All Brier scores: lower = better, unless stated as "Brier Index" (higher = better, defined as (1 − √Brier) × 100%).*

---

## 1. ForecastBench (Karger, Bastani, Yueh-Han, Jacobs, Halawi, Zhang, Tetlock — arXiv 2409.19839, ICLR 2025)

**URLs:** https://arxiv.org/abs/2409.19839 · https://www.forecastbench.org · FRI Substack analysis: https://forecastingresearch.substack.com/p/ai-llm-forecasting-model-forecastbench-benchmark

**Benchmark design**
- Dynamic, contamination-free benchmark: ~1,000 automatically generated questions per round, **only about future events with unknown answers at submission time** (no pastcasting, no leakage by construction). Updated continually (v5 of paper Feb 2025).
- Two question classes: **market questions** (sourced from Metaculus, Manifold, Polymarket, RFI) and **dataset questions** (auto-generated from time series: ACLED, DBnomics, FRED, Wikipedia, Yahoo Finance, etc.).
- Two tracks: **Baseline leaderboard** (LLM alone, no tools) and **Tournament leaderboard** (any scaffolding/fine-tuning/ensembling allowed).
- Human comparison: superforecasters + general public answered a 200-question subset.

**ICLR 2025 paper numbers** (human-comparison leaderboard; dataset N=2,737, market N=912 for LLM board):
- **Superforecaster median: 0.093 overall** (dataset 0.135, market overall 0.051). **Public median: 0.107.** Best LLM (Claude-3.5-Sonnet, scratchpad + market "freeze values"): **0.111**; GPT-4 zero-shot+freeze 0.113. Superforecasters > best LLM at **p < 0.001**.
- Top of LLM-only board: GPT-4-Turbo-2024-04-09 scratchpad+freeze **0.113** (dataset 0.175 / market 0.051); Claude-3.5-Sonnet scratchpad+freeze 0.113; GPT-4o 0.120; Gemini-1.5-Pro 0.121. "Always 0.5" baseline: 0.208; random uniform: 0.292.
- Scaffolding effects in-paper: **scratchpad (CoT) prompts and giving current market values ("freeze values") helped most**; adding news retrieval helped only marginally and sometimes hurt (e.g., GPT-4o scratchpad+news 0.123 vs 0.120 without). An "Imputed Forecaster" (just echo market price, 0.5 on dataset Qs) scored 0.136 overall with **market score 0.021 — better than any LLM on market questions**, exposing how much of LLM "skill" on market Qs is price-copying.

**Latest standings (as of mid-2026, FRI Substack + leaderboard)**
- Brier Index scale: **superforecasters 70.6%**, top systems **67.9%** — a tie between **CassiAI's `ensemble_2_crowdadj`** (tournament track) and **xAI's Grok 4.20 (Preview)**. Gap ≈ 2.7 pts ≈ ~1 year of progress.
- Baseline track: best model **GPT-4.5, Brier 0.101 vs superforecasters 0.081** (~25% worse); LLMs now **beat the public median** (a flip from 2024–25).
- Trend: SOTA improves **~0.016 Brier points/year** (0.020/yr on dataset Qs); linear extrapolation puts **LLM–superforecaster parity at November 2026 (95% CI Dec 2025 – Jan 2028)**; dataset-question parity ~June 2026. o3 leads on dataset questions; GPT-4.5 on market questions.
- Key documented weakness: on market questions LLMs' forecasts correlate **0.994** with the market price they're shown — they anchor/copy rather than add information.

**Lessons for a Metaculus bot**
- Use CoT/scratchpad + inject the current community/market forecast as an anchor — single biggest cheap win on FB; but recognize you won't beat the crowd by copying it (Metaculus peer scoring punishes pure copying).
- Dataset-style (quantitative time-series) questions are where LLMs lag humans most — invest in tooling for base-rate/time-series extrapolation.
- Track ForecastBench's baseline board for model selection; as of latest board GPT-4.5-class/Grok-4.x/o3-class models top it.

---

## 2. The Schoenegger / Tetlock cluster

### 2a. GPT-4 in a real tournament (Schoenegger & Park, arXiv 2310.13014)
- GPT-4 entered a 3-month Metaculus tournament (Jul–Oct 2023, 843 human participants). **GPT-4 was significantly worse than the crowd median and not significantly better than always-50%.** The canonical negative baseline result; explains why naive single-shot bots fail.

### 2b. Wisdom of the Silicon Crowd (Schoenegger, Tuminauskaite, Park, Tetlock — arXiv 2402.19379; *Science Advances* 10:eadp1528, Nov 2024)
- **Study 1:** 12-model ensemble (GPT-4, Claude 2, Bard/PaLM-2, Falcon-180B, etc.), 31 binary questions, vs 925 human forecasters. **LLM crowd Brier 0.20 (SD 0.12) vs human crowd 0.19 (SD 0.19)**, t(60)=0.19, p=0.850; equivalence within medium-effect bounds. Individual models: GPT-4 0.15, GPT-4+Bing 0.16, Bard 0.19, Claude 2 0.21, …, Coral/Command 0.38.
- **Study 2:** Showing GPT-4/Claude 2 the human-crowd median improved them **17.6% (GPT-4: 0.17→0.14, p=0.003)** and **31.8% (Claude 2: 0.22→0.15, p<0.001)** — but simply **averaging the machine and human forecasts beat the updated machine forecasts**.
- Documented biases: **acquiescence effect** — mean prediction 57.35% (SD 20.93), significantly >50% (t(1006)=86.20, p<0.001) although only ~45% of questions resolved Yes; **round-number clumping** (5%/10% multiples).
- *Bot lesson:* ensemble across **different model families** (diversity is the point); debias the systematic Yes-lean; blend with the community prediction rather than replacing it.

### 2c. AI-Augmented Predictions (Schoenegger, Park, Karger, Trott, Tetlock — arXiv 2402.07862)
- N=991 humans, 6 questions, GPT-4-Turbo assistant. Access to an LLM assistant improved human accuracy **24–28%** vs control — and even a deliberately **overconfident, base-rate-neglecting assistant helped (+28%)**, implying gains come from structured reasoning prompts, not advice quality. Less directly bot-relevant; relevant to human-in-the-loop review of bot outputs.

### 2d. Prompt engineering doesn't save you (Schoenegger, Jones, Tetlock, Mellers — arXiv 2506.01578)
- **38 prompts** × Claude-3.5 Sonnet/Haiku, GPT-4o, Llama 3.1-405B (Study 1); compound + external prompts incl. o1/o1-mini (Study 2). **Nearly all prompt tweaks gave negligible gains**; small benefit from **base-rate references**; **"think Bayesianly" prompts significantly *hurt* accuracy**.
- *Bot lesson:* don't burn cycles on prompt micro-optimization; gains come from model choice, retrieval, aggregation, and calibration layers. Do include base rates/reference classes; avoid explicit Bayesian-updating scripts.

---

## 3. Training / fine-tuning approaches

### 3a. LLMs Can Teach Themselves to Better Predict the Future (Turtel, Franklin, Schoenegger — arXiv 2502.05253)
- Outcome-driven **self-play + DPO**: generate diverse reasoning traces + probabilities for post-cutoff questions, rank trace pairs by distance to realized outcomes, DPO-fine-tune. No human-curated rationales.
- **+7–10% accuracy** for Phi-4-14B and DeepSeek-R1-14B vs base and vs randomized-label DPO control, reaching **GPT-4o-level** forecast accuracy with 14B models.

### 3b. Outcome-based RL to Predict the Future (Turtel, Franklin, Skotheim, Hewitt, Schoenegger — arXiv 2505.17989)
- RLVR adapted to noisy/delayed binary rewards. Data: **~10k resolved Polymarket questions + 100k synthetic**; base model DeepSeek-R1-Distill-Qwen-14B; ReMax (+ GRPO variants) with **guardrails: token-length limits, gibberish filter, early stop**.
- Results: best model (ReMax, 7-run ensemble) **soft-Brier 0.190 vs o1's 0.202** (significant), **ECE 0.062 vs o1's 0.093**; simulated Polymarket trading **>10% ROI** (~$52 vs o1's $39; ~20% ROI concentrated in 40–60% "uncertain" markets).
- Failure modes during RL: vanilla GRPO produced **39.3% of predictions in the extreme 0–10%/90–100% bins** (overconfidence collapse) and language-switching gibberish; guardrails cut extremes to **13.1%**.
- *Bot lesson:* a small RL-tuned open model can beat frontier reasoning models on calibration; if you fine-tune, expect degenerate overconfidence without explicit regularization. Hidden alpha is in mid-probability questions.

### 3c. Mantic × Thinking Machines: Training LLMs to Predict World Events (blog, late 2025/2026 — https://thinkingmachines.ai/news/training-llms-to-predict-world-events/)
- RL (policy gradient w/ GRPO-style advantage normalization, via Tinker) on **gpt-oss-120b**, ~**10,000 binary questions** (Aug 2024–Dec 2025), eval on Metaculus Q2-2025 AIB questions. Internal score improved **38.6 → 45.8**, reaching **parity with Gemini 3 Pro**.
- In their production ensemble (with Grok 4, Gemini 3 Pro, GPT-5), the fine-tune earned **40–56% weight** and was "second least replaceable" after Grok 4 — value comes from **decorrelation** with frontier models. Forecast-specific context + tools during training amplified gains.
- Mantic placed **8th of 549** in the human **Metaculus Cup Fall 2025** — first bot in the top 10 of that series (caveats: 60 questions, mostly amateur field, coverage scoring favors always-on bots). Coverage: Time, https://time.com/7318577/ai-model-forecasting-predict-future-metaculus/

### 3d. Earlier event-forecasting training line
- **Autocast** (Zou et al., NeurIPS 2022 D&B, arXiv 2206.15474): thousands of tournament questions (≈3,962 T/F, 929 MC, 812 numeric) + dated news corpus; best ML system ~**65% accuracy on binary vs 92% human crowd**. Established retrieval helps and scale helps, but big human gap.
- **AutoCast++** (Yan et al., ICLR 2024, arXiv 2310.01880): zero-shot ranking-based news retrieval + recency re-ranking + human-aligned loss → **+48% on MCQ** over Autocast baseline. Lesson: **filtering/ranking news by question relevance and recency matters more than volume**.
- **Position paper:** "Advancing Event Forecasting through Massive Training of LLMs" (Lee, Yang, Kwak, Siegel — arXiv 2507.19477): argues for large-scale superforecaster-LLM training; identifies noisy/sparse rewards, knowledge cutoff, and simplistic reward structure as core obstacles; proposes Bayesian-network event simulation, counterfactual events, auxiliary rewards, aggressive market/web data collection.

---

## 4. Agentic / retrieval systems and their benchmarks

### 4a. AIA Forecaster (Bridgewater AIA Labs — arXiv 2511.07678, Nov 2025) — **the strongest published "parity" claim**
- Architecture: (1) **agentic iterative news search** per agent; (2) **~10 independent agent forecasts**; (3) **supervisor agent** that reconciles disagreements with targeted follow-up search (not naive averaging); (4) **Platt-scaling calibration layer** to counteract LLM hedging-toward-0.5.
- Results on ForecastBench question sets: **FB-7-21: 0.1076 vs superforecasters 0.1110; FB-8-14: 0.1099 vs 0.1152 — statistically indistinguishable from superforecasters** ("first work that verifiably achieves expert-level forecasting at scale"). On liquid markets: 0.1258 vs market 0.1106 (trails, but diversifying).
- Ablations: agentic search 0.1140 vs 0.1230 without; **providing market prices erodes ~42% of search's edge** (explains why prior work saw little search benefit); **Platt scaling alone 0.1140 → 0.1076** (worth as much as the supervisor agent); single runs are unstable — averaging ~10 runs is "mandatory"; mean ≈ median ≈ trimmed mean.
- Honest leakage accounting: ~**1.65%** of search results leaked post-cutoff info; robustness checks bound the effect at ≤0.6% of performance.
- *Bot lesson:* the winning 2025-era recipe = agentic search → K independent forecasts → LLM reconciliation → **statistical extremization/calibration fitted on resolved questions**. Each layer is worth ~0.005–0.01 Brier.

### 4b. FutureSearch: Bench to the Future (arXiv 2506.21558) and BTF-2 (arXiv 2604.26106)
- **BTF:** "pastcasting" benchmark — hundreds of resolved questions, each with a **frozen offline corpus of tens of thousands of web pages** (RetroSearch), so agents research realistically without leakage; validated that pastcast scores track live-forecast scores. Tested Claude 4-era models, agent vs CoT scaffolds.
- **BTF-2 (2026):** **1,417 questions, frozen 15M-document corpus**; resolution sensitive to **0.004 Brier** differences; separates research skill from judgment skill. Their composite forecaster is **0.011 Brier better than any single frontier agent**. Current public leaderboard (evals.futuresearch.ai, Apr 20 2026): **FutureSearch Agent 0.119 < Opus 4.6 0.130 < Gemini 3.1 Pro 0.141 < GPT-5.4 0.152 < Grok 4.20 Beta 0.165**.
- Qualitative findings: the best forecaster differs mainly via **pre-mortem analysis of blind spots and black-swan consideration**; expert-identified frontier-agent failures: judging **leaders' incentives**, **follow-through on stated plans**, and **institutional processes**.
- Related: **Deep Research Bench** (arXiv 2506.06287): 89 human-verified web-research tasks; documents agent hallucination, tool-use failures, and forgetting in long traces — the research half of forecasting.
- *Bot lesson:* add an explicit pre-mortem/black-swan step to the reasoning scaffold; treat "who has power and what are their incentives" questions as a known weak spot deserving special prompting or human review.

### 4c. MIRAI (Ye et al., arXiv 2407.01231)
- Agentic benchmark for international-event forecasting on cleaned GDELT: 991,759 event records, 59,161 events, 296,630 news articles; 705 test QA pairs with code-based tool APIs. Shows LLM agents are weak at long-horizon relational event prediction and at choosing the right tools/APIs. Niche but useful if your bot covers geopolitics via structured event databases.

---

## 5. Aggregation science applied to LLM crowds

- **Model-diversity ensembles:** Silicon Crowd (12 heterogeneous models) reached human-crowd parity (§2b). Mantic's production system explicitly weights members by decorrelation; their RL'd open model earned 40–56% ensemble weight despite frontier peers (§3c).
- **Sample ensembles (one model, many runs):** AIA found single runs unstable; **mean of ~10 independent agent runs is necessary**, with mean ≈ median ≈ trimmed mean (§4a). Halawi-adjacent benchmark comparisons report median/trimmed-mean Brier 0.197 vs geometric mean 0.194 — aggregation-function choice is second-order; *doing* aggregation is first-order.
- **Prompt-diversity ensembles:** DIPPER (arXiv 2412.15238, NeurIPS workshop) — injecting diversity via different prompts to one model: 3-prompt ensemble ≈ +10 accuracy points on reasoning tasks; "Calibrating LMs via Augmented Prompt Ensembles" shows prompt ensembles improve calibration. Evidence in forecasting specifically is thinner — most Metaculus bot-makers' repeated-call median is the deployed version of this.
- **Extremization:** the human-forecasting extremization literature now has an LLM analogue: AIA's **Platt scaling** (fit on resolved questions) is functionally extremization to counter hedging — worth **0.0064 Brier** (0.1140→0.1076). Conversely, RL fine-tuning *over*-extremizes without guardrails (§3b). ForecastBench's own "LLM crowd" entry (GPT-4o + Claude-3.5 + Gemini-1.5 with news) scored only 0.137 — naive cross-model averaging of weak forecasts does not beat the best single configuration.
- **Human anchor:** showing the model the human/market median, then averaging with it, beat either alone (§2b). On Metaculus, the community prediction is visible after some delay — use it where allowed.
- *Bot lesson:* aggregate K≥10 runs per question (median or mean), across ≥2–3 decorrelated model families if budget allows; then apply a calibration/extremization map **fitted on your own resolved-question history**, with separate maps per question type.

---

## 6. Documented LLM forecasting failure modes

1. **Miscalibration in both directions.** Hedging toward 0.5 in frontier instruction-tuned models (AIA: Platt scaling fixes ~0.006 Brier); overconfidence in older models and in RL-tuned models without guardrails (39.3% of predictions at extremes, §3b); persistent overconfidence at high-probability bins in calibration plots (2507.04562).
2. **Acquiescence / Yes-bias.** Mean prediction 57.35% vs 45% base rate of Yes resolutions (Silicon Crowd, §2b). Negating question wording shifts probabilities (same paper's framing tests; Paleka's NEG checks).
3. **Logical inconsistency / negation-complement violations.** *Consistency Checks for Language Model Forecasters* (Paleka et al., arXiv 2412.18544, ICLR 2025): arbitrage-based metric over checks (negation, paraphrase, mutually-exclusive partitions, conditional/Bayes coherence, monotonicity); frontier forecasters violate these enough to be arbitraged; **instantaneous consistency violation metrics correlate with future ground-truth Brier** — usable as a free, immediate eval. They release a 2028-resolving benchmark. *Bot use:* run negation/partition consistency checks in CI; enforce sum-to-one across MC options and monotonicity across date-laddered questions.
4. **Round-number anchoring.** Predictions clump at multiples of 5/10% (Silicon Crowd); titotal's 539 experiments show the same plus **massive run-to-run variance** (same question: 30/30/40/40/60%).
5. **Insensitivity to question parameters.** 539 bot gave flat or *decreasing* probabilities for Iran-nuclear questions as horizon extended decades; probabilities across exclusive energy-mix outcomes didn't sum to 1; 35% on a near-certain (95% market) Jimmy Carter question (titotal, https://titotal.substack.com/p/is-superhuman-ai-forecasting-bs-some). Cautionary tale: CAIS's "superhuman" claim rested on 177 selected questions vs a 20-person crowd — *Contra papers claiming superhuman AI forecasting* (LessWrong) is the standard rebuttal.
6. **Knowledge-cutoff confabulation & leakage.** *Pitfalls in Evaluating LM Forecasters* (Paleka, Goel, Geiping, Tramèr — arXiv 2506.00723): ≥3.8% of Halawi et al.'s backtest resolved before forecast date; ≥10% trivially answerable in Tao et al.; >90% of news-generated questions in some datasets are degenerate; GPT-4o "Oct 2023 cutoff" demonstrably knows Nov 15 2023 events; date-restricted search still leaks via re-dated articles and future-trained rankers; benchmark gaming via correlated bets on latent variables (e.g., conditioning everything on one election outcome). Recommendations: use release dates not stated cutoffs, validate every resolution, report Brier + calibration + log score, measure edge *vs the human crowd*. *Simulated Ignorance Fails* (arXiv 2601.13717, Jan 2026): across 477 questions × 9 models, prompting a model to "pretend it's before the cutoff" leaves a **52% performance gap vs true ignorance** — prompts cannot rewind knowledge; reasoning models are *worse* at pretending. Implication: never trust your own backtests on pre-cutoff questions.
7. **Market price parroting.** 0.994 correlation with shown market prices on ForecastBench market questions (§1) — looks accurate, adds zero alpha, and on Metaculus peer scoring earns ~0 points.
8. **Numeric/continuous distribution construction.** Metaculus AIB: bots' head-to-head deficit vs Pros was much worse on **numeric (−23.2) and multiple-choice (−32.9) than binary (−14.8)** questions (Q2 2025) — building CDFs is a distinct unsolved skill; invest in explicit distribution-construction code (mixture of base-rate + trend extrapolation, clamped tails) rather than raw LLM quantiles.

---

## 7. Have frontier models reached superforecaster level? (status as of June 2026)

- **ForecastBench:** Not yet, but close. Brier Index: superforecasters **70.6%** vs best AI **67.9%** (CassiAI ensemble & Grok 4.20 Preview); baseline-track GPT-4.5 Brier 0.101 vs SF 0.081. Trend-extrapolated parity: **Nov 2026 (95% CI Dec 2025–Jan 2028)**; dataset-question parity ~June 2026.
- **Bridgewater AIA Forecaster (arXiv 2511.07678):** with full agentic scaffold + calibration, **statistically indistinguishable from superforecasters on two ForecastBench rounds (0.1076 vs 0.1110; 0.1099 vs 0.1152)** — parity achieved by a *system*, not a bare model — but still trails liquid market prices (0.1258 vs 0.1106).
- **Metaculus AIB/FutureEval (the venue you'd compete in):** Pros have won **every season** through Fall 2025. Head-to-head (bot-team minus pro-team): Q3'24 −11.3, Q4'24 −8.9 (p=0.079), Q1'25 −17.7 (p=0.001), Q2'25 −20.03 (p=0.00001); Fall 2025: all 10 Pros individually beat every bot. Projections from Metaculus: bots pass the **community** ~April 2026, **Pros mid-2027**. Q2'25 survey findings: **model quality > scaffolding**; biggest point-gains associated with forecast aggregation (+1,799 coverage-adjusted points), manual review (+1,041), custom test-question creation (+2,216); best models: o3 (Q1–Q2 2025); top bot-makers were hobbyists/students (e.g., Panshul42, pgodzinai); most successful bots = repeated LLM calls + median, almost none fine-tuned. Fall 2025 note: a GPT-4o bot with strong scaffolding beat the o1 template bot — scaffolding *can* trump raw model within limits. Summer 2026 tournament: $50k, 300–500 questions, started May 18 2026, free Anthropic/Google/OpenAI/AskNews credits.
- **BTF-2 (Apr 2026):** best single frontier agent (Opus 4.6, 0.130) still beaten by FutureSearch's ensemble (0.119); GPT-5.4 agent only 0.152 — frontier-model raw forecasting still has headroom.
- **Independent check (arXiv 2507.04562, Lu):** on 464 Metaculus questions, frontier models beat the general crowd but "significantly underperform" expert forecasters.

**Synthesis:** bare frontier models ≈ skilled-public level, below superforecasters; *engineered systems* (agentic search + ensembling + fitted calibration ± outcome-trained components) reached statistical superforecaster-parity on ForecastBench in late 2025, but no bot has beaten Metaculus Pros in a live tournament as of the Fall 2025 season results.

---

## Consolidated playbook for a Metaculus tournament bot (evidence-backed)

1. **Model choice first** (Metaculus surveys; FB baseline board): use the strongest reasoning model available (o3/GPT-5.x, Opus 4.x, Grok 4.x, Gemini 3); gains from model upgrades exceed prompt tweaks.
2. **Agentic, iterative news search** — not one-shot retrieval (AIA: 0.1230→0.1140; AutoCast++: relevance+recency re-ranking +48% MCQ).
3. **K≈10 independent runs per question, aggregate by median/mean** (AIA; Metaculus survey; aggregation +1,799 pts), ideally across 2–3 decorrelated model families (Silicon Crowd; Mantic ensemble-weight evidence).
4. **Reconcile, don't just average:** a supervisor pass that investigates disagreements adds ~0.005 Brier (AIA).
5. **Fit a calibration/extremization map (Platt scaling) on your resolved-question history** — worth ~0.006 Brier; correct the documented hedging-to-0.5 and acquiescence (+7pt Yes-lean) biases.
6. **Include base rates/reference classes in the prompt; skip "Bayesian reasoning" instructions** (2506.01578). Add a **pre-mortem / black-swan step** (BTF-2).
7. **Run consistency checks** (negation, MC sum-to-one, date-ladder monotonicity) as automated QA — violations predict bad Brier (2412.18544).
8. **Special-case numeric/MC questions** with programmatic distribution construction — bots' worst gap vs Pros (−23.2/−32.9 head-to-head).
9. **Use the community prediction when visible** (anchor + average beats either alone — 2402.19379), but don't parrot it (0.994-correlation trap; peer scores reward deviation when right).
10. **If fine-tuning:** outcome-based DPO/RL on resolved market questions works at 14B–120B scale (+7–10%; ECE 0.062 vs o1's 0.093; Mantic 38.6→45.8) but needs anti-overconfidence guardrails; its biggest payoff is as a *decorrelated ensemble member*.
11. **Distrust your own backtests:** no simulated-ignorance evals (52% gap, 2601.13717); validate question resolution dates, use model release dates, and prefer prospective testing or frozen-corpus pastcasting (BTF) (2506.00723).
12. **Manual review of forecasts before submission** correlated with +1,041 points (Metaculus Q2'25 survey) — keep a human spot-check loop early on.

Sources: [arXiv 2409.19839](https://arxiv.org/abs/2409.19839) · [forecastbench.org leaderboards](https://www.forecastbench.org/assets/iclr2025/leaderboards/leaderboard_overall.html) · [FRI Substack: How well can LLMs predict the future?](https://forecastingresearch.substack.com/p/ai-llm-forecasting-model-forecastbench-benchmark) · [FRI Substack: Brier Index](https://forecastingresearch.substack.com/p/introducing-the-brier-index) · [arXiv 2402.19379](https://arxiv.org/abs/2402.19379) / [Sci. Adv.](https://www.science.org/doi/10.1126/sciadv.adp1528) · [arXiv 2402.07862](https://arxiv.org/abs/2402.07862) · [arXiv 2310.13014](https://arxiv.org/abs/2310.13014) · [arXiv 2506.01578](https://arxiv.org/abs/2506.01578) · [arXiv 2502.05253](https://arxiv.org/abs/2502.05253) · [arXiv 2505.17989](https://arxiv.org/abs/2505.17989) · [Thinking Machines × Mantic](https://thinkingmachines.ai/news/training-llms-to-predict-world-events/) · [Time on Mantic](https://time.com/7318577/ai-model-forecasting-predict-future-metaculus/) · [arXiv 2206.15474](https://arxiv.org/abs/2206.15474) · [arXiv 2310.01880](https://arxiv.org/abs/2310.01880) · [arXiv 2507.19477](https://arxiv.org/abs/2507.19477) · [arXiv 2511.07678](https://arxiv.org/html/2511.07678v1) · [arXiv 2506.21558](https://arxiv.org/abs/2506.21558) · [arXiv 2604.26106](https://arxiv.org/abs/2604.26106) · [evals.futuresearch.ai](https://evals.futuresearch.ai/) · [arXiv 2506.06287](https://arxiv.org/abs/2506.06287) · [arXiv 2407.01231](https://arxiv.org/abs/2407.01231) · [arXiv 2412.18544](https://arxiv.org/abs/2412.18544) · [arXiv 2506.00723](https://arxiv.org/html/2506.00723v1) · [arXiv 2601.13717](https://arxiv.org/abs/2601.13717) · [arXiv 2412.15238](https://arxiv.org/html/2412.15238v1) · [titotal on 539](https://titotal.substack.com/p/is-superhuman-ai-forecasting-bs-some) · [Metaculus Q1'25 results](https://www.lesswrong.com/posts/rDy5z8ZEtMrEGnfBd/q1-ai-benchmark-results-pro-forecasters-crush-bots) · [Metaculus Q2'25 results](https://forum.effectivealtruism.org/posts/F2stjK9wHSy3HPEC9/q2-ai-benchmark-results-pros-maintain-clear-lead) · [Q4'24 results](https://www.lesswrong.com/posts/P8YwCvHoF2FHQoHjF/metaculus-q4-ai-benchmarking-bots-are-closing-the-gap) · [Summer 2026 FutureEval announcement](https://forum.effectivealtruism.org/posts/ZfLAN557rGWACKtmc/announcing-metaculus-summer-2026-futureeval-bot-tournament) · [arXiv 2507.04562](https://arxiv.org/abs/2507.04562)
