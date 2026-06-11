# Deep-Read Notes: "Approaching Human-Level Forecasting with Language Models"

**Authors:** Danny Halawi*, Fred Zhang*, Chen Yueh-Han*, Jacob Steinhardt (UC Berkeley; * joint authorship)
**arXiv:** 2402.18563 (submitted Feb 28, 2024, v1 only) — published at **NeurIPS 2024**
**Sources used:**
- Abstract page: https://arxiv.org/abs/2402.18563
- Full HTML: https://arxiv.org/html/2402.18563 (read in full, including all appendices)
- PDF: https://arxiv.org/pdf/2402.18563 (used to verify exact numbers)
- Code: https://github.com/dannyallover/llm_forecasting — datasets on HuggingFace: `YuehHanChen/forecasting` (cleaned) and `YuehHanChen/forecasting_raw` (raw)
- NeurIPS proceedings: https://proceedings.neurips.cc/paper_files/paper/2024/hash/5a5acfd0876c940d81619c1dc60e7748-Abstract-Conference.html

**One-line summary:** A retrieval-augmented GPT-4 pipeline (LM-generated news queries → relevance-rated/summarized articles → 3 scratchpad prompts on base GPT-4 + 3 samples from a self-supervised fine-tuned GPT-4 → trimmed-mean ensemble) scores Brier .179 vs. human crowd .149 on 914 post-knowledge-cutoff binary questions, beats the crowd in selective regimes (crowd uncertain, early retrieval, many articles), and improves the crowd when averaged with it (.149 → .146).

---

## 1. Dataset and Evaluation Setup

### Dataset
- Sourced from **5 platforms**: Metaculus, Good Judgment Open (GJOpen), INFER (CSET), Polymarket, Manifold (via APIs + scraping).
- **Raw:** 48,754 questions, 7,174,607 user forecasts, spanning 2015–2024. By type: 33,664 binary, 9,725 multiple-choice, 4,019 numerical, 1,346 other.
- **Curation:** focus on binary questions (MC questions split into binaries); GPT-3.5 prompt-based screening of ill-defined/personal questions (few-shot flagging prompt) plus manual review; removed low-volume Manifold/Polymarket questions. GPT-3.5-Turbo assigns each question 1 of 11 categories.
- **Curated splits (5,516 binary questions):** train 3,762 / validation 840 / test 914.
  - Per-platform test counts: Metaculus 275, GJOpen 38, INFER 4, Polymarket 300, Manifold 297.
- **Leakage control:** all **test** questions *opened on/after June 1, 2023* (after the knowledge cutoff of every model used); all train/val questions *resolved before* June 1, 2023; questions straddling the date discarded. They additionally verified no leakage from post-training: GPT-4-1106-Preview and GPT-3.5-Turbo-1106 fail/hallucinate on 20 manually checked events from Jun–Sep 2023 but answer 20 pre-cutoff events correctly.

### Retrieval schedule (simulated forecast dates)
- **n = 5 retrieval dates per question**, geometrically spaced between begin and close date:
  `retrieval_date_k = date_begin + (date_close − date_begin − 1)^(k/n)`
- Rationale: activity concentrates early; close dates are often far after actual resolution. Dates after early resolution are dropped (86% of dates retained on average). Crucially, dates are **not** anchored to the resolve date, which would leak the outcome (early resolution correlates with "Yes" for "will X happen by date" questions).
- Average question window ≈ 70 days; average time-to-resolution ≈ 42 days.

### Metric
- **Brier score** (f − o)²; unskilled 0.5-forecast = .250. Final score = mean over retrieval dates per question, then mean over questions. Also report accuracy and **RMS calibration error**. They caution that reported standard errors assume i.i.d. data and likely underestimate true error (time-series data).

### Baseline ("models are not naturally good at forecasting")
- 14 instruction-tuned LMs evaluated with no retrieval, zero-shot and scratchpad prompts (5 zero-shot and 4 scratchpad candidates; best picked on 200 validation questions). Zero-shot prompt forces an answer: "You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES… If you can't answer, pick the base rate."
- Best baseline: **GPT-4-1106-Preview .208** (zero-shot) / .209 (scratchpad). Others: GPT-4-0613 .219/.222, Claude-2.1 .220/.215, GPT-3.5-Turbo .237/.257, Llama-2-13B .226/.268, Mixtral-8x7B .238/.238, Gemini-Pro .243/.230, Llama-2-7B .353 zero-shot. Most ≈ random (.25); **all trail the crowd's .149 badly**. Zero-shot base models are also poorly calibrated.
- Models would sometimes refuse forecasting questions ("I don't know"), attributed to safety training; scratchpad prompt selection used refusal counts as tiebreaker (winner: 88 refusals vs. 106/93/94).

---

## 2. End-to-End System Architecture

### 2.1 Retrieval pipeline (4 steps)

**Step 1 — Search query generation:**
- **2 prompts**, both applied, union of results (plus the raw question itself as a query):
  1. *Straightforward query expansion*: given question + background, "Generate brief search queries (up to {max_words} words each)… You must generate this exact amount of queries: {num_keywords}" with a Thoughts → Search Queries (semicolon-separated) structure.
  2. *Sub-question decomposition*: "Start off by writing down sub-questions. Then use your sub-questions to help steer the search queries you produce." (Motivation: direct expansion misses sub-considerations — e.g., for an election question it searches polls only, while decomposition covers campaign finance, economic indicators, geopolitics.)
- Model: **GPT-4-1106-Preview at temperature 0**, **6 queries per prompt** (so ~12–13 queries total after union).
- Prompt selection used proxy metrics over 6 candidates (not end-to-end Brier): average relevance rating of retrieved articles (top-2 prompts: 3.08 and 3.09 vs. <3.04 for the rest) and average rating among articles rated ≥4 (4.37, 4.38). Top 2 prompts overlap little, so both are kept.

**Step 2 — News retrieval:**
- 5 APIs evaluated (must accept retrieval date ranges to prevent leakage): Google News (via Python `gnews`), NewsCatcher, Newsdata.io, Aylien, NewsAPI.org. GPT-4 relevance-scored retrievals on 24 unresolved questions; total scores: **Google News 39, NewsCatcher 35**, Aylien 30.5, NewsAPI.org 23.5, Newsdata.io 16.5 → **NewsCatcher + Google News selected**.
- **Top 10 articles per API per query**, default API ranking, English only, published within [question begin date, retrieval date].
- If the question background contains URLs, the system scrapes them (whitelist of news sites with timestamps only; articles with missing/out-of-range publish dates discarded).

**Step 3 — Relevance filtering & re-ranking:**
- **GPT-3.5-Turbo at temperature 0** rates each article's relevance on a **1–6 scale** (1 irrelevant … 6 most relevant) given question, background, resolution criteria, and article; instructions include "output a score of 1" for paywall/JS error pages; numeric extracted after "Rating:".
- Cost saving: only **title + first 250 words** (~330 tokens of avg 1,087.6-token articles) shown → ~**70% cost reduction**. Validated against GPT-4 full-text gold labels on 169 articles from 15 validation questions: recall 0.73 / precision 0.65 (vs. Mixtral-8x7B-DPO at threshold 3: 0.70/0.63; embedding cosine similarity with text-embedding-3-large at threshold 0.48: 0.73/0.54). Title-only rejected because scraper failures make titles misleading.
- **Articles rated ≤ 3 are discarded** (keep ≥4); threshold not tuned. Survivors ranked by relevance score.

**Step 4 — Summarization:**
- **GPT-3.5-Turbo at temperature 0.2**; prompt: "condense it to no more than 100 words… do not remove any details that would be helpful for… the following forecasting question" (question + background in context). Articles exceeding context are truncated. Best of 5 candidate prompts by end-to-end validation Brier (best .193, runner-up .201).
- Final context: **top k = 15 article summaries, ordered by relevance** (swept jointly: ordering ∈ {relevance, recency} × k ∈ {5, 10, 15, 20, 30}; relevance + k=15 best at .177 validation Brier on GPT-4-1106-Preview; k=20 similar).

### 2.2 Reasoning pipeline

- All prompts begin with a fixed header: Question / Background / Resolution criteria / **Today's date: {retrieval date}** / **Question close date** / "We have retrieved the following information: {retrieved_info}" (the 15 summaries).
- **Base model: GPT-4-1106-Preview** with the **top 3 of 15 hand-crafted scratchpad prompts** (validation Briers: best .167, then .170, .174). (GPT-4-1106-Preview consistently beat Claude-2.1 by 0.01–0.03 Brier as reasoning model.)
- **Best scratchpad prompt — 7 steps (verbatim structure):**
  1. Rephrase and expand the question (maintain all original info) — comprehension + knowledge expansion.
  2. Provide a few reasons why the answer might be **no**, rating the strength of each, using world knowledge + retrieved info.
  3. Same for **yes**.
  4. "Aggregate your considerations. Think like a superforecaster (e.g. Nate Silver)." — weighting arguments by importance, not equally.
  5. Output an **initial probability**.
  6. "Evaluate whether your calculated probability is excessively confident or not confident enough. Also consider anything else… (e.g. **base rate of the event**)." — explicit calibration/base-rate step.
  7. Output the **final prediction** (number between 0 and 1, delimited with asterisks `*0.xx*`).
- **Fine-tuned model: GPT-4-0613 (fine-tuned, Section 3 below)**, prompted with **only the basic-info header + summaries, no scratchpad instructions** (it has internalized which reasoning to apply), **sampled 3 times at temperature T = 0.5**.

### 2.3 Aggregation / ensembling

- **6 forecasts total**: 3 from base GPT-4-1106-Preview (one per top-3 scratchpad prompt) + 3 samples from fine-tuned GPT-4-0613.
- **Trimmed mean** (custom variant for small N): uniform weights; halve the weight of the forecast **furthest from the median**; redistribute that half-weight uniformly to the others; output the weighted average.
- Validation comparison (Table 14): trimmed mean **.1649** < median .1651 < geometric mean .1655 < mean .1656 < no-ensemble baseline .1676 < USC (GPT-4 as LM aggregator over reasoning-prediction pairs) .1691. Human crowd on same set: .1600. So ensembling gains ~.003 Brier; USC *hurts*.

### 2.4 Hyperparameter sweep methodology
- Hyperparameters split into groups of 1–2, optimized **sequentially/greedily** on the **validation set** Brier (proxy metrics only for search-query prompts); parameters not yet swept are randomized per question to avoid confounding. During the sweep, retrieval date fixed at the **midpoint between open and resolve date** (crowd Brier at that point: .160 on validation).

---

## 3. Fine-Tuning Approach (self-supervised on outcomes)

- **Goal:** teach a model to produce reasonings that yield accurate predictions *without* prescriptive scratchpad instructions.
- **Data generation (on the train set, all retrieval-schedule dates):** 16 candidate forecasts per question:
  - 2 retrieval configurations per question (data augmentation: independently re-sample retrieval prompt, number of queries, articles per query → 2 different article sets);
  - × 4 scratchpad prompts per input (the optimal prompt + 3 sampled with probability inversely proportional to their validation Brier);
  - × 2 reasoning models (**GPT-4-1106-Preview and Claude-2.1** — Claude-2.1 is better on some questions; it is used only for data generation, never in the final system).
- **Selection filters (the key "self-supervision against outcomes + crowd" trick):**
  1. Keep only outputs whose Brier **beats the human crowd** on that question/date.
  2. Anti-overconfidence: **discard** pairs where the prediction deviates **> 0.15 from the crowd prediction**.
  3. The **target prediction is averaged with the crowd prediction** when constructing the output.
- Yield: **73,632 reasonings generated → 13,253 pass the filters**.
- **Fine-tuned GPT-4-0613 on the 6,000 most recent points for 2 epochs** (budget constraint; GPT-4-1106-Preview wasn't available for fine-tuning). Input = question/background/criteria/dates + summarized articles (no scratchpad instructions); target = reasoning + prediction.
- Suggested extension (Conclusion): **iterative self-supervision** — re-generate data with the fine-tuned model and repeat until training data is exhausted.

---

## 4. Calibration

- **No post-hoc adjustment is used.** The full system is "naturally well calibrated" on both validation and test; **binning and isotonic regression did not improve performance** and were dropped. No extremizing is applied.
- Test-set **RMS calibration error: ".42" vs. human crowd ".38"** as printed in the paper (note: given the system is described as well calibrated and the reliability plots, this is almost certainly a typesetting artifact for **0.042 vs 0.038**).
- Direction of miscalibration: **underconfidence at the extremes** — events predicted near 0 occur even less often than predicted, events near 1 occur more often than predicted. The model **hedges away from 0/1**, which the authors attribute to **safety training** (it rarely outputs very low probabilities even when evidence is decisive).
- Note the *implicit* calibration mechanisms: scratchpad step 6 (overconfidence check + base-rate consideration), the ±0.15-from-crowd filter and crowd-averaging in fine-tuning targets, and the 6-forecast trimmed-mean ensemble. Zero-shot base models are miscalibrated; fine-tuning + ensembling fixed this without calibration-specific training.

---

## 5. Quantitative Results

### Headline (test set, 914 questions, averaged over all retrieval dates)
| Metric | System | Crowd | System+Crowd aggregate |
|---|---|---|---|
| Brier | **.179** (±.003) | **.149** (±.003) | **.146** (±.002) — weighted avg, crowd weight 4× (tuned on validation) |
| Accuracy | 71.5% (±.7) | 77.0% (±.7) | 77.8% (±.6) |

- System (.179) ≫ best no-retrieval baseline (GPT-4-1106-Preview .208).
- Context vs prior work: on Autocast, Zou et al. 2022 got 65.4% accuracy (crowd 92.8%); Yan et al. 2024 got 67.9%. This system: 71.5% on harder questions (crowd only 77.0%).

### Selective settings (Table 3, test set; bold = beats crowd)
| Regime | System Brier | Crowd | Sys+Crowd avg (unweighted) | % forecasts / % questions retained |
|---|---|---|---|---|
| All questions | .179 | .149 | **.146** | 100 / 100 |
| **Crowd uncertain** (crowd p ∈ [.3, .7]) | **.238** | .240 | **.233** | 51 / 56 |
| Early retrieval (dates 1–3 of 5) | .186 | .162 | **.159** | 84 (text says 66%) / 100 |
| ≥5 relevant articles retrieved | .175 | .142 | **.140** | 84 / 94 |
| **All 3 criteria jointly** | **.240** | .247 | **.237** | 22 / 43 |

- Under all three criteria jointly the system **beats the crowd by >1.5 SE in both Brier and accuracy** (58.0% vs 54.2%).
- **System complements the crowd in every setting**: the system+crowd aggregate beats the crowd alone in all rows (e.g., .149 → .146 overall).

### Strengths/weaknesses analysis (validation set, Section 6.2)
- Crowd uncertain (.3–.7): system .199 vs crowd **.246** — large win.
- Crowd near-certain (within .05 of 0/1): system accuracy is **+7%** vs crowd but Brier is **worse by .04** — because the system won't output extreme probabilities (hedging).
- Earlier retrieval dates (1–3): system outperforms; at dates 4–5 the crowd pulls ahead — the system's Brier improves more slowly as evidence accumulates because it keeps hedging.
- More relevant articles → monotonically better system Brier, surpassing the crowd with enough articles.

### By category (test, system/crowd Brier)
Science & Tech .143/.114; Healthcare & Biology .074/.063; Economics & Business .198/.147; Politics & Governance .172/.145; Education & Research .163/.129; Arts & Recreation .221/.146 (worst gap); Security & Defense .174/.129; **Sports .175/.171 (near parity)**. Environment & Energy noted as much behind. (Authors caution: small subsamples.)

### By platform (test, system/crowd Brier)
Metaculus .134/.104; GJOpen .193/.157; INFER .247/.310 (system better, but n=4); Polymarket .172/.127; Manifold .219/.200.

### Ablations (Table 5, test Brier)
| Configuration | Brier |
|---|---|
| Full system (FT GPT-4-0613 + base GPT-4-1106-Preview) | **.179** |
| Fine-tuned GPT-4-0613 only | .182 |
| Fine-tuned GPT-3.5 + base GPT-4 | .181 |
| Fine-tuned GPT-3.5 only (FT on all 13,253 samples) | .183 |
| Base GPT-4 only (no fine-tuned model) | .186 |
| Base GPT-4, **no retrieval** | .206 |

Reading: **retrieval is the biggest single component** (.186 → .206 without it, i.e., ~.02 Brier); **fine-tuning adds ~.007** (.186 → .179); prompt optimization alone over the baseline is minor (.206 ablation ≈ .208 baseline); performance does **not** hinge on GPT-4 as the fine-tuned model (FT GPT-3.5 nearly as good, .182–.183); ensembling adds ~.003 (validation).

---

## 6. Key Takeaways and Limitations (authors' own)

1. **Off-the-shelf LMs are bad forecasters** (≈ random; best .208 vs crowd .149); retrieval + optimized reasoning + fine-tuning close most of the gap.
2. **Hedging/underconfidence near extremes** is the dominant failure mode — attributed to safety training; hurts most when the crowd is near-certain and at late retrieval dates. (A natural reimplementation fix the paper does *not* apply: extremizing.)
3. **Dependence on retrieval quality**: the system only approaches/beats the crowd with ≥5 relevant articles; performance scales with article count.
4. **Knowledge cutoffs constrain everything**: train/val must pre-date the cutoff, test must post-date it; fine-tuning was stuck on the older GPT-4-0613. Newer models forecast better naturally (GPT-4-1106 beats GPT-4-0613 by .02 baseline Brier), so fine-tuning a newer model should help.
5. Statistical caveats: SEs assume i.i.d. but data are time-series (underestimated errors); category/platform subsample conclusions are noisy.
6. Future directions named: iterative self-supervision; "postdiction" data generation (later-cutoff LMs writing questions for earlier LMs); harvesting historical events in the wild as forecasting questions; domain-adaptive fine-tuning (baseline category Brier correlates with pre-training knowledge per Section B.3).
7. Best deployment framing: **complement, not replacement** — averaging with the crowd improves the crowd in every regime; selective forecasting (uncertain crowd + early + ≥5 articles) beats the crowd outright.

---

## 7. Reimplementation Cheat Sheet (all configs in one place)

| Component | Setting |
|---|---|
| Query generation | GPT-4-1106-Preview, temp 0, 2 prompts (expansion + sub-question decomposition), 6 queries each, union + raw question |
| News APIs | NewsCatcher + Google News (gnews); top 10 articles/query/API; English; date-range-bounded |
| Relevance rating | GPT-3.5-Turbo, temp 0, 1–6 scale, input = title + first 250 words; keep rating ≥ 4 |
| Summarization | GPT-3.5-Turbo, temp 0.2, ≤100-word question-conditioned summaries |
| Context for reasoning | top **k = 15** summaries, **ordered by relevance** (k swept over {5,10,15,20,30}) |
| Base reasoning model | GPT-4-1106-Preview, top-3 of 15 scratchpad prompts (7-step: rephrase → no-reasons → yes-reasons → superforecaster aggregation → initial prob → over/underconfidence + base-rate check → final prob in `*0.xx*`) |
| Fine-tuned model | GPT-4-0613, 6,000 most recent of 13,253 filtered samples, 2 epochs; inference: no scratchpad instructions, 3 samples at **T = 0.5** |
| FT data filters | beats crowd Brier; \|pred − crowd\| ≤ 0.15; target = average(model pred, crowd pred); generated by GPT-4-1106-Preview + Claude-2.1 across 2 retrieval configs × 4 prompts |
| Ensemble | trimmed mean over 6 forecasts (halve weight of forecast furthest from median, redistribute) |
| Calibration | none post-hoc (binning/isotonic tried, no gain) |
| Eval | Brier, 5 geometric retrieval dates/question, avg over dates then questions |
| Crowd weighting | system+crowd weighted average with crowd weight 4× (optimal on validation) |

---

## 8. Follow-up / Related Work Building on This Paper

- **Code & data:** [dannyallover/llm_forecasting](https://github.com/dannyallover/llm_forecasting) (the paper's official repo; modules for IR, relevance/summarization, reasoning+aggregation; demo notebook); datasets `YuehHanChen/forecasting` and `YuehHanChen/forecasting_raw` on HuggingFace. A third-party reimplementation exists: [getdatachimp/llm-superforecaster](https://github.com/getdatachimp/llm-superforecaster).
- **Consistency Checks for Language Model Forecasters** (Paleka et al., [arXiv 2412.18544](https://arxiv.org/abs/2412.18544), **ICLR 2025 Oral**): evaluates forecasters via logical-consistency metrics (resolvable instantly rather than waiting for outcomes); directly uses/extends Halawi et al.'s `AdvancedForecaster` in its codebase ([dpaleka/consistency-forecasting](https://github.com/dpaleka/consistency-forecasting)); consistency violations correlate with future Brier scores.
- **ForecastBench** (Karger et al., Forecasting Research Institute, [arXiv 2409.19839](https://arxiv.org/abs/2409.19839), **ICLR 2025**): the de facto successor benchmark — dynamic, contamination-free, nightly-updated ([repo](https://github.com/forecastingresearch/forecastbench)); finds expert human forecasters still beat top LLMs as of its publication, tempering this paper's "near-human" claim.
- **Skepticism thread:** Halawi himself publicly [pushed back](https://x.com/dannyhalawi15/status/1833219810530103430) on a 2024 "superhuman forecasting" claim (539/CAIS), noting knowledge-cutoff leakage in its evaluation; see also LessWrong's ["Contra papers claiming superhuman AI forecasting"](https://www.lesswrong.com/posts/uGkRcHqatmPkvpGLq/contra-papers-claiming-superhuman-ai-forecasting). Key methodological lesson inherited from this paper: rigorous cutoff/leakage control and date-bounded retrieval.
- **2025 systems surpassing this pipeline:**
  - **AIA Forecaster** ([arXiv 2511.07678](https://arxiv.org/html/2511.07678v1)): reports superforecaster-parity Brier (0.0753 vs 0.0740 human SOTA on FB-Market), explicitly building on the agentic search + ensemble recipe.
  - **Mantic / Thinking Machines** ([Training LLMs to Predict World Events](https://thinkingmachines.ai/news/training-llms-to-predict-world-events/)): RL fine-tuning on outcomes (an evolution of this paper's self-supervised outcome-filtered SFT); Mantic beat the Metaculus community prediction in Fall 2025.
  - **Evaluating LLMs on Real-World Forecasting Against Expert Forecasters** ([arXiv 2507.04562](https://arxiv.org/abs/2507.04562)): o3 reaches Brier 0.1352 vs the 0.149 crowd reference — frontier models now exceed this paper's full pipeline zero-shot-with-search.
  - Metaculus's quarterly **AI Forecasting Benchmark** tournaments (Q2/Q3 2025 et seq.) operationalize exactly this human-crowd-vs-bot comparison; pro forecasters still led as of [Q2 2025](https://www.lesswrong.com/posts/Surnjh8A4WjgtQTkZ/q2-ai-benchmark-results-pros-maintain-clear-lead).
- Related contemporaneous work cited in/around the paper: Schoenegger et al. "Wisdom of the Silicon Crowd" ([arXiv 2402.19379]) — LLM ensembles rival crowds; "AI-Augmented Predictions" ([arXiv 2402.07862](https://arxiv.org/pdf/2402.07862)) — LLM assistants improve human forecasters.

**Notable nuances for a Metaculus-bot builder:** (1) the biggest wins come from retrieval and from knowing *when not to forecast* (selective regimes); (2) the trimmed-mean-of-6 ensemble and the crowd-anchored fine-tuning filters are cheap, effective tricks; (3) the system's main residual weakness — refusing to go below ~0.05 or above ~0.95 — is exactly where log-score-style Metaculus scoring punishes hardest, so adding an extremizing/calibration layer (which the authors found unnecessary for Brier) may still pay off under other scoring rules.
