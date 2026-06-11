# Metaculus AI Benchmark / FutureEval Tournament Series & Scoring Mechanics — Research Notes

*(Compiled June 2026. Sources: Metaculus scores FAQ — exact text recovered from the open-source Metaculus frontend repo since metaculus.com is Cloudflare-gated; Metaculus notebooks recovered via Wayback Machine; LessWrong/EA Forum mirrors; Metaculus GitHub repos.)*

---

## 1. Metaculus scoring math (exact formulas)

Source of record: **Scores FAQ** — https://www.metaculus.com/help/scores-faq/ (content verified verbatim from `front_end/src/app/(main)/help/scores-faq/page.tsx` + `components/{baseline_math,peer_math,further_math}.tsx` in https://github.com/Metaculus/metaculus).

### 1.1 Log score (foundation of everything)
- Binary / multiple choice: `log score = ln(P(outcome))` where P(outcome) is the probability assigned to the outcome that happened. Always ≤ 0 (0 for a 100% correct prediction, −∞ for 0%).
- Continuous (numeric/date): `log score = ln(pdf(outcome))` — the predicted **probability density function evaluated at the resolution value**. Can exceed 0 since pdf values can exceed 1.
  - **All pdfs have a uniform distribution of height 0.01 added to them** ("prevents extreme log scores").
  - For **discrete** questions, the pmf is used; the minimum value assigned to any outcome cannot go below `0.01 / (number of possible inbound outcomes)` (equivalent operation to the 0.01 uniform on pdfs).
  - If a continuous question resolves **outside the bounds**, it is scored like a binary question on the total probability mass placed beyond that bound ("Will the value be below the lower bound?").
- Asymmetry near 0/1 (FAQ's own table): predicting 99% vs 99.9% on a Yes-resolution gives −0.01 vs −0.001 (gain +0.009), but on a No-resolution gives −4.6 vs −6.9 (loss −2.3). "Only use extreme probabilities when you're sure they're appropriate."

### 1.2 Platform clamping (the actual numbers)
- **Binary predictions are restricted to [0.1%, 99.9%]** (FAQ: "Metaculus restricts binary predictions to be between 0.1% and 99.9%"). The frontend validation string is "Probability must be between 0.1 and 99.9%", and the API client in forecasting-tools raises `ValueError("Prediction value must be between 0.001 and 0.999")` (`forecasting_tools/helpers/metaculus_client.py`). So **not** 0.5%–99.5% and not 1%–99% — those are *bot-side self-caps* people add (the official template bot clamps its own LLM output to [1%, 99%]: `number = min(99, max(1, number))` in `main_with_no_framework.py`).
- **Continuous pdfs restricted to [0.01, ~35]** (FAQ). With the 201-point CDF this comes from the per-bin caps below.
- Resulting score extremes (FAQ tables):
  - Baseline: best +99.9 (binary/MC), +183 (continuous); worst −897 (binary), −232 (MC, 8 options), −230 (continuous). Empirical median baseline ≈ +17 (binary), +14 (continuous); mean ≈ +13 (as of Nov 2023).
  - Peer: best/worst ±691 (binary & MC), ±408 (continuous). Empirical median peer ≈ +2 (binary), +3 (continuous); mean exactly 0 by construction.

### 1.3 Baseline score
General form:
```
Baseline = 100 × (log score(prediction) − log score(baseline)) / scale
```
- Binary: `100 × (ln P(outcome) − ln 0.5)/ln 2` ≡ `100 × (log₂ P(outcome) + 1)`.
- Multiple choice (N options): `100 × (ln P(outcome) − ln(1/N)) / ln N`.
- Continuous: `100 × (ln pdf(outcome) − ln(baseline)) / 2`, where baseline = **1.0** (both bounds closed), **0.95** (one open bound; uniform + 5% out the open bound), **0.9** (both open). The divisor 2 was chosen empirically to match binary score averages.
- Worked example (FAQ): predicting 80% → +68 if Yes / −132 if No (E=+28 at true p=0.8); 90% → +85/−232 (E=+21); 99% → +99/−564 (E=−34). Demonstrates properness/anti-extremizing in expectation.

### 1.4 Peer score
```
Peer = 100 × (1/N) Σᵢ [log score(p) − log score(pᵢ)]   over the N *other* predictions
     ≡ 100 × (ln p − ln GM(pᵢ))         (GM = geometric mean of all others)
```
- **Continuous/discrete Peer scores are further divided by 2** (empirical normalization): `100 × ½ × (1/N) Σ …`.
- Zero-sum: the sum of all participants' Peer scores on a question is always 0.
- The Community Prediction's Peer score is positive because CP (recency-weighted median) > geometric mean of forecasters in most cases, and aggregates beat their inputs. **Site-wide, the CP averages a Peer score of ~12.9 and has ranked top-10 on the global leaderboard over every 2-year window since 2016** (Q2 2025 analysis, https://www.metaculus.com/notebooks/40456/q2-ai-benchmark-results/).

### 1.5 Spot scores (spot peer / spot baseline)
FAQ definition: a "spot" score is the given score type evaluated **only at one instant** — no time-averaging. "Unless otherwise indicated, spot scores are evaluated at the same time the Community Prediction is revealed. Coverage is 100% if there is an active prediction at that time, and 0% if there is not. The math is the same as the given score type."
- **In the AI Benchmark/FutureEval bot tournaments, spot peer scores are evaluated when the question closes** (Q2 2025 analysis, Methodology). Only the last standing forecast before close counts.
- **Head-to-head score** (used in pros-vs-bots analyses, methodology notebook https://www.metaculus.com/notebooks/28552/): a spot peer score computed *as if only two participants existed* — e.g. `100 × ln(bot_median / pro_median)` for a question resolving Yes.

### 1.6 Time-averaging, coverage, truncation (non-spot scores)
- Scores are time-averaged over every instant between open and **scheduled** close; before your first prediction you score 0; predictions persist until updated/withdrawn; exact timestamps (an hour counts 1/24 of a day).
- **Score truncation**: if a question resolves early, the span between resolution and scheduled close counts as 0 in the average — required for properness (otherwise "resolve-early-only-Yes" questions are gameable by predicting high early and dropping later).
- **Coverage** = fraction of question lifetime with a standing prediction; used by tournaments to incentivize early forecasting.

### 1.7 Tournament ranking & prize allocation (current method, introduced March 2024)
```
total score  = Σ_questions (peer score × question weight)     [0 for unanswered]
take         = max(total score, 0)²
prize share  = your take / Σ all users' takes
```
- Rank = order by total score. Question weights ≠ 1.0 are shown in the question sidebar; used to down-weight correlated questions (especially question groups).
- "For a tournament with a sufficiently large number of independent questions, this scoring method is essentially proper." Squaring the take "incentivizes forecasting every question and forecasting early."
- **Min payout rule: tournaments ending after June 1, 2025 don't pay prizes below $50** (redistributed upward); before that the floor was $10.
- **AIB/FutureEval difference**: identical formula but with **spot peer scores** instead of time-averaged peer scores (resources page: "The only difference is that FutureEval uses spot peer rather than regular peer scores"). All AIB seasonal tournaments from Q3 2024 onward have `score_type = "spot_peer_tournament"` (verified in archived tournament page JSON for aibq3 and aibq1).
- **Legacy method** (pre-March 2024): tournament Score = Σ Relative scores; `Relative = log₂(p) − log₂(m)` vs the median m of all other predictions; `Take = e^Score × Coverage`; prize ∝ take. Coverage = average per-question coverage.
- **Hidden period / hidden coverage weights** (human tournaments): CP hidden early in a question's life; coverage accrued during the hidden period can be weighted (e.g. 50% or 100% of question coverage) to discourage CP-copying.
- **Metaculus points** (pre-Nov-2023 legacy): proper log-based mix of baseline-like and peer-like components; retired.

### 1.8 Withdrawal (general FAQ, https://www.metaculus.com/faq/)
- Any forecaster (including bots, via API) can **withdraw** while a question is open. After withdrawal: no scores accrue, no coverage accrues, you're removed from the Community Prediction going forward — none of it retroactive.
- **Auto-withdrawal** feature: by default a prediction is auto-withdrawn after 10% of the question lifetime passes without update/reaffirmation (configurable in settings).
- Interaction with spot scoring: if you're withdrawn at the spot time, spot coverage is 0 → the question contributes 0 to your tournament sum, exactly as if you had never forecast it.

---

## 2. Strategic implications of the scoring rules

- **Per-question properness vs. tournament selection effects.** Each spot peer score is proper (predict your true belief). But the tournament sum gives 0 for skipped questions while a below-average forecast has *negative* expected peer score — so **abstaining/withdrawing on question types where your bot is below the field average is sum-positive** (e.g. Q1 2025 showed several bots strong on binary but weak on the new numeric/MC formats: "numeric and multiple-choice questions … were difficult for competitors" — Q1 winners notebook, https://www.metaculus.com/notebooks/37692/). Metaculus's counterweights: take = max(Σ,0)² rewards volume, and missing questions forfeits expected positive points for above-average bots ("every missed question is a missed point opportunity").
- **Extremizing.** Peer/log scoring is zero-sum against the field. Q3 2024 finding: bots clustered near 50% while Pros confidently went <2% / >90%; the Pros' edge was **discrimination, not calibration** (Q4 2024: Pros' Yes-vs-No forecast gap 44 pts vs bots' 26; Q3: 36 vs 21) — https://www.lesswrong.com/posts/LHdNtJCm93pxNHJKb/ and https://www.lesswrong.com/posts/P8YwCvHoF2FHQoHjF/. So *justified* extremity wins points. But the asymmetric tail (−897 floor on binary baseline; ±691 peer swings) makes unjustified extremity catastrophic: one hallucinated "already resolved" call at 99% can erase a season. Hence the dominant winner habit: **cap final outputs at ~[2%, 98%]** — 38% of Fall 2025 winners cap, and within winners capping is the strongest score correlate (r=+0.48, p=0.005) (Fall 2025 survey, https://www.metaculus.com/notebooks/43337/fall-2025-futureeval-survey/). Q1 2025 survey note: aggregating many runs "blunts extremizing scores, which tend to do a little better."
- **Predicting/copying the community prediction.** On AIB/FutureEval questions the human CP and Pro forecasts are **hidden from bots** (Q1 2025 analysis: "The pro and quarterly cup forecasts were hidden from bots so that bots could not just copy the other's predictions"; Q3 2024: the benchmark's strength is "no community prediction to piggyback on"). On the bot tournament itself, questions open only ~1–1.5 h and are spot-scored at close, so there's no bot-aggregate to copy either. The rules *do* allow using any publicly visible forecast (incl. main-site Metaculus CP, other platforms) — and 41% of Fall 2025 winners sanity-checked against the CP — but **training to minimize deviation from the CP failed when run live**: the CP-optimized GPT-4.1 prompt beat its control on the CP-distance eval, then *underperformed live* in Fall 2025 (+2.76 vs +4.71 avg/question over ~351–354 questions; Prompt-optimization Pt 2, https://www.metaculus.com/notebooks/43362/automated-forecast-prompting-pt-2/). Within winners, CP-checking weakly *negatively* correlates with score (r=−0.32, p=0.08) — "a calibration safety net for bots that aren't yet competitive," while top bots profit by disagreeing. Also, bots are now near CP-level anyway (FutureEval trendline: bots pass the community ~April 2026), so the CP carries shrinking signal.
- **Clamping**: platform-enforced [0.1%, 99.9%] binary; continuous pdf floor 0.01 / ceiling ~35. Bot-side defensive caps ([1,99] in the template, [2,98] common among winners) are strategy, not platform rules.
- **Withdrawing/abstaining is possible for bots** (API withdrawal endpoint + simply not forecasting). AIB "does not require updating" forecasts (unlike Market Pulse), and Metaculus asks bots to submit **one forecast per question** in bot tournaments — the last one before close is what's spot-scored.
- **Leaderboard noise**: most questions resolve at season end; mid-season ranks can move 10+ places ("41st → 12th in the last few days"); "early-resolver bias" favors yes-leaning bots mid-season (Metaculus now often delays resolution to scheduled dates to counter it) — resources page, https://www.metaculus.com/notebooks/38928/.

---

## 3. The tournament series, season by season

**Series overview:** Year 1 = $120k over four quarterly $30k tournaments (Q3 2024 → Q2 2025). Year 2 = $175k: three 4-month $50k seasons (Fall 2025, Spring 2026, Summer 2026) + $1k bi-weekly MiniBench. Rebranded **FutureEval** in Feb 2026. Pros beat the bot team every quarter: **−11.3 → −8.9 → −17.7 → −20.03** (head-to-head spot peer, bot team vs Pro team).

### Q3 2024 (tournament `aibq3`, Jul 8 – Oct 8, 2024, $30k, score_type spot_peer_tournament)
- 55 bots, 255 weighted questions (binary only), 10 Pros on 113 overlap questions. Results notebook: https://www.metaculus.com/notebooks/28784/aibq3results/ (mirror: https://www.lesswrong.com/posts/LHdNtJCm93pxNHJKb/).
- **Pros beat the 9-bot-median team: head-to-head −11.3 [−21.8, −0.7], p = 0.036** (weighted t-test). Pro median beat all 34 bots that answered >half the weighted questions (31/34 significant). Interpretation aid from the notebook: −11.3 ≈ a forecaster randomly saying 28%/72% vs a perfect 50% forecaster.
- Bots showed positive bias ("when the bots forecast 29%, it happens 12% of the time"), poor scope sensitivity (P(B)+P(C) = 1.24×P(A) vs Pros 1.02×), and avoided extremes. Worst bot losses came at 70–75% on No-resolutions, not at extremes.
- Metaculus baseline bots: mf-bot-1 (GPT-4o) best, mf-bot-3 (Claude 3.5 Sonnet) close behind, mf-bot-2 (GPT-3.5) **dead last of 55, worse than always-50%**.
- **Winners: 1st "Cassie" aka FJ_Researcher0 ($4,447), 2nd MWG ($3,572), 3rd histerio ($3,493)** (winners chunk recovered from FutureEval methodology page archive; full list at https://www.metaculus.com/aib/2024/q3/).

### Q4 2024 (tournament `aibq4`, Oct–Dec 2024, $30k)
- 44 bots, 402 questions (280 bot-only), 10 Pros on 122. Results: https://www.metaculus.com/notebooks/35291/q4-ai-benchmarking-results/ (mirror: https://www.lesswrong.com/posts/P8YwCvHoF2FHQoHjF/metaculus-q4-ai-benchmarking-bots-are-closing-the-gap).
- **Head-to-head −8.9 [−18.8, +1.0], p = 0.079 (not significant)** — "bot team" was a team of one: the single top bot, **pgodzinai** (Phil Godzin). Pros beat all 32 individual bots answering ≥50% of questions (29 significant). Both sides well-calibrated; Pros won on discrimination.
- Metaculus house-bot LLM ranking (avg spot peer): o1-preview 6.6 [1, 12.1] > Claude 3.5 Sonnet (Oct) 3.6 > GPT-4o 1.3 > Claude 3.5 Sonnet (June) −9.9. House bots switched **Perplexity → AskNews** this quarter.
- **Winners: pgodzinai $9,658; MWG $4,477; GreeneiBot2 $3,930; manticAI $3,410; histerio $3,312** (https://forum.effectivealtruism.org/posts/nBp3CiMDPAnf8GHTz/; winners notebook: https://www.metaculus.com/notebooks/31370/ai-forecasting-benchmark-q4-winners-q1-warmups/). pgodzinai's recipe: question grouping for consistency, Perplexity+AskNews both, 3×GPT-4o + 5×Claude-3.5 runs, drop extremes, average; 15–40 hours invested (~$240/hr).
- Survey: 76% of winners aggregated repeated LLM calls; 75% used <100 calls/question; 69% solo developers.

### Q1 2025 (tournament `aibq1`, scored Jan 20 – Apr 11, 2025, $30k, `spot_peer_tournament`)
- **First quarter with numeric + multiple-choice questions** (mix: 57% binary / 21% numeric / 22% MC). 34 bot makers + 11 Metac house bots; 424 questions (15 annulled); **10 questions launched at a random hour each weekday, open for ~1 hour**. Pros: 100 questions → 96 scorable overlaps; Pro tournament made public from Q1 (https://www.metaculus.com/tournament/pro-benchmark-q12025/). Analysis: https://www.metaculus.com/notebooks/38673/q1-ai-benchmarking-results/ (mirror: https://www.lesswrong.com/posts/rDy5z8ZEtMrEGnfBd/q1-ai-benchmark-results-pro-forecasters-crush-bots; interactive data: https://q1-2025-aib-analysis.streamlit.app/).
- **Head-to-head −17.7 [−28.3, −7.0], p = 0.001** (one-sided t-test on spot peer scores, top-10 bot team).
- **Best bot overall = Metaculus's own template bot metac-o1** (o1 + AskNews; code: https://github.com/Metaculus/forecasting-tools/blob/main/forecasting_tools/forecast_bots/official_bots/q1_template_bot.py) — house bots are prize-ineligible. **Quarterly Cup CP beat the bot team** on the 27 shared questions (n too small for significance); metac-o1 placed **25th of 617** humans in the Quarterly Cup (top 5%). Control decay: metac-gpt-4o fell 4th (Q3) → 17th (Q4) → 44th/second-to-last (Q1) — the field is improving. Search comparison: metac-o1+AskNews avg 7.99 (1st), +Perplexity Sonar-Pro 2.08 (4th), +Exa −0.30 (6th) — not significant.
- **Prize winners: 1st manticAI $7,685.18; 2nd acm_bot $5,498.51; 3rd GreeneiBot2 $4,065.10; +7 others $12,752** (https://www.metaculus.com/notebooks/37692/). Binary-only leaderboard: manticAI, pgodzinai, GreeneiBot2, acm_bot, then metac-o1 — competitors struggled with the new numeric/MC formats.

### Q2 2025 (tournament `aibq2`, Apr – Jun 2025, $30k)
- 54 bot-maker bots + **42 Metac house bots** (= "96 forecasting bots" in Metaculus's tweet, https://x.com/metaculus/status/1958239204325879854); 348 questions (58% binary / 21% numeric / 21% MC); Pros on ~96 questions (93 used in overlap analysis). Analysis: https://www.metaculus.com/notebooks/40456/q2-ai-benchmark-results/ (mirror: https://www.lesswrong.com/posts/Surnjh8A4WjgtQTkZ/q2-ai-benchmark-results-pros-maintain-clear-lead).
- **Head-to-head −20.03 [−28.63, −11.41], p = 0.00001.** "All 10 Pros would rank 1st–10th if competing individually" against the bots. By type: binary −14.8, numeric −23.2, MC −32.9.
- **Best LLM = OpenAI o3** (metac-o3+asknews finished **2nd overall** and topped MC-only); **Lightning Rod Labs topped numeric-only**. Search-provider sweeps: "best search is uncertain" (AskNews best with Gemini 2.5 Pro; Perplexity Sonar-Reasoning-Pro best with DeepSeek-R1).
- **Winners: 1st Panshul42 $7,550 (open-sourced: https://github.com/Panshul42/Forecasting_Bot_Q2); 2nd pgodzinai $4,563; 3rd CumulativeBot $3,270; +16 others $14,617** (https://www.metaculus.com/notebooks/39140/winners-of-q2-2025-ai-benchmark-tournament/). Top three were students/hobbyists.
- Survey (19 respondents): aggregation +1,799 coverage-adjusted points; manual review +1,041; LLM calls/question correlated 0.40 with score; dev-time only 0.20.

### Year 2 plan (announced Aug 2025): https://www.metaculus.com/notebooks/38909/another-year-of-ai-benchmarking-the-plan/
- **$175k total**: 3×$50k seasonal tournaments aligned with the Metaculus Cup (Fall/Spring/Summer), 300–500 questions each, many **imported/synced from the main site to enable bot-vs-community-prediction comparison**; plus **MiniBench**: back-to-back 2-week $1k tournaments, ~60 auto-generated/auto-resolved questions (FRED, Google Trends, Yahoo Finance, stocks, Metaculus-CP questions), warm-up Aug 18, 2025, official start Sept 1, 2025; MiniBench list: https://www.metaculus.com/aib/minibench/. Bots also prize-eligible in Market Pulse ($7–7.5k/quarter; requires numeric *group* questions and continuous updating — unlike AIB).

### Fall 2025 (tournament `fall-aib-2025`, Sept 1 – Dec 2025; most questions resolved Jan 1–20, 2026; finalized Feb 2026; $50k)
- **123 bots — largest season to date; 377 questions.** No quarterly pros-vs-bots writeup this time: the comparison moved to the continuous **FutureEval Model Leaderboard**. Survey analysis (39 respondents: 29 winners, 10 non-winners): https://www.metaculus.com/notebooks/43337/fall-2025-futureeval-survey/; long-form advice: https://www.metaculus.com/notebooks/43357/bot-advice-fall-2025/.
- Key findings: 34/39 used frontier models ("table stakes"); **#1 predictor = number of distinct research sources (r=0.42, p=0.006; winners 1.72–1.75 vs 1.00)**; winners ~28 LLM calls/question vs 7 (p=0.022); winners' median cost ~$0.90/question (top-15 ~$1.40); historical-question lookup 34% vs 0% (p=0.04); capping predictions strongest within-winner differentiator (r=+0.48); top-5 scaffolded bots beat their same-model metac baseline by **5–11 peer points/question (1,600–3,800 total)**; ~27-point/question spread within the GPT-5 family from scaffolding alone; "good scaffolding ≈ 9 months of base-model progress" (models improving ~0.9 pts/month). metac-gpt-5-high+asknews − metac-gpt-4o+asknews ≈ 13 pts/question.
- Notable metac bots: **metac-asknews-deepnews[research-only] was the top house bot (3,491 pts)** vs metac-gpt-5+asknews (3,252); with DeepSeek-R1, GPT-4o-search-preview was the best search (avg 3.85); Exa Online worst (−7.79).
- Named placements (winner names weren't published in a results notebook; 29 bots won prizes): **nostreambot 9th** (top open-source, https://github.com/No-Stream/metaculus-bot, retrospective: https://nostream.substack.com/p/on-writing-a-bitter-pilled-metaculus), **Metaculus template bot 13th**, geemus+bot 14th, OliveTestBot 16th, BottyMcBotFace 26th, etc. Full final leaderboard: https://www.metaculus.com/tournament/fall-aib-2025/ and https://www.metaculus.com/aib/2025/fall/.
- Independent MiniBench retrospective (Sept–Dec 2025, 7 rounds, 31–45 bots/round): community-of-bots aggregate ranks top-10; author's per-question scores: binary +3.3, MC +12.8, numeric +28.3 — "almost all binary questions are difficult, almost all numeric questions are easy"; flags leakage risk on Metaculus-CP-linked questions (https://predictably.substack.com/p/fall-ai-forecasting-retrospective).

### FutureEval rebrand (launched ~Feb 12, 2026; CEO post by Deger Turan)
- https://www.metaculus.com/futureeval/ — central home for the benchmark, model leaderboard, bot tournaments. "They're not better than the pros yet… this might come as soon as next year." **Trendlines: bots pass the Metaculus Community Prediction ~April 2026; pass Pro forecasters ~June 2027** (naive linear extrapolation).
- **Model Leaderboard methodology** (https://www.metaculus.com/futureeval/methodology/): skill score from a **Ridge regression on pairwise head-to-head Peer-score differences** (`score_Ai − score_Bi = skill_A − skill_B + ε`); handles staggered participation/sparsity; λ = σ²_error/σ²_skill (Bayesian prior); **spot-scored (bot-tournament) matchups weighted ⅓** vs time-averaged ones; only a model's first year of predictions used; rescaled to Peer-score range and **anchored at GPT-4o = 0**; bootstrap CIs; code: https://github.com/Metaculus/metaculus/blob/.../scoring/management/commands/update_global_bot_leaderboard.py.
- Leaderboard snapshot (archived May 20, 2026): **Metaculus Pro Forecasters 36.00; Metaculus Community 22.38; Claude Opus 4.5 High 32k 19.54; Gemini 3 Pro 18.04; GPT-5.1 High 16.07; Claude Sonnet 4.5 High 32k 14.91; o3 High 14.47; GPT-5.1 13.74; GPT-5 13.71; o3 13.61; GPT-5 High 13.59; GPT-5.2 High 13.38; Kimi K2 High 12.53; Opus 4.1 High 12.28; Sonnet 4.5 12.03; Gemini 3 Flash 11.37; DeepSeek v3.1 High 10.17; DeepSeek 3.2 High 10.07; Grok 4 Fast High 9.98; Grok 4 9.88.** Best model trails Pros by ~16 peer points ("roughly the GPT-4o → GPT-5.1 gap").

### Spring 2026 (tournament `spring-aib-2026`, ID 32916; Jan 5 – ~May 1, 2026; $50k)
- Announcement: https://forum.effectivealtruism.org/posts/5EX9dz7nKthcxECTe/announcing-spring-2026-ai-forecasting-benchmark. Most questions resolve by May 8, 2026 (stragglers to May 31); **payouts targeted June/July 2026** — final winners not yet published as of mid-June 2026. nostreambot was running ~7th as of the May 2026 synthesis.
- **Rule change: max probability change between adjacent CDF bins dropped 0.59 → 0.2** (aligning bot API limits with the human UI); `_standardize_cdf` reference implementation: https://github.com/Metaculus/forecasting-tools/.../numeric_report.py. Survey now mandatory for **all** participants (not just winners) for next-season prize eligibility.

### Summer 2026 (tournament `summer-futureeval-2026`, ID 33022; starts May 18, 2026 – ~Sept 2026; $50k) — currently running
- Announcement: https://forum.effectivealtruism.org/posts/ZfLAN557rGWACKtmc/announcing-metaculus-summer-2026-futureeval-bot-tournament. New **bot-testing-area** (slug `bot-testing-area`, ID 32977) with one practice question per type; question types limited to **Binary, Numeric, Discrete, Multiple Choice** (date/conditional supported in forecasting-tools but not launched); no new question types planned; OpenAI credits running low (participants asked to lean on Anthropic/Google credits).

### The May 2026 synthesis ("AI forecasting in 2026: what 10 analyses say")
- Top conclusions: pros still lead every live forward-looking comparison; all "superhuman/superforecaster-parity" claims rest on backtesting with leakage concerns (Silicon Crowd n=31; Reasoning&Tools n=201; Phan "539" n=177 — fails replication; AIA Forecaster Brier 0.108 vs 0.111, backtest); ForecastBench extrapolation puts LLM-superforecaster parity ~Nov 2026 [Dec 2025 – Jan 2028], FutureEval's own line says ~June 2027; Platt scaling post-hoc improved bot Brier by 0.016 (binary, p=0.00052) and 0.005 (MC) on Q1+Q2 data (https://www.metaculus.com/notebooks/43356/calibration-adjustment-analysis/); fine-tuned open-weight models reach prior-generation parity only; "Bayesian"-themed prompts underperformed in both prompt-optimization rounds. Related: search-provider analysis https://www.metaculus.com/notebooks/43361/search-provider-analysis/, literature review https://www.metaculus.com/notebooks/43430/ai-forecasting-literature-review/.

---

## 4. Bot infrastructure & API

- **Registration**: create a human account → Settings → "My Forecasting Bots" → "Create a Bot" → copy the **METACULUS_TOKEN** (API key). Docs/participate page: https://www.metaculus.com/futureeval/participate/; resources notebook: https://www.metaculus.com/notebooks/38928/ (originally "ai-benchmark-resources", now "futureeval-resources-page").
- **Template repo**: https://github.com/Metaculus/metac-bot-template — fork, add `METACULUS_TOKEN` + `OPENROUTER_API_KEY` as GitHub Actions secrets, enable Actions; the `run_bot_on_tournament.yaml` workflow polls the live seasonal tournament + MiniBench **every 20 minutes**, skipping already-forecast questions; `main.py` uses the **forecasting-tools** package (https://github.com/Metaculus/forecasting-tools), `main_with_no_framework.py` is a raw-requests reference. 30-min video: https://www.loom.com/share/fc3c1a643b984a15b510647d8f760685.
- **API** (base `https://www.metaculus.com/api`, docs at https://www.metaculus.com/api/ — auth required to browse): key endpoints used by bots: `GET /posts/?tournaments=[slug-or-id]&statuses=open` (list questions), `GET /posts/{id}/`, `POST /questions/forecast/` (payload: binary `probability_yes`; MC `probability_yes_per_category`; numeric/discrete `continuous_cdf` with **201 values**), `POST /comments/create/` (bots must post a private-note comment with reasoning; Metaculus flips them public after close, weekly).
- **Forecast constraints**: binary ∈ [0.001, 0.999]; MC options clamped/normalized (template uses [0.01,0.99] then renormalizes); numeric CDF: 201 points, **monotonically increasing by ≥ 5e-05 per step** (= 0.01/200 uniform floor), **max PMF mass per bin 0.2** (was 0.59 until Spring 2026), minimum mass outside open bounds (~0.001), nothing outside closed bounds.
- **Rate limits**: no published hard numbers found; forecasting-tools' client politely sleeps 3.5 s ± 1 s jitter between requests with exponential-backoff retries; the official cadence (Actions every 20 min, ≤5–10 simultaneous question launches, 1–1.5 h windows) keeps volumes low.
- **Question cadence**: questions launch in batches (up to 10 at random hours in Q1 2025; "up to 5 at a time, open 1.5 hours" in the FutureEval era) — bots must run continuously/scheduled; one forecast per question requested; **no updating required** (spot-scored at close).
- **Community prediction visibility**: hidden on bot-tournament questions during their open window (CP reveal = close = spot-scoring time); Pro tournament forecasts private; main-site synced questions' CP is visible *on the main site* and legal to consult ("a bot may use any resources generally available to human forecasters … including publicly available forecasts on Metaculus itself").
- **Free credits**: LLM credits via Metaculus-managed OpenRouter keys (donated by OpenAI, Anthropic, Google; request form, check `limit_remaining` via OpenRouter API); **AskNews: 1,000 calls/month, 4,000 calls/tournament, 5M tokens** (latest-news call = 1, archive call = 5; /news and /deepnews endpoints; account renewal required each season).
- **Rules** (https://www.metaculus.com/notebooks/28751/, summarized on resources page): **no human in the loop** (no previewing/rerunning your bot on open tournament questions, no human-entered forecasts, manual daily *triggering* allowed but bot must forecast via API); bots may train on anything public; updates to bot code mid-season allowed; teams allowed (one prize bot per team; extra bots must be labeled secondary, e.g. `-v2`, and are prize-ineligible); winners must describe or show code and pass an inspection; `metac-*` house bots excluded from prizes; required post-season survey for prize eligibility (since Fall 2025 for all participants).

---

## 5. Rule changes over the seasons (chronological)

| When | Change |
|---|---|
| Nov 2023 | Baseline/Peer scores replace Metaculus points sitewide (https://www.metaculus.com/notebooks/20025/) |
| Mar 2024 | New tournament scoring: Σ(peer × weight), take = max(Σ,0)², replaces legacy `e^Score × Coverage` |
| Jul 2024 | AIB Q3 launches: $30k/quarter, binary-only, **spot peer tournament scoring from day one**, weighted questions, CP/Pros hidden, no-human-in-the-loop rules |
| Q4 2024 | House bots switch Perplexity → AskNews; deliberately more-uncertain questions (avg baseline 29 vs 62 in Q3); single-best-bot "team" methodology |
| Jan 2025 (Q1) | **Numeric + multiple-choice added** (CDF/PMF submissions); randomized launch times, ~1 h windows, up to 10 simultaneous; template bot with scheduler; Pro tournaments made public |
| Jun 1, 2025 | Tournament minimum payout raised $10 → **$50** |
| Aug–Sep 2025 (Year 2) | $120k/yr → **$175k/yr**; quarterly → **3 seasonal $50k tournaments** (Sept/Jan/May) + **$1k bi-weekly MiniBench** (auto-generated questions); **discrete** question type added; questions partly synced with main site/Metaculus Cup for CP comparison; required bot surveys; per-season AskNews renewal |
| Feb 2026 | **FutureEval rebrand**; continuous ridge-regression Model Leaderboard replaces quarterly pros-vs-bots studies; quarterly h2h series (−11.3, −8.9, −17.7, −20.03) frozen as Year-1 record |
| Jan 2026 (Spring) | **Numeric CDF max per-bin step 0.59 → 0.2** (API aligned with human UI); survey mandatory for all participants |
| May 2026 (Summer) | Dedicated bot-testing-area (ID 32977); question types fixed at binary/numeric/discrete/MC; OpenAI credit budget tightened |

**Key open question for Fall 2025/Spring 2026**: Metaculus did not publish named 1st–3rd winners notebooks for Fall 2025 (only the survey/advice writeups; 29 bots won prizes); Spring 2026 winners are pending the June/July 2026 payout cycle. The authoritative final leaderboards live at https://www.metaculus.com/tournament/fall-aib-2025/ and https://www.metaculus.com/tournament/spring-aib-2026/ (Cloudflare-gated; "advanced" toggle reveals house bots/aggregates).
