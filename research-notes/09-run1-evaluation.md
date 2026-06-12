# 09 — Run-1 Evaluation: Current Bot on the Clean Stratum (2026-06-11)

*First live backtest of the production bot through Retrocast.
Run: `current-bot-clean-0-91c43068` — 200 clean post-cutoff binary questions,
T = close−7d, strict cutoffs, research = AskNews-archive-at-T + market-price-at-T.
121/200 forecasts completed (43 lost to OpenRouter weekly-limit 402s near the
end of the run, 36 to the no-research spend guard after the AskNews monthly
quota drained at exactly 200 archive calls × 5 credits). Error cells are
manifest-resumable once limits reset. Run artifacts: logs/backtest/current-bot-clean/.*

## Headline results (n=121 unless noted)

| Comparison | Result |
|---|---|
| Mean baseline score | **+55.8** (chance = 0) |
| Mean Brier | **0.0944** |
| vs chance (paired) | +55.8 [+43.1, +67.1] |
| vs market-price-at-T, all (n=99) | **+5.31 [+2.19, +8.75]** |
| vs market, non-pinned (n=66) | **+8.06 [+3.73, +13.07]** |
| vs market, pinned ≤0.05/≥0.95 (n=33) | −0.19 [−0.94, +0.56] (parity, expected) |
| h2h spot-peer vs market, non-pinned | **+5.59** |
| vs cp_echo (n=22, stale snapshots) | −2.9 [−52, +37] — unpowered, ignore |

**The bot beat the market price one week before close, with a CI excluding
zero, on questions whose markets were genuinely live.** Markets are the
hardest available baseline (Metaculus's own analyses show top AIB bots
roughly matching market prices near close). Read as an upper bound:
backtest ≠ live, single horizon, market-native corpus (87 Polymarket /
20 Manifold / 13 ForecastBench in-sample), residual leakage cannot be
fully excluded despite strict mode.

## Diagnostics

- **Calibration:** excellent at the extremes (p<0.1: freq 0.038 at mean 0.040;
  p>0.9: freq 1.00 at 0.965). **Mid-range is yes-skewed overconfident:**
  [0.1,0.3) freq 0.091 vs mean p 0.191; [0.3,0.5) freq 0.118 vs mean p 0.406.
  Base rate 0.240 vs mean forecast 0.307. → Platt with b<0 (yes-bias
  correction) is exactly what roadmap #1 predicts; fit on these artifacts.
- **Discrimination separation: +0.529** (mean p on Yes-resolvers 0.710 vs
  No-resolvers 0.180) — far above the 26pp documented for AIB bots, though
  the corpus differs in kind.
- **Cap hits:** 16.5% of forecasts sit on the [0.02, 0.98] caps.
- **Spread→error signal (System-2 trigger):** r = +0.295 between ensemble
  spread and squared error — the dormant divergence-escalation seam has
  measurable expected value.
- **Per-family member Brier:** gemini-3.1-pro **0.0897** < gpt-5.4 0.1012 <
  claude-opus-4-6 **0.1175** (n≈117–121). Supports the member-swap
  experiment (Opus→Sonnet-4.5 first).
- **Diversity:** raw member error correlations 0.94–0.96 across families
  (shared question difficulty + one shared research dossier dominate);
  families do disagree ~0.3–0.4 log-odds around consensus. The ensemble
  behaves closer to one forecaster sampled nine times than nine forecasters.

## Worst-loss post-mortems (8 questions, agent-verified)

Taxonomy: **5 genuinely-hard** (market lost by similar margins — Eurovision
semifinal surprises, a final-day sports decider), **2 research-bound**
(the two LARGEST losses: single-title-query AskNews dossiers missed an
imminent GPT-5.5 release the market had at 89%, and returned zero on-topic
documents for an IPO question at 89.5%; one stale recycled-2025 article
actively misled members), **1 elicitation-bound** (WTI $110: decisive
truce-expiry evidence WAS in the dossier; members reasoned it away to 9%
vs market 21.5%; the one member who got it right was trimmed by
aggregation), **0 data defects** (all resolutions independently verified).

Cross-cutting pattern: in 6/8 losses the bot sat on the wrong side of the
market without a stated edge → market-anchor shrinkage / calibration is the
single lever touching the genuinely-hard majority. Corpus hygiene: Manifold
descriptions are blank corpus-wide (fetcher never pulls textDescription);
manifold:9tdQEt09dL is a self-referential 4-bettor market and should be
excluded from scoring.

## Immediate actions (evaluation × roadmap)

1. **Resume the run** when the OpenRouter weekly limit resets (manifest
   resume retries the 79 error cells; research is cached) → n≈200.
2. **Calibration experiment** ($0 replay): fit Platt on run-1 artifacts
   (temporal split), arms vs fixed-α extremization; mid-range yes-skew is
   the documented target.
3. **Market-anchor shrinkage variant** (design + replay-cheap): shrink
   toward price-at-T when ensemble has no stated edge; post-mortems say
   this recovers most genuinely-hard losses' h2h deficit.
4. **Opus→Sonnet-4.5 member swap** (~$30, frozen research).
5. **Multiquery research provider** (~$50–75): the two biggest losses are
   its direct test case.
6. **Fix Manifold descriptions; exclude self-referential markets.**
7. **Divergence-aware aggregation / System-2 trigger** design against the
   r=0.295 spread-error signal (also: trimming buried the lone correct
   member on the WTI question).

*Companion docs: research-notes/08-improvement-roadmap.md (full ranked
roadmap), logs/backtest/current-bot-clean/.../report.md (generated report).*
