# 10 — Knowledge-Cutoff Audit of Run-1 (2026-06-11)

*Empirical audit of cutoff enforcement in run `current-bot-clean-0-91c43068`:
mechanical checks over 1,035 evidence-ledger entries, selection/price
integrity for all 200 questions, and four LLM judges reading all 121
research dossiers in full. Companion to research-notes/09 (run-1 evaluation).*

---

# Final Report — Knowledge-Cutoff Audit, Backtest Run-1 (`current-bot-clean-0-91c43068`)

Run shape: 800 artifacts (200 qids × 4 systems), cutoff_mode=strict, horizon = close − 7d exactly, 121 scored `current` forecasts, providers = asknews_historical + markets_historical only, cache_hit=False on every cell.

---

## 1. VERDICT: CLEAN — cutoff was enforced; headline result stands

**0 hard violations** across all automated probes, **0 LEAK** verdicts across all 121 dossiers read in full by four content judges. **6 SUSPECT** verdicts, all the same pattern (recycled/re-crawled page content with anachronistic dates), and **none contains outcome-relevant post-as_of information**:

| qid | suspect content | why it does not affect the result |
|---|---|---|
| forecastbench:polymarket:0x9ff262… (ETH dip $1,800 May; as_of 2026-05-25) | "the lowest was USD 4.9497 on June 2, 2026" (52-week low dated after as_of) | Concerns ENS, not ETH — irrelevant ticker; date typo (June 2, 2025 fits the 52-week window) equally plausible |
| polymarket:1390374 (Finland top 5, Eurovision; as_of 2026-05-09) | "The Eurovision Song Contest 2026 semi-finals have commenced" (semis began May 12) in a pub_date 2026-05-01 article | No results or placements revealed; rest is pre-contest odds; likely anticipatory/sloppy copy |
| forecastbench:polymarket:0xc3fdb5… (Trump AG by June 30; as_of 2026-06-23) | Article ingested 2026-06-04 describes "a settlement agreement released by the U.S. Department of Justice on June 19" as past | Worst-case date (June 19) still **precedes as_of (June 23)** — no post-as_of info reached the forecaster; demonstrates the re-ingestion mechanism only |
| polymarket:1390378 (Greece top 5, Eurovision; as_of 2026-05-09) | Same "semi-finals have commenced" recycled copy | Only pre-contest betting odds; no outcomes |
| manifold:8lUcA292PO (SPY < $720 May 29; as_of 2026-05-22) | "52-week low was USD 20.15 on July 4, 2026" in article ingested 2026-05-19 | Concerns SGP (Spyglass Pharma), not SPY; logically must be a typo for July 4, 2025; zero bearing on the question |
| polymarket:1390363 (Albania top 5, Eurovision; as_of 2026-05-09) | "The grand final of the festival took place on May 16" (past tense) in article ingested 2026-05-05 | Date + venue were publicly scheduled pre-as_of; no result revealed; reads as translation tense artifact (German "findet statt") |

**Impact on headline (+8.06 [3.73, 13.07] vs market, n=66): none.** No suspect item reveals a question's resolution or any post-T fact a forecaster could use; each is an irrelevant-ticker date typo, publicly known schedule information, or content bounded pre-as_of. The headline survives this audit as a **cutoff-compliance** claim.

Separate validity caveat (not a leak): the already-decided-at-T guard was a guaranteed no-op (`actual_resolve_time` NULL for all 200 questions), and of 99 scored cells with a price at T, 11 sat at ≤2%/≥98% and 33 at ≤5%/≥95%; in at least two cases T landed after the deciding event (PSG Champions League final before as_of 2026-06-06, Brier 0.0004; Świątek elimination before as_of 2026-06-01). This is legitimate at-T knowledge, not contamination — but it inflates apparent skill relative to a true forecasting setting and should be flagged alongside the headline.

---

## 2. Evidence summary

**Ledger-date compliance — PASS.** 1,035 ledger entries across 164 of 200 `current` artifacts, all `date_trust="snapshot"`: 893 asknews_historical + 142 markets_historical; **0 violations** of doc_date ≤ as_of; **0** undatable (doc_date=None). Minimum margin (as_of − doc_date) = 26m 14s for asknews; median 6d 5h. All 142 markets entries have doc_date exactly == as_of (price-at-T by construction — boundary-inclusive pass).

**Selection integrity — PASS.** All 200 scored qids: open_time > 2026-03-05 (min 2026-03-06T06:02:44Z, max 2026-05-29T07:09:45Z); open_time ≤ as_of < scheduled_close_time; as_of = close − 7.000d for all 200. 0 violations.

**Price provenance — PASS.** All 383 score rows with `price_at_T` have `price_at_T_ts ≤ as_of` (min margin 12m 58s, median ~20h); no null timestamps. 5/5 random spot-checks (seed=42) match `CorpusStore.value_at_with_ts` exactly on value and timestamp (manifold:lR8AP6cUps 0.48206569413887873 @ 2026-05-24T15:56:13.247Z; manifold:9tdQEt09dL 0.7500000000000001 @ 2026-05-14T17:39:46.516Z; polymarket:1390429 0.026; 1390410 0.38; 1390396 0.023).

**Content judges — 121/121 dossiers read, full coverage via /tmp/audit_manifest.json (121/121 cache files resolved):**

| Judge | dossiers | CLEAN | SUSPECT | LEAK |
|---|---|---|---|---|
| 1 (i%4==0) | 31 | 29 | 2 | 0 |
| 2 (i%4==1) | 30 | 30 | 0 | 0 |
| 3 (i%4==2) | 30 | 28 | 2 | 0 |
| 4 (i%4==3) | 30 | 28 | 2 | 0 |
| **Total** | **121** | **115 (95%)** | **6 (5%)** | **0** |

**Research-presence natural experiment (121 `current` rows):** asknews+markets n=68 mean Brier 0.0989; asknews-only n=22, 0.1654; markets-only n=31, 0.0343. News-present groups score **worse** — the opposite of a news-channel leak signature (markets-only's low Brier is expected confounding: market-priced questions skew easy/near-resolved).

**Quality issues logged (not leaks):** ~16 dossiers are market-price-only (~240 chars; asknews returned nothing); several wholly off-topic retrievals (Terminator 2 film coverage for a Manifold bot-profit question; RIKU Dining dossier 100% off-topic; SF Prop A dossier covering San Diego Measure A instead); recurring source date garbles all bounded pre-as_of ("withdraw in December 2026" for the Dec 2025 Spain event; CL final dated both May 30 and June 2; a recycled April-2025 GPT story with year rewritten).

---

## 3. Enforcement improvements, ranked (risk reduction per unit effort)

A key lesson from the violations: **all 6 SUSPECTs are the recycled/re-ingested-content class (criterion e)** — anachronisms in article *text* under pre-T ingestion timestamps. A pub_date filter alone cannot catch them; only a content-level screen can. This elevates the screen_docs wiring relative to the original gap analysis.

1. **Run-level ledger audit asserting doc_date ≤ as_of** — new `audit_ledger()` in `backtest/timetravel/leakage.py`, hooked into `runner.py` after `system.forecast()`, writing violations to the artifact + run-level `audit.json`. Effort: small. Closes: every provider timestamp regression, current and future; converts the ledger from forensic record to enforcement (this run's 0/1,035 was verified only post-hoc, by hand).
2. **Client-side `pub_date ≤ T` hard filter in `_asknews_historical_research`** (`providers.py` ~line 126), dropping post-T articles with a discard ledger entry. Effort: minutes. Closes: server-side filter trust on the only text channel into prompts; AskNews `end_timestamp` semantics drift.
3. **Wire `screen_docs` to asknews (config-gated, default on in strict)** — ~1 cheap judge call/question. Effort: low-medium. Closes: **the recycled/re-ingestion content class — the only leak class actually observed in run-1** (all 6 SUSPECTs; the Trump-AG case proves post-pub_date edits do reach dossiers). Currently zero documents were leak-screened: `screen_docs` is wired only to the unused serper path.
4. **Decided-at-T detection** — backfill `actual_resolve_time` in the Manifold/Polymarket fetchers, plus a cheaper price-pinned heuristic (≤0.02/≥0.98 from some ts ≤ T through close) to skip or stratum-flag cells. Effort: medium. Closes: not a leak class but the largest **score-validity** hole — the existing guard was inert on 100% of run-1 questions; 11–33% of priced questions were near-certain at T.
5. **`--probe` CLI flag + probe a sample of clean-stratum questions** — any confident-correct hit on a "clean" question falsifies the release-date assumption; fail loud. Effort: low. Closes: weights-memorization risk — the proxy (open_time > 2026-03-05 vs model release dates) the entire "clean" run rests on is currently asserted, never measured.
6. **Persist evidence ledger (+ as_of, cutoff_mode) in research_cache records** (`ensemble_adapter.py`), replaying entries on cache hit. Effort: low. Closes: silent audit-trail loss — any resumed/replayed run currently produces artifacts with empty research ledgers, making this entire audit impossible to repeat.
7. **Live canary: key-gated test that asknews with a 2024 T returns only pre-T pub_dates, plus a per-run synthetic canary cell** grepping a known post-T outcome string against composed research before LLM spend. Effort: medium. Closes: upstream API semantic drift — the failure mode no local code can prevent.
8. **Enable `gdelt_wayback` next run + sampled Wayback CDX cross-check of asknews URLs** (capture ≤ T containing the headline). Effort: medium-high. Closes: the snapshot-trust gap with the only fully independent verification available.
9. **llm_cache full-prompt persistence + StageRecords for leak-judge calls.** Effort: low, low urgency — member prompts are already fully persisted in artifacts.jsonl (1,476/1,476 verified).
10. **As-of-T question-text reconstruction.** Effort: high, risk low-medium for this corpus — document as a comparability caveat in `backtest/reports/report.py` and prefer ForecastBench-frozen question sets for headline numbers.

---

## 4. Residual risk statement — what this audit CANNOT rule out

- **AskNews server-side filtering is trusted, not independently verified.** All 893 news entries' timestamps come from the same API whose filter we are auditing; no Wayback/GDELT cross-check was performed. The Trump-AG SUSPECT proves the ingestion pipeline *can* carry post-pub_date content — in that instance bounded pre-T, but the mechanism exists and only content-reading caught it.
- **Content judging catches only textually detectable anachronisms.** Subtle leakage — post-T editorial selection of which articles AskNews indexed/summarized, hindsight-tinged framing without explicit dates — is invisible to this method. The judges are themselves LLMs and can miss leaks.
- **Weights memorization is excluded by assumption, not measurement.** The clean stratum relies on the open_time > 2026-03-05 release-date proxy; the memorization probe was never run (and is gated to skip clean-stratum questions even when enabled).
- **Question text is fetch-time (June 2026), not as-of-T.** Post-T edits or resolution-criteria clarifications would have reached prompts unflagged; no mechanism exists to detect this.
- **Cutoff compliance ≠ forecasting skill.** The +8.06 [3.73, 13.07] headline is leak-clean per this audit, but a material fraction of questions were effectively decided at T (11/99 priced at ≤2%/≥98%; some T post-dated the deciding event), and ~16 dossiers contained nothing but the market price. The number is valid as measured; its interpretation as prospective skill on genuinely open questions carries this caveat until improvement #4 lands.
- **This audit covers run-1 only.** The enforcement that passed here is partly one-shot (post-hoc checks, fresh fetches); a resumed run would lose ledgers (#6) and an API semantics change would pass silently (#7).

Key paths: run dir `/Users/aaronbao/Developer/sideprojects/forecasting/logs/backtest/current-bot-clean/current-bot-clean-0-91c43068`; manifest `/tmp/audit_manifest.json`; judge dossier dumps `/tmp/judge3/`, `/tmp/judge4/`; code references `backtest/timetravel/{providers,leakage,asof,probe,llm_cache}.py`, `backtest/systems/ensemble_adapter.py`, `backtest/experiments/runner.py`, `backtest/corpus/{store,model_releases}.py`, `scripts/backtest.py`, `experiments/current-bot-clean.yaml`.