"""Per-run markdown report (spec §6.6): leaderboard vs baselines, scores
stratified by contamination, diagnostics, evidence-ledger leakage summary,
cost totals from StageRecords, worst-10 post-mortem table.

Diagnostics and worst-10 titles need question resolutions; the runner
records corpus_root in manifest.json — when it is missing or stale the
report degrades gracefully instead of failing.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from backtest.core import artifact_from_dict
from backtest.scoring.diagnostics import (calibration_curve,
                                          discrimination_separation,
                                          murphy_decomposition)
from backtest.scoring.stats import paired_bootstrap


def _read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _fmt(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def build_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    scores = _read_jsonl(run_dir / "scores.jsonl")
    artifacts = [artifact_from_dict(d)
                 for d in _read_jsonl(run_dir / "artifacts.jsonl")]
    manifest = {}
    mpath = run_dir / "manifest.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text())

    store = None
    root = manifest.get("corpus_root") or ""
    if root and Path(root).exists():
        from backtest.corpus.store import CorpusStore
        store = CorpusStore(Path(root))

    lines = [f"# Retrocast report — {manifest.get('run_id', run_dir.name)}",
             ""]
    status_counts = Counter(manifest.get("statuses", {}).values())
    if status_counts:
        lines += [f"Statuses: {dict(sorted(status_counts.items()))}", ""]

    by_variant = defaultdict(list)
    for r in scores:
        by_variant[r["variant"]].append(r)

    # ---- Leaderboard (ranked by mean baseline score) ----
    lines += ["## Leaderboard", "",
              "| variant | n | mean baseline | mean log | "
              "h2h vs CP (mean [95% CI]) |",
              "|---|---|---|---|---|"]
    ranked = sorted(
        by_variant.items(),
        key=lambda kv: -(_mean([r["baseline_score"] for r in kv[1]])
                         if kv[1] else float("-inf")))
    for variant, rows in ranked:
        h2h = [r["h2h_cp"] for r in rows if r["h2h_cp"] is not None]
        if h2h:
            bs = paired_bootstrap(h2h)
            h2h_cell = (f"{bs['mean']:+.2f} [{bs['lo']:+.2f}, "
                        f"{bs['hi']:+.2f}] (n={len(h2h)})")
        else:
            h2h_cell = "—"
        lines.append(
            f"| {variant} | {len(rows)} | "
            f"{_fmt(_mean([r['baseline_score'] for r in rows]))} | "
            f"{_fmt(_mean([r['log_score'] for r in rows]))} | {h2h_cell} |")

    # ---- Contamination strata ----
    lines += ["", "## Mean baseline score by contamination stratum", "",
              "| variant | stratum | n | mean baseline |", "|---|---|---|---|"]
    strata = defaultdict(list)
    for r in scores:
        strata[(r["variant"], r["stratum"])].append(r["baseline_score"])
    for (variant, stratum), vals in sorted(strata.items()):
        lines.append(f"| {variant} | {stratum} | {len(vals)} | "
                     f"{_fmt(_mean(vals))} |")

    # ---- Diagnostics (binary) ----
    lines += ["", "## Diagnostics", ""]
    if store is None:
        lines.append("_corpus unavailable (manifest corpus_root missing) — "
                     "diagnostics skipped_")
    else:
        for variant in sorted(by_variant):
            pairs = []
            for a in artifacts:
                if (a.system_name == variant and a.forecast is not None
                        and a.forecast.binary_prob is not None):
                    q = store.get_question(a.qid)
                    if q is not None and isinstance(q.resolution, bool):
                        pairs.append((a.forecast.binary_prob, q.resolution))
            if len(pairs) < 2:
                lines.append(f"- **{variant}**: not enough binary pairs "
                             f"(n={len(pairs)})")
                continue
            lines.append(
                f"- **{variant}** (n={len(pairs)}): "
                f"murphy={murphy_decomposition(pairs)}, "
                f"discrimination={discrimination_separation(pairs)}, "
                f"calibration_curve={calibration_curve(pairs, n_bins=5)}")

    # ---- Evidence-ledger leakage summary ----
    lines += ["", "## Evidence-ledger leakage summary", ""]
    trust = Counter(e.date_trust
                    for a in artifacts for e in a.evidence_ledger)
    if trust:
        lines += ["| date_trust | entries |", "|---|---|"]
        for k, v in sorted(trust.items()):
            lines.append(f"| {k} | {v} |")
    else:
        lines.append("_no ledger entries recorded_")

    # ---- Cost (from StageRecords) ----
    lines += ["", "## Cost", ""]
    if artifacts:
        cost_by: dict = defaultdict(float)
        tok_by: dict = defaultdict(int)
        for a in artifacts:
            for s in a.stages:
                cost_by[a.system_name] += s.cost_usd
                tok_by[a.system_name] += s.tokens_in + s.tokens_out
        lines += ["| variant | cost (USD) | tokens |", "|---|---|---|"]
        for name in sorted({a.system_name for a in artifacts}):
            lines.append(f"| {name} | {cost_by[name]:.4f} | "
                         f"{tok_by[name]} |")
    else:
        lines.append("_no artifacts_")

    # ---- Worst-10 post-mortem table ----
    lines += ["", "## Worst 10 by log score", "",
              "| qid | title | variant | horizon | log score |",
              "|---|---|---|---|---|"]
    scored = [r for r in scores if r.get("log_score") is not None]
    for r in sorted(scored, key=lambda r: r["log_score"])[:10]:
        title = ""
        if store is not None:
            q = store.get_question(r["qid"])
            title = (q.title if q is not None else "")[:60]
        lines.append(f"| {r['qid']} | {title} | {r['variant']} | "
                     f"{r['horizon']} | {r['log_score']:.3f} |")

    out = run_dir / "report.md"
    out.write_text("\n".join(lines) + "\n")
    return out
