"""Core backtest types (contract: stdlib-only; see docs/superpowers/specs/
2026-06-11-backtesting-system-design.md and docs/backtesting/COMPATIBILITY.md
(created in Phase 5 of the build plan)).

Artifacts serialize to JSONL via artifact_to_dict / artifact_from_dict:
datetimes as iso8601 strings, enums as their string values, tuples as lists.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from enum import Enum


def _only_known(cls, d: dict) -> dict:
    """Return a copy of *d* with only keys that are declared fields of *cls*.

    Allows forward-compatible deserialization: if a future schema version adds a
    field, older readers can still reconstruct the dataclass without TypeError.
    """
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in names}


class QType(str, Enum):
    BINARY = "binary"
    MULTIPLE_CHOICE = "multiple_choice"
    NUMERIC = "numeric"
    DISCRETE = "discrete"


@dataclass(frozen=True)
class BacktestQuestion:
    qid: str                      # "<platform>:<native_id>", e.g. "metaculus:31678"
    platform: str                 # metaculus|polymarket|kalshi|manifold|halawi|forecastbench
    native_id: str
    qtype: QType
    title: str
    description: str = ""
    resolution_criteria: str = ""
    fine_print: str = ""
    open_time: datetime | None = None
    scheduled_close_time: datetime | None = None
    spot_scoring_time: datetime | None = None
    actual_resolve_time: datetime | None = None
    resolution: object | None = None   # bool (binary) | option str (MC) | float (numeric) | None
    options: tuple[str, ...] = ()
    scaling: dict | None = None        # numeric/discrete: {"range_min","range_max","zero_point",
                                       #  "open_lower_bound","open_upper_bound","cdf_size","unit"}
    question_weight: float = 1.0
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Forecast:
    qtype: QType
    binary_prob: float | None = None          # (0,1)
    mc_probs: dict[str, float] | None = None  # option -> prob, sums to ~1
    cdf: tuple[float, ...] | None = None      # ascending values in [0,1]; len == cdf_size (201 default)


@dataclass
class StageRecord:
    stage: str                    # "research"|"member_llm"|"aggregation"|"calibration"|"probe"|...
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    started_at: str = ""          # iso8601
    ended_at: str = ""


@dataclass
class LedgerEntry:
    provider: str
    query: str
    url: str | None = None
    doc_date: str | None = None   # iso or None
    date_trust: str = "low"       # "snapshot" | "publisher" | "low"
    note: str = ""


@dataclass
class ForecastArtifact:
    qid: str
    horizon: str                  # "close-7d"|"close-30d"|"close-365d"|"spot"|"geo-k"
    as_of: datetime
    system_name: str
    config_fingerprint: str
    forecast: Forecast | None     # None if skipped/failed
    stages: list[StageRecord] = field(default_factory=list)
    evidence_ledger: list[LedgerEntry] = field(default_factory=list)
    contamination: str = "contaminated"   # "clean"|"contaminated"|"contaminated-confirmed"
    cutoff_mode: str = "strict"
    error: str | None = None
    schema_version: int = 1


def artifact_to_dict(a: ForecastArtifact) -> dict:
    """JSON-safe dict: datetimes -> iso strings, enums -> values, tuples -> lists."""
    d = asdict(a)
    d["as_of"] = a.as_of.isoformat()
    if a.forecast is not None:
        d["forecast"]["qtype"] = a.forecast.qtype.value
        if a.forecast.cdf is not None:
            d["forecast"]["cdf"] = list(a.forecast.cdf)
    return d


def latest_artifacts(run_dir, variant: str | None = None) -> dict:
    """The canonical 'latest line wins' scan over a run's artifacts.jsonl
    (append-only; retries append a fresh line per cell). Returns
    {(qid, horizon, system_name): artifact}, or {(qid, horizon): artifact}
    filtered to one variant when given. Single home for the convention —
    the runner's scoring pass, the optimization fitnesses, and the failure
    taxonomy must never disagree on which artifact is current."""
    import json
    from pathlib import Path
    latest: dict = {}
    path = Path(run_dir) / "artifacts.jsonl"
    if not path.exists():
        return latest
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            a = artifact_from_dict(json.loads(line))
            if variant is None:
                latest[(a.qid, a.horizon, a.system_name)] = a
            elif a.system_name == variant:
                latest[(a.qid, a.horizon)] = a
    return latest


def stage_value(artifact, stage: str, attr: str, key: str):
    """First non-None value of stages[stage].<attr>[key]; None-tolerant on
    the artifact itself."""
    for s in (artifact.stages if artifact else []):
        if s.stage == stage:
            val = getattr(s, attr).get(key)
            if val is not None:
                return val
    return None


def artifact_from_dict(d: dict) -> ForecastArtifact:
    fc = d.get("forecast")
    forecast = None
    if fc is not None:
        forecast = Forecast(
            qtype=QType(fc["qtype"]),
            binary_prob=fc.get("binary_prob"),
            mc_probs=fc.get("mc_probs"),
            cdf=tuple(fc["cdf"]) if fc.get("cdf") is not None else None,
        )
    return ForecastArtifact(
        qid=d["qid"],
        horizon=d["horizon"],
        as_of=datetime.fromisoformat(str(d["as_of"]).replace("Z", "+00:00")),
        system_name=d["system_name"],
        config_fingerprint=d["config_fingerprint"],
        forecast=forecast,
        stages=[StageRecord(**_only_known(StageRecord, s)) for s in d.get("stages", [])],
        evidence_ledger=[LedgerEntry(**_only_known(LedgerEntry, e)) for e in d.get("evidence_ledger", [])],
        contamination=d.get("contamination", "contaminated"),
        cutoff_mode=d.get("cutoff_mode", "strict"),
        error=d.get("error"),
        schema_version=d.get("schema_version", 1),
    )
