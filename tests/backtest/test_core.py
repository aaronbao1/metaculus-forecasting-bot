"""Core type contract: QType, BacktestQuestion, Forecast, StageRecord,
LedgerEntry, ForecastArtifact, and the artifact <-> dict JSONL round-trip."""
import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from backtest.core import (
    BacktestQuestion,
    Forecast,
    ForecastArtifact,
    LedgerEntry,
    QType,
    StageRecord,
    artifact_from_dict,
    artifact_to_dict,
)


def _question(**kw):
    base = dict(
        qid="metaculus:31678",
        platform="metaculus",
        native_id="31678",
        qtype=QType.BINARY,
        title="Will X happen?",
    )
    base.update(kw)
    return BacktestQuestion(**base)


def test_qtype_values_are_platform_strings():
    assert QType.BINARY.value == "binary"
    assert QType.MULTIPLE_CHOICE.value == "multiple_choice"
    assert QType.NUMERIC.value == "numeric"
    assert QType.DISCRETE.value == "discrete"
    assert QType("binary") is QType.BINARY


def test_question_defaults_and_frozen():
    q = _question()
    assert q.description == ""
    assert q.resolution_criteria == ""
    assert q.fine_print == ""
    assert q.open_time is None and q.scheduled_close_time is None
    assert q.spot_scoring_time is None and q.actual_resolve_time is None
    assert q.resolution is None
    assert q.options == ()
    assert q.scaling is None
    assert q.question_weight == 1.0
    assert q.provenance == {}
    with pytest.raises(FrozenInstanceError):
        q.title = "nope"


def test_forecast_is_frozen_with_optional_fields():
    f = Forecast(qtype=QType.BINARY, binary_prob=0.6)
    assert f.mc_probs is None and f.cdf is None
    with pytest.raises(FrozenInstanceError):
        f.binary_prob = 0.7


def test_artifact_round_trip_full():
    art = ForecastArtifact(
        qid="metaculus:31678",
        horizon="close-7d",
        as_of=datetime(2025, 3, 1, 12, 30),
        system_name="ensemble",
        config_fingerprint="abc123",
        forecast=Forecast(
            qtype=QType.NUMERIC, cdf=tuple(i / 200 for i in range(201))
        ),
        stages=[
            StageRecord(
                stage="research",
                inputs={"question": "Will X happen?"},
                outputs={"report": "evidence summary"},
                model="openai/gpt-5.4",
                tokens_in=10,
                tokens_out=20,
                cost_usd=0.01,
                started_at="2025-03-01T12:00:00",
                ended_at="2025-03-01T12:01:00",
            )
        ],
        evidence_ledger=[
            LedgerEntry(
                provider="asknews",
                query="x event",
                url="https://example.com/a",
                doc_date="2025-02-27",
                date_trust="snapshot",
                note="archive hit",
            )
        ],
        contamination="clean",
        cutoff_mode="strict",
        error=None,
    )
    d = artifact_to_dict(art)
    wire = json.loads(json.dumps(d))  # must survive a JSONL round-trip
    back = artifact_from_dict(wire)
    assert back == art
    assert back.forecast.cdf == art.forecast.cdf  # tuple restored, not list


def test_artifact_round_trip_skipped_forecast():
    art = ForecastArtifact(
        qid="q",
        horizon="spot",
        as_of=datetime(2025, 1, 1),
        system_name="chance",
        config_fingerprint="00",
        forecast=None,
        error="skipped: no data",
    )
    back = artifact_from_dict(json.loads(json.dumps(artifact_to_dict(art))))
    assert back == art
    assert back.forecast is None
    assert back.error == "skipped: no data"


def test_artifact_round_trip_mc_forecast():
    art = ForecastArtifact(
        qid="metaculus:9",
        horizon="close-30d",
        as_of=datetime(2025, 6, 1, 0, 0, 1),
        system_name="s",
        config_fingerprint="f",
        forecast=Forecast(
            qtype=QType.MULTIPLE_CHOICE, mc_probs={"a": 0.7, "b": 0.3}
        ),
    )
    back = artifact_from_dict(json.loads(json.dumps(artifact_to_dict(art))))
    assert back == art
    assert back.forecast.mc_probs == {"a": 0.7, "b": 0.3}


def test_dict_uses_iso_strings_and_enum_values():
    art = ForecastArtifact(
        qid="q",
        horizon="spot",
        as_of=datetime(2025, 1, 2, 3, 4, 5),
        system_name="s",
        config_fingerprint="f",
        forecast=Forecast(qtype=QType.BINARY, binary_prob=0.6),
    )
    d = artifact_to_dict(art)
    assert d["as_of"] == "2025-01-02T03:04:05"
    assert d["forecast"]["qtype"] == "binary"
    assert d["schema_version"] == 1
    assert d["contamination"] == "contaminated"  # default per contract
    assert d["cutoff_mode"] == "strict"


# --- new tests for code-review fixes ---


def test_extra_fields_in_stage_and_ledger_are_ignored():
    """Forward-compatible reads: unknown keys in StageRecord/LedgerEntry dicts must not raise."""
    art = ForecastArtifact(
        qid="metaculus:1",
        horizon="spot",
        as_of=datetime(2025, 3, 1, 12, 30, tzinfo=timezone.utc),
        system_name="s",
        config_fingerprint="f",
        forecast=Forecast(qtype=QType.BINARY, binary_prob=0.5),
        stages=[StageRecord(stage="research")],
        evidence_ledger=[LedgerEntry(provider="asknews", query="q")],
    )
    d = artifact_to_dict(art)
    wire = json.loads(json.dumps(d))
    # Inject unknown future fields
    wire["stages"][0]["latency_ms"] = 42
    wire["evidence_ledger"][0]["latency_ms"] = 99
    # Must not raise TypeError
    back = artifact_from_dict(wire)
    assert back.stages[0].stage == "research"
    assert back.evidence_ledger[0].provider == "asknews"


def test_tz_aware_as_of_round_trips():
    """tz-aware as_of must survive json.dumps/loads + artifact_from_dict with tzinfo intact."""
    art = ForecastArtifact(
        qid="metaculus:1",
        horizon="spot",
        as_of=datetime(2025, 3, 1, 12, 30, tzinfo=timezone.utc),
        system_name="s",
        config_fingerprint="f",
        forecast=Forecast(qtype=QType.BINARY, binary_prob=0.5),
    )
    wire = json.loads(json.dumps(artifact_to_dict(art)))
    back = artifact_from_dict(wire)
    assert back == art
    assert back.as_of.tzinfo is not None


def test_z_suffix_as_of_is_parsed():
    """artifact_from_dict must accept a 'Z'-suffixed ISO timestamp for as_of."""
    art = ForecastArtifact(
        qid="metaculus:1",
        horizon="spot",
        as_of=datetime(2025, 3, 1, 12, 30, tzinfo=timezone.utc),
        system_name="s",
        config_fingerprint="f",
        forecast=Forecast(qtype=QType.BINARY, binary_prob=0.5),
    )
    d = artifact_to_dict(art)
    wire = json.loads(json.dumps(d))
    # Replace the iso string with a Z-suffixed form
    wire["as_of"] = "2025-03-01T12:30:00Z"
    back = artifact_from_dict(wire)
    assert back.as_of == datetime(2025, 3, 1, 12, 30, tzinfo=timezone.utc)
    assert back.as_of.tzinfo is not None
