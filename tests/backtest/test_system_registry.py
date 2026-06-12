"""Task 6: registry-backed extra systems in experiment specs."""
import pytest

from backtest.experiments.spec import ExperimentSpec
from backtest.systems.registry import build_extra_systems


def _spec(**kw):
    base = dict(name="t", variants=[{"name": "current", "patch": {}}],
                baselines=["chance"])
    base.update(kw)
    return ExperimentSpec(**base)


def test_spec_accepts_systems_key(tmp_path):
    yaml_text = """
name: t
profile: smoke
systems:
  - name: ts_naive
    kind: timeseries
    params: {engine: naive_drift}
"""
    p = tmp_path / "s.yaml"
    p.write_text(yaml_text)
    spec = ExperimentSpec.from_yaml(p)
    assert spec.systems == [{"name": "ts_naive", "kind": "timeseries",
                             "params": {"engine": "naive_drift"}}]


def test_spec_rejects_system_without_name_or_kind(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("name: t\nsystems:\n  - {kind: timeseries}\n")
    with pytest.raises(ValueError, match="name"):
        ExperimentSpec.from_yaml(p)


def test_build_extra_systems_timeseries():
    systems = build_extra_systems(
        _spec(systems=[{"name": "ts_naive", "kind": "timeseries",
                        "params": {"engine": "naive_drift"}}]), store=None)
    assert sorted(systems) == ["ts_naive"]
    assert systems["ts_naive"].engine_name == "naive_drift"
    assert systems["ts_naive"].name == "ts_naive"


def test_build_extra_systems_unknown_kind():
    with pytest.raises(ValueError, match="unknown system kind"):
        build_extra_systems(
            _spec(systems=[{"name": "x", "kind": "nope"}]), store=None)


def test_runner_build_systems_includes_registry_and_rejects_collisions(tmp_path):
    from backtest.corpus.store import CorpusStore
    from backtest.experiments.runner import _build_systems
    store = CorpusStore(tmp_path / "corpus")
    spec = _spec(systems=[{"name": "ts_naive", "kind": "timeseries",
                           "params": {"engine": "naive_drift"}}])
    systems = _build_systems(spec, store)
    assert {"current", "chance", "ts_naive"} <= set(systems)

    collide = _spec(systems=[{"name": "chance", "kind": "timeseries"}])
    with pytest.raises(ValueError, match="collide"):
        _build_systems(collide, store)
