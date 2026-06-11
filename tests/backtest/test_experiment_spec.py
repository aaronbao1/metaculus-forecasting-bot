"""ExperimentSpec.from_yaml + budget profiles."""
import pytest

from backtest.experiments.profiles import PROFILES
from backtest.experiments.spec import ExperimentSpec

CONTRACT_YAML = """\
name: agg-sweep
corpus: {platforms: [metaculus], qtypes: [binary], resolved_only: true, stratum: clean}
horizons: [close-7d, close-30d]
cutoff_mode: strict
profile: standard
variants:
  - {name: trimmed_gmo, patch: {aggregation_method: trimmed_gmo}}
baselines: [chance, cp_echo]
seed: 0
"""


def test_from_yaml_parses_contract_example(tmp_path):
    p = tmp_path / "exp.yaml"
    p.write_text(CONTRACT_YAML)
    spec = ExperimentSpec.from_yaml(p)
    assert spec.name == "agg-sweep"
    assert spec.corpus["platforms"] == ["metaculus"]
    assert spec.corpus["stratum"] == "clean"
    assert spec.horizons == ["close-7d", "close-30d"]
    assert spec.cutoff_mode == "strict"
    assert spec.profile == "standard"
    assert spec.variants == [{"name": "trimmed_gmo",
                              "patch": {"aggregation_method": "trimmed_gmo"}}]
    assert spec.baselines == ["chance", "cp_echo"]
    assert spec.seed == 0


def test_from_yaml_defaults(tmp_path):
    p = tmp_path / "min.yaml"
    p.write_text("name: tiny\n")
    spec = ExperimentSpec.from_yaml(p)
    assert spec.cutoff_mode == "strict"
    assert spec.profile == "smoke"
    assert spec.horizons == ["close-7d"]
    assert spec.seed == 0
    assert spec.variants == []
    assert spec.baselines == []


def test_from_yaml_rejects_unknown_keys_and_missing_name(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\nbudget: 99\n")
    with pytest.raises(ValueError, match="unknown spec keys"):
        ExperimentSpec.from_yaml(p)
    p2 = tmp_path / "noname.yaml"
    p2.write_text("profile: smoke\n")
    with pytest.raises(ValueError, match="name"):
        ExperimentSpec.from_yaml(p2)


def test_from_yaml_rejects_unknown_profile_and_bad_variant(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("name: x\nprofile: mega\n")
    with pytest.raises(ValueError, match="profile"):
        ExperimentSpec.from_yaml(p)
    p2 = tmp_path / "v.yaml"
    p2.write_text("name: x\nvariants: [{patch: {}}]\n")
    with pytest.raises(ValueError, match="variant"):
        ExperimentSpec.from_yaml(p2)


def test_profiles_shapes():
    assert set(PROFILES) == {"smoke", "standard", "full"}
    smoke = PROFILES["smoke"]
    assert smoke.max_questions == 30
    assert len(smoke.ensemble_override) == 1
    assert smoke.ensemble_override[0].runs == 1
    assert PROFILES["standard"].max_questions == 300
    assert PROFILES["standard"].ensemble_override is None
    assert PROFILES["full"].max_questions is None
    for p in PROFILES.values():
        assert p.cost_cap_usd > 0
        assert 0 < p.est_cost_per_question_usd < p.cost_cap_usd
