"""Declarative experiment spec (spec §6.5): corpus filter × horizons ×
variant grid × budget profile × cutoff mode, loaded from YAML."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ExperimentSpec:
    name: str
    corpus: dict = field(default_factory=dict)
    horizons: list = field(default_factory=lambda: ["close-7d"])
    cutoff_mode: str = "strict"
    profile: str = "smoke"
    variants: list = field(default_factory=list)
    baselines: list = field(default_factory=list)
    # Registry-backed non-ensemble systems (backtest/systems/registry.py):
    # [{name, kind, params}] — e.g. kind "timeseries" for TS submodels.
    systems: list = field(default_factory=list)
    seed: int = 0

    @classmethod
    def from_yaml(cls, path) -> "ExperimentSpec":
        from backtest.experiments.profiles import PROFILES
        d = yaml.safe_load(Path(path).read_text())
        if not isinstance(d, dict) or "name" not in d:
            raise ValueError(
                f"experiment spec {path} must be a mapping with a 'name'")
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown spec keys: {sorted(unknown)}")
        if d.get("profile", "smoke") not in PROFILES:
            raise ValueError(f"unknown profile: {d.get('profile')!r} "
                             f"(choose from {sorted(PROFILES)})")
        for v in d.get("variants") or []:
            if not isinstance(v, dict) or "name" not in v:
                raise ValueError(
                    f"each variant needs at least a 'name': {v!r}")
        for s in d.get("systems") or []:
            if not isinstance(s, dict) or "name" not in s or "kind" not in s:
                raise ValueError(
                    f"each system needs a 'name' and a 'kind': {s!r}")
        return cls(**d)
