"""Registry of non-ensemble systems an experiment spec can request via its
`systems:` key — [{name, kind, params}]. These run alongside variants and
baselines but spend no LLM budget (cost projection treats them as free, like
baselines; any StageRecord cost they do report still accrues to actual
spend)."""
from __future__ import annotations


def _timeseries_factory(store, params: dict):
    from backtest.systems.timeseries import TimeSeriesSystem
    return TimeSeriesSystem(**params)


SYSTEM_FACTORIES = {
    "timeseries": _timeseries_factory,
    # reserved for plan item 6 (evolutionary candidates): "code:*" kinds plug
    # in here once an evolved component graduates to a runnable system.
}


def build_extra_systems(spec, store) -> dict:
    """name -> system for spec.systems entries. Raises ValueError on unknown
    kinds (typos must fail before any spend)."""
    systems: dict = {}
    for entry in getattr(spec, "systems", None) or []:
        kind = entry.get("kind")
        factory = SYSTEM_FACTORIES.get(kind)
        if factory is None:
            raise ValueError(f"unknown system kind: {kind!r} "
                             f"(choose from {sorted(SYSTEM_FACTORIES)})")
        params = dict(entry.get("params") or {})
        params.setdefault("name", entry["name"])
        systems[entry["name"]] = factory(store, params)
    return systems
