"""ForecastSystem protocol conformance — pure typing, no I/O."""
from datetime import datetime, timezone

from backtest.core import BacktestQuestion, Forecast, ForecastArtifact, QType
from backtest.systems.protocol import ForecastSystem


class _Dummy:
    name = "dummy"

    def config_fingerprint(self) -> str:
        return "dummy-v1"

    async def forecast(self, q, ctx) -> ForecastArtifact:
        return ForecastArtifact(
            qid=q.qid, horizon="", as_of=datetime.now(timezone.utc),
            system_name=self.name, config_fingerprint="dummy-v1",
            forecast=Forecast(qtype=QType.BINARY, binary_prob=0.5))


def test_conforming_instance_passes_runtime_check():
    assert isinstance(_Dummy(), ForecastSystem)


def test_nonconforming_instance_fails_runtime_check():
    class Nope:
        pass

    assert not isinstance(Nope(), ForecastSystem)


def test_protocol_module_reexports_core_types():
    from backtest.systems import protocol
    assert protocol.BacktestQuestion is BacktestQuestion
    assert protocol.Forecast is Forecast
    assert protocol.ForecastArtifact is ForecastArtifact
    assert protocol.QType is QType
