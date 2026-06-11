from bot.config import DEFAULT_CONFIG, BotConfig, EnsembleMember, config_hash


def test_default_config_has_diverse_ensemble():
    families = {m.model.split("/")[0] for m in DEFAULT_CONFIG.ensemble}
    assert len(families) >= 2          # decorrelation requires >= 2 model families
    assert sum(m.runs for m in DEFAULT_CONFIG.ensemble) >= 5


def test_config_hash_is_stable_and_sensitive():
    h1 = config_hash(DEFAULT_CONFIG)
    h2 = config_hash(DEFAULT_CONFIG)
    assert h1 == h2 and len(h1) == 12
    other = BotConfig(ensemble=[EnsembleMember("openai/gpt-5.4", 1, 0.5)])
    assert config_hash(other) != h1


def test_submit_defaults_false(monkeypatch):
    monkeypatch.delenv("SUBMIT_PREDICTION", raising=False)
    assert BotConfig().submit_predictions is False
