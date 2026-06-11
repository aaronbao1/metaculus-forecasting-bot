import numpy as np
import pytest

from bot.calibration import (
    MIN_FIT_N, PlattParams, apply_platt, calibrate_binary, cap,
    fit_platt, load_platt, save_platt,
)


def test_cap_clamps_both_ends():
    assert cap(0.001) == 0.02
    assert cap(0.999) == 0.98
    assert cap(0.5) == 0.5


def test_platt_identity_is_noop():
    p = PlattParams()
    assert apply_platt(0.37, p) == pytest.approx(0.37, abs=1e-6)


def test_platt_negative_intercept_lowers_probability():
    p = PlattParams(a=1.0, b=-0.539)  # the '30% -> 20%' yes-bias map
    assert apply_platt(0.30, p) == pytest.approx(0.20, abs=0.01)


def test_fit_platt_recovers_known_bias():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.05, 0.95, size=4000)
    logit = np.log(true_p / (1 - true_p))
    reported = 1 / (1 + np.exp(-(logit + 0.6)))    # bot systematically too-yes
    outcomes = rng.binomial(1, true_p)
    fitted = fit_platt(reported, outcomes)
    assert fitted.a == pytest.approx(1.0, abs=0.15)
    assert fitted.b == pytest.approx(-0.6, abs=0.15)  # map undoes the bias


def test_fit_platt_refuses_small_samples():
    with pytest.raises(ValueError):
        fit_platt([0.5] * (MIN_FIT_N - 1), [1] * (MIN_FIT_N - 1))


def test_calibrate_binary_identity_params_just_caps():
    assert calibrate_binary(0.995, None) == 0.98
    assert calibrate_binary(0.5, PlattParams()) == 0.5


def test_save_and_load_roundtrip(tmp_path):
    p = PlattParams(a=1.2, b=-0.3, n_fit=150)
    path = tmp_path / "binary.json"
    save_platt(p, path)
    loaded = load_platt(path)
    assert loaded == p


def test_load_missing_file_returns_identity(tmp_path):
    assert load_platt(tmp_path / "nope.json").is_identity()
