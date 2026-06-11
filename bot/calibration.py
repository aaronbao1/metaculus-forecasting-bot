"""Post-aggregation calibration: fitted Platt map, then hard caps.

Don't hardcode bias corrections without data: the Platt layer
sigmoid(a*logit(p) + b) subsumes yes-bias correction (intercept b) and
hedging/overconfidence correction (slope a). It ships at identity and is
fitted by scripts/fit_calibration.py once >= MIN_FIT_N of OUR OWN forecasts
have resolved (+0.016 Brier on Metaculus bot data). Hard caps are tail-risk
insurance under log scoring and ALWAYS apply last (winner correlate r=+0.48).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

MIN_FIT_N = 100


@dataclass
class PlattParams:
    a: float = 1.0
    b: float = 0.0
    n_fit: int = 0

    def is_identity(self) -> bool:
        return self.a == 1.0 and self.b == 0.0


def cap(p: float, lo: float = 0.02, hi: float = 0.98) -> float:
    return float(min(max(p, lo), hi))


def apply_platt(p: float, params: PlattParams) -> float:
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    z = params.a * (np.log(p) - np.log1p(-p)) + params.b
    return float(1.0 / (1.0 + np.exp(-z)))


def fit_platt(predictions, outcomes) -> PlattParams:
    """Minimize log loss of sigmoid(a*logit(p)+b) against resolved outcomes."""
    preds = np.clip(np.asarray(predictions, dtype=float), 1e-4, 1 - 1e-4)
    y = np.asarray(outcomes, dtype=float)
    if len(preds) < MIN_FIT_N:
        raise ValueError(
            f"need >= {MIN_FIT_N} resolved forecasts to fit, got {len(preds)}")
    lo = np.log(preds) - np.log1p(-preds)

    def nll(params):
        z = params[0] * lo + params[1]
        return float(np.mean(np.logaddexp(0.0, z) - y * z))

    res = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead")
    return PlattParams(a=float(res.x[0]), b=float(res.x[1]), n_fit=len(preds))


def calibrate_binary(p: float, params: PlattParams | None,
                     lo: float = 0.02, hi: float = 0.98) -> float:
    if params is not None and not params.is_identity():
        p = apply_platt(p, params)
    return cap(p, lo, hi)


def load_platt(path) -> PlattParams:
    """Fitted params if present; identity otherwise (stopgap default)."""
    path = Path(path)
    if not path.exists():
        return PlattParams()
    d = json.loads(path.read_text())
    return PlattParams(a=d["a"], b=d["b"], n_fit=d.get("n_fit", 0))


def save_platt(params: PlattParams, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"a": params.a, "b": params.b, "n_fit": params.n_fit}))
