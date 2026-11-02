# -*- coding: utf-8 -*-
"""
Calibration des probabilités : température (1X2) et Platt (binaire).
Fit hors échantillon pour coller les % prédits aux fréquences réelles.
"""

from __future__ import annotations

import logging
import numpy as np
from scipy.optimize import minimize, minimize_scalar

logger = logging.getLogger(__name__)

OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}


def _softmax_temp(logp: np.ndarray, T: float) -> np.ndarray:
    z = logp / max(T, 0.05)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-12, None)


class TemperatureCalibrator:
    """p' = softmax(log(p) / T). T>1 = moins confiant, T<1 = plus confiant."""

    def __init__(self):
        self.T = 1.0
        self.fitted = False

    def fit(self, probs: np.ndarray, outcomes: list[str]) -> "TemperatureCalibrator":
        probs = np.asarray(probs, dtype=float)
        if probs.ndim != 2 or probs.shape[0] < 15:
            return self
        probs = np.clip(probs, 1e-6, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        y = np.zeros_like(probs)
        valid = []
        for i, o in enumerate(outcomes):
            idx = OUTCOME_INDEX.get(o)
            if idx is None:
                continue
            y[i, idx] = 1.0
            valid.append(i)
        if len(valid) < 15:
            return self
        logp = np.log(probs[valid])
        y = y[valid]

        def nll(T):
            t = float(np.ravel(T)[0])
            p = _softmax_temp(logp, t)
            return float(-np.mean(np.log(np.clip((p * y).sum(axis=1), 1e-12, 1.0))))

        res = minimize_scalar(nll, bounds=(0.35, 2.8), method="bounded")
        self.T = float(res.x)
        self.fitted = True
        logger.info("Calibration température T=%.3f (n=%d, nll=%.4f)", self.T, len(valid), res.fun)
        return self

    def transform(self, home_win: float, draw: float, away_win: float) -> tuple[float, float, float]:
        p = np.array([[home_win, draw, away_win]], dtype=float)
        p = np.clip(p, 1e-8, 1.0)
        p = p / p.sum()
        if not self.fitted:
            return float(p[0, 0]), float(p[0, 1]), float(p[0, 2])
        out = _softmax_temp(np.log(p), self.T)[0]
        return float(out[0]), float(out[1]), float(out[2])


class PlattCalibrator:
    """p' = sigmoid(a * logit(p) + b) pour un marché binaire (NBA)."""

    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self.fitted = False

    def fit(self, p: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
        y = np.asarray(y, dtype=float)
        if p.size < 20:
            return self
        logit = np.log(p / (1.0 - p))

        def nll(ab):
            a, b = float(ab[0]), float(ab[1])
            z = a * logit + b
            z = np.clip(z, -30, 30)
            pr = 1.0 / (1.0 + np.exp(-z))
            pr = np.clip(pr, 1e-12, 1 - 1e-12)
            return float(-np.mean(y * np.log(pr) + (1 - y) * np.log(1 - pr)))

        res = minimize(nll, x0=np.array([1.0, 0.0]), method="Nelder-Mead")
        if res.success:
            self.a, self.b = float(res.x[0]), float(res.x[1])
            self.fitted = True
            logger.info("Calibration Platt a=%.3f b=%.3f (n=%d)", self.a, self.b, p.size)
        return self

    def transform(self, p_home: float) -> float:
        p = min(max(float(p_home), 1e-5), 1 - 1e-5)
        if not self.fitted:
            return p
        logit = math_logit(p)
        z = self.a * logit + self.b
        z = min(max(z, -30), 30)
        return 1.0 / (1.0 + np.exp(-z))


def math_logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))
