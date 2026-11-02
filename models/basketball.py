# -*- coding: utf-8 -*-
"""
models/basketball.py - Modèle NBA

Avant-match : Elo + net rating (attaque/défense) + total attendu + back-to-back.
En direct   : marge restante ∝ temps restant (pas le spread plein match).
Calibration Platt sur P(victoire domicile).
"""

from __future__ import annotations

import math
import logging
import numpy as np
from scipy.stats import norm

from models.elo import EloRating
from models.calibration import PlattCalibrator
from models.features import (
    nba_team_ratings,
    rest_days,
    sort_matches,
)

logger = logging.getLogger(__name__)

SIGMA_MARGIN_FULL = 12.0
SIGMA_TOTAL_FULL = 14.0
TOTAL_GAME_MINUTES = 48
NBA_HOME_ADVANTAGE_POINTS = 3.0
ELO_TO_POINTS = 100 / 3.0
B2B_PENALTY_POINTS = 2.4
LEAGUE_AVG_PTS = 114.0
SHRINK_GAMES = 10.0


class BasketballModel:
    def __init__(self, elo: EloRating | None = None):
        self.elo = elo or EloRating(
            k_factor=18,
            home_advantage=NBA_HOME_ADVANTAGE_POINTS * ELO_TO_POINTS,
            tau_days=70,
        )
        self.history: list[dict] = []
        self.ratings: dict = {}
        self.league_avg_pts = LEAGUE_AVG_PTS
        self.sigma_margin = SIGMA_MARGIN_FULL
        self.sigma_total = SIGMA_TOTAL_FULL
        self.calibrator = PlattCalibrator()

    def _minutes_remaining(self, match_state: dict) -> float:
        time_remaining_secs = match_state.get("time_remaining_secs")
        if time_remaining_secs is not None:
            return time_remaining_secs / 60.0

        period = match_state.get("period", 0)
        if period == 0:
            return float(TOTAL_GAME_MINUTES)

        quarters_left = max(0, 4 - period)
        time_str = match_state.get("time", "12:00")
        try:
            parts = time_str.split(":")
            mins = int(parts[0])
            secs = int(parts[1]) if len(parts) > 1 else 0
            current_remaining = mins + secs / 60.0
        except (ValueError, IndexError):
            current_remaining = 12.0
        return quarters_left * 12.0 + current_remaining

    def _team_off_def(self, team: str) -> tuple[float, float, int]:
        r = self.ratings.get(team)
        if not r:
            return self.league_avg_pts, self.league_avg_pts, 0
        return r["off"], r["def"], r["games"]

    def _shrink(self, value: float, prior: float, games: int) -> float:
        w = games / (games + SHRINK_GAMES)
        return w * value + (1 - w) * prior

    def _prematch_means(self, home_team: str, away_team: str, match_date=None) -> dict:
        off_h, def_h, n_h = self._team_off_def(home_team)
        off_a, def_a, n_a = self._team_off_def(away_team)
        off_h = self._shrink(off_h, self.league_avg_pts, n_h)
        def_h = self._shrink(def_h, self.league_avg_pts, n_h)
        off_a = self._shrink(off_a, self.league_avg_pts, n_a)
        def_a = self._shrink(def_a, self.league_avg_pts, n_a)

        home_pts = 0.5 * (off_h + def_a) + NBA_HOME_ADVANTAGE_POINTS / 2
        away_pts = 0.5 * (off_a + def_h) - NBA_HOME_ADVANTAGE_POINTS / 2

        elo_spread = (
            (self.elo.get_rating(home_team) + self.elo.home_advantage - self.elo.get_rating(away_team))
            / ELO_TO_POINTS
        )
        rating_spread = home_pts - away_pts
        n = min(n_h, n_a)
        w = n / (n + SHRINK_GAMES)
        spread = w * rating_spread + (1 - w) * elo_spread

        rest_h = rest_days(self.history, home_team, match_date)
        rest_a = rest_days(self.history, away_team, match_date)
        if rest_h is not None and rest_h <= 1:
            spread -= B2B_PENALTY_POINTS
            home_pts -= 1.4
            away_pts += 0.4
        if rest_a is not None and rest_a <= 1:
            spread += B2B_PENALTY_POINTS
            away_pts -= 1.4
            home_pts += 0.4

        total = home_pts + away_pts
        return {
            "spread": spread,
            "total": total,
            "home_pts": home_pts,
            "away_pts": away_pts,
            "rest_home": rest_h,
            "rest_away": rest_a,
        }

    def predict_prematch(
        self,
        home_team: str,
        away_team: str,
        match_date=None,
        calibrate: bool = True,
    ) -> dict:
        means = self._prematch_means(home_team, away_team, match_date)
        p_home = float(norm.cdf(means["spread"] / self.sigma_margin))
        if calibrate:
            p_home = self.calibrator.transform(p_home)
        p_home = max(0.02, min(0.98, p_home))
        return {
            "home_win": round(p_home, 4),
            "draw": 0.0,
            "away_win": round(1 - p_home, 4),
            "point_spread": round(means["spread"], 1),
            "expected_total": round(means["total"], 1),
            "home_elo": round(self.elo.get_rating(home_team), 0),
            "away_elo": round(self.elo.get_rating(away_team), 0),
            "model": "prematch_net_total",
        }

    def predict_live(self, match_state: dict) -> dict:
        home_team = match_state.get("home_team", "")
        away_team = match_state.get("away_team", "")
        home_score = match_state.get("home_score", 0) or 0
        away_score = match_state.get("away_score", 0) or 0
        minutes_remaining = self._minutes_remaining(match_state)
        current_margin = home_score - away_score

        if minutes_remaining <= 0:
            if current_margin > 0:
                return {"home_win": 1.0, "draw": 0.0, "away_win": 0.0, "model": "final"}
            return {"home_win": 0.0, "draw": 0.0, "away_win": 1.0, "model": "final"}

        means = self._prematch_means(home_team, away_team, match_state.get("match_date"))
        remaining_frac = max(0.01, minutes_remaining / TOTAL_GAME_MINUTES)
        expected_final_margin = current_margin + means["spread"] * remaining_frac
        sigma_remaining = self.sigma_margin * math.sqrt(remaining_frac)
        p_home = float(norm.cdf(expected_final_margin / sigma_remaining))
        p_home = self.calibrator.transform(p_home)
        p_home = max(0.01, min(0.99, p_home))

        return {
            "home_win": round(p_home, 4),
            "draw": 0.0,
            "away_win": round(1 - p_home, 4),
            "current_margin": current_margin,
            "expected_margin": round(means["spread"], 1),
            "minutes_remaining": round(minutes_remaining, 1),
            "sigma_remaining": round(sigma_remaining, 2),
            "home_score": home_score,
            "away_score": away_score,
            "model": "live_brownian",
        }

    def predict(self, match_state: dict) -> dict:
        status = match_state.get("status", "")
        home_score = match_state.get("home_score", 0) or 0
        away_score = match_state.get("away_score", 0) or 0
        period = match_state.get("period", 0)
        if period > 0 or home_score > 0 or away_score > 0:
            return self.predict_live(match_state)
        return self.predict_prematch(
            match_state.get("home_team", ""),
            match_state.get("away_team", ""),
            match_date=match_state.get("match_date"),
        )

    def train(self, matches: list[dict], *, calibrate: bool = True) -> "BasketballModel":
        self.history = sort_matches(matches)
        self.elo.ratings.clear()
        self.elo.train_on_history(self.history)
        self.ratings = nba_team_ratings(self.history, tau_days=self.elo.tau_days)

        if self.history:
            totals = [m["home_score"] + m["away_score"] for m in self.history]
            margins = [m["home_score"] - m["away_score"] for m in self.history]
            self.league_avg_pts = float(np.mean(totals) / 2)
            self.sigma_total = float(max(8.0, np.std(totals)))
            self.sigma_margin = float(max(8.0, np.std(margins)))

        if calibrate:
            self._fit_calibrator()

        logger.info(
            "Modèle basketball entraîné: %d matchs, σ_marge=%.1f, σ_total=%.1f",
            len(self.history), self.sigma_margin, self.sigma_total,
        )
        return self

    def _fit_calibrator(self):
        matches = self.history
        if len(matches) < 40:
            return
        split = int(len(matches) * 0.75)
        shadow = BasketballModel(
            elo=EloRating(
                k_factor=self.elo.k_factor,
                tau_days=self.elo.tau_days,
                home_advantage=self.elo.home_advantage,
            )
        )
        shadow.train(matches[:split], calibrate=False)
        p, y = [], []
        for m in matches[split:]:
            pred = shadow.predict_prematch(
                m["home_team"], m["away_team"], m.get("match_date"), calibrate=False,
            )
            p.append(pred["home_win"])
            y.append(1.0 if m["home_score"] > m["away_score"] else 0.0)
        self.calibrator.fit(np.array(p), np.array(y))
