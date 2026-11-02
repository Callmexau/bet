# -*- coding: utf-8 -*-
"""
models/football.py - Modèle de prédiction football

Avant-match :
  - Elo avec décroissance temporelle
  - Attaque / défense style Maher-Dixon-Coles (λ par équipe et par ligue)
  - Forme courte, jours de repos
  - ρ et avantage domicile estimés sur l'historique
  - Calibration par température

En direct :
  - Intensité de buts non linéaire (plus élevée en fin de match)
  - Temps additionnel
  - Malus carton rouge en buts attendus restants
  - Poisson restant indépendant (pas de τ Dixon-Coles sur l'incrément)
"""

from __future__ import annotations

import math
import logging
import numpy as np
from scipy.stats import poisson

from models.elo import EloRating
from models.calibration import TemperatureCalibrator
from models.features import (
    competition_code,
    exp_decay_weight,
    form_multiplier,
    is_neutral_venue,
    latest_date,
    league_goal_averages,
    parse_date,
    recent_form,
    rest_days,
    rest_multiplier,
    sort_matches,
)

logger = logging.getLogger(__name__)

LEAGUE_AVG_GOALS_HOME = 1.45
LEAGUE_AVG_GOALS_AWAY = 1.15
MAX_GOALS = 10
RED_CARD_GOALS = 0.38  # buts attendus perdus sur 90 min par carton rouge
RHO_DEFAULT = 0.10
SHRINK_K = 8.0

# Intensité relative de buts par minute (0-94 inclut ~5 min d'arrêts de jeu)
_SCORING_RATES = np.ones(95, dtype=float)
_SCORING_RATES[0:15] = 0.85
_SCORING_RATES[15:30] = 0.92
_SCORING_RATES[30:45] = 1.05
_SCORING_RATES[45:60] = 0.98
_SCORING_RATES[60:75] = 1.08
_SCORING_RATES[75:90] = 1.28
_SCORING_RATES[90:95] = 1.15
_SCORING_TOTAL = float(_SCORING_RATES.sum())


def remaining_goal_fraction(minute: float) -> float:
    """Fraction de buts encore attendus après `minute` (arrêts de jeu inclus)."""
    m = int(min(max(minute, 0), 94))
    return float(_SCORING_RATES[m:].sum() / _SCORING_TOTAL)


def _tau(home_goals: int, away_goals: int, lh: float, la: float, rho: float) -> float:
    if home_goals == 0 and away_goals == 0:
        return max(0.01, 1 - lh * la * rho)
    if home_goals == 1 and away_goals == 0:
        return 1 + la * rho
    if home_goals == 0 and away_goals == 1:
        return 1 + lh * rho
    if home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


class FootballModel:
    """
    Elo + attaque/défense Poisson + Dixon-Coles + calibration.
    """

    def __init__(self, elo: EloRating | None = None, rho: float = RHO_DEFAULT):
        self.elo = elo or EloRating()
        self.rho = rho
        self.history: list[dict] = []
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.league_avgs: dict[str, tuple[float, float]] = {
            "_global": (LEAGUE_AVG_GOALS_HOME, LEAGUE_AVG_GOALS_AWAY)
        }
        self.games_played: dict[str, int] = {}
        self.calibrator = TemperatureCalibrator()

    # ── λ ──────────────────────────────────────────────────────────────────

    def _league_means(self, match_or_code) -> tuple[float, float]:
        if isinstance(match_or_code, dict):
            code = competition_code(match_or_code) or "_global"
            if is_neutral_venue(match_or_code):
                gh, ga = self.league_avgs.get(code, self.league_avgs["_global"])
                mid = 0.5 * (gh + ga)
                return mid, mid
        else:
            code = match_or_code or "_global"
        return self.league_avgs.get(code, self.league_avgs.get("_global", (1.45, 1.15)))

    def _ad_lambdas(self, home: str, away: str, comp_code: str, match: dict | None) -> tuple[float, float]:
        mu_h, mu_a = self._league_means(match if match else comp_code)
        att_h = self.attack.get(home, 1.0)
        att_a = self.attack.get(away, 1.0)
        def_h = self.defense.get(home, 1.0)
        def_a = self.defense.get(away, 1.0)
        if match and is_neutral_venue(match):
            mid = 0.5 * (mu_h + mu_a)
            mu_h, mu_a = mid, mid
        return mu_h * att_h * def_a, mu_a * att_a * def_h

    def _elo_lambdas(self, home: str, away: str, mu_h: float, mu_a: float) -> tuple[float, float]:
        delta = (self.elo.get_rating(home) - self.elo.get_rating(away)) / 400.0
        strength = math.pow(10, delta)
        return mu_h * math.sqrt(strength), mu_a / math.sqrt(strength)

    def _mix_weight(self, home: str, away: str) -> float:
        n = min(self.games_played.get(home, 0), self.games_played.get(away, 0))
        return n / (n + SHRINK_K)

    def _context_multipliers(self, team: str, match_date) -> float:
        rest = rest_days(self.history, team, match_date)
        form = recent_form(self.history, team, match_date, n=6)
        return rest_multiplier(rest, sport="football") * form_multiplier(form, scale=0.03)

    def _expected_goals(
        self,
        home_team: str,
        away_team: str,
        comp_code: str = "",
        match_date=None,
        match: dict | None = None,
    ) -> tuple[float, float]:
        mu_h, mu_a = self._league_means(match if match else comp_code)
        lh_ad, la_ad = self._ad_lambdas(home_team, away_team, comp_code, match)
        lh_elo, la_elo = self._elo_lambdas(home_team, away_team, mu_h, mu_a)
        w = self._mix_weight(home_team, away_team)
        lh = w * lh_ad + (1 - w) * lh_elo
        la = w * la_ad + (1 - w) * la_elo

        as_of = match_date or (match.get("match_date") if match else None)
        lh *= self._context_multipliers(home_team, as_of)
        la *= self._context_multipliers(away_team, as_of)
        return max(0.25, min(lh, 4.8)), max(0.25, min(la, 4.8))

    def _compute_alternative_markets(self, grid: dict[tuple[int, int], float], probs_1x2: dict) -> dict:
        over15 = sum(p for (h, a), p in grid.items() if (h + a) > 1.5)
        over25 = sum(p for (h, a), p in grid.items() if (h + a) > 2.5)
        over35 = sum(p for (h, a), p in grid.items() if (h + a) > 3.5)
        btts_yes = sum(p for (h, a), p in grid.items() if h >= 1 and a >= 1)

        hw = probs_1x2.get("home_win", 0)
        dr = probs_1x2.get("draw", 0)
        aw = probs_1x2.get("away_win", 0)

        return {
            "over_1_5":  float(round(over15, 4)),
            "under_1_5": float(round(max(0.0, 1.0 - over15), 4)),
            "over_2_5":  float(round(over25, 4)),
            "under_2_5": float(round(max(0.0, 1.0 - over25), 4)),
            "over_3_5":  float(round(over35, 4)),
            "under_3_5": float(round(max(0.0, 1.0 - over35), 4)),
            "btts_yes":  float(round(btts_yes, 4)),
            "btts_no":   float(round(max(0.0, 1.0 - btts_yes), 4)),
            "dc_1x":     float(round(min(1.0, hw + dr), 4)),
            "dc_x2":     float(round(min(1.0, dr + aw), 4)),
            "dc_12":     float(round(min(1.0, hw + aw), 4)),
        }

    def _dixon_coles_probs(self, lambda_home: float, lambda_away: float, rho: float | None = None) -> dict[str, float]:
        rho = rho if rho is not None else self.rho
        p_home_win = p_draw = p_away_win = 0.0
        for h in range(MAX_GOALS + 1):
            p_h = poisson.pmf(h, lambda_home)
            for a in range(MAX_GOALS + 1):
                joint = _tau(h, a, lambda_home, lambda_away, rho) * p_h * poisson.pmf(a, lambda_away)
                if h > a:
                    p_home_win += joint
                elif h == a:
                    p_draw += joint
                else:
                    p_away_win += joint
        total = p_home_win + p_draw + p_away_win
        if total <= 0:
            return {"home_win": 1 / 3, "draw": 1 / 3, "away_win": 1 / 3}
        return {
            "home_win": p_home_win / total,
            "draw":     p_draw / total,
            "away_win": p_away_win / total,
        }

    def _score_grid(self, lambda_home: float, lambda_away: float, max_goals: int = 6, use_tau: bool = True) -> dict:
        grid = {}
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                tau = _tau(h, a, lambda_home, lambda_away, self.rho) if use_tau else 1.0
                grid[(h, a)] = tau * poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
        total = sum(grid.values())
        if total > 0:
            grid = {k: v / total for k, v in grid.items()}
        return grid

    def _apply_calibration(self, probs: dict) -> dict:
        h, d, a = self.calibrator.transform(probs["home_win"], probs["draw"], probs["away_win"])
        return {
            **probs,
            "home_win": round(h, 4),
            "draw": round(d, 4),
            "away_win": round(a, 4),
        }

    # ── Prédiction avant-match ─────────────────────────────────────────────

    def predict_prematch(
        self,
        home_team: str,
        away_team: str,
        comp_code: str = "",
        match_date=None,
        match: dict | None = None,
        calibrate: bool = True,
    ) -> dict:
        lh, la = self._expected_goals(home_team, away_team, comp_code, match_date, match)
        probs = self._dixon_coles_probs(lh, la)
        grid = self._score_grid(lh, la)
        top5 = sorted(grid.items(), key=lambda x: x[1], reverse=True)[:5]
        top_scores = [{"score": f"{h}-{a}", "prob": round(p * 100, 1)} for (h, a), p in top5]
        out = {
            **probs,
            "lambda_home": round(lh, 2),
            "lambda_away": round(la, 2),
            "home_elo": round(self.elo.get_rating(home_team), 0),
            "away_elo": round(self.elo.get_rating(away_team), 0),
            "top_scores": top_scores,
            "markets": {},
            "model": "prematch_dc_ad",
        }
        if calibrate:
            out = self._apply_calibration(out)
        out["markets"] = self._compute_alternative_markets(grid, out)
        return out

    # ── Ajustement en direct ───────────────────────────────────────────────

    def predict_live(self, match_state: dict) -> dict:
        home_team = match_state.get("home_team", "")
        away_team = match_state.get("away_team", "")
        home_score = match_state.get("home_score") or 0
        away_score = match_state.get("away_score") or 0
        minute = match_state.get("minute") or 0
        red_home = match_state.get("red_cards_home", 0) or 0
        red_away = match_state.get("red_cards_away", 0) or 0
        comp_code = match_state.get("competition_code", "") or competition_code(match_state)

        if minute == 0 or (home_score == 0 and away_score == 0 and minute < 5):
            result = self.predict_prematch(
                home_team, away_team, comp_code=comp_code,
                match_date=match_state.get("match_date"), match=match_state,
            )
            result["model"] = "live_prematch"
            return result

        rem_frac = remaining_goal_fraction(float(minute))
        lh_full, la_full = self._expected_goals(
            home_team, away_team, comp_code,
            match_state.get("match_date"), match_state,
        )
        lh_rem = max(0.04, lh_full * rem_frac - RED_CARD_GOALS * rem_frac * red_home)
        la_rem = max(0.04, la_full * rem_frac - RED_CARD_GOALS * rem_frac * red_away)

        p_home_win = p_draw = p_away_win = 0.0
        live_grid: dict[tuple[int, int], float] = {}

        for h_add in range(MAX_GOALS + 1):
            p_h = poisson.pmf(h_add, lh_rem)
            for a_add in range(MAX_GOALS + 1):
                joint = p_h * poisson.pmf(a_add, la_rem)
                fh, fa = home_score + h_add, away_score + a_add
                live_grid[(fh, fa)] = live_grid.get((fh, fa), 0.0) + joint
                if fh > fa:
                    p_home_win += joint
                elif fh == fa:
                    p_draw += joint
                else:
                    p_away_win += joint

        total = p_home_win + p_draw + p_away_win or 1.0
        probs_1x2 = {
            "home_win": p_home_win / total,
            "draw":     p_draw / total,
            "away_win": p_away_win / total,
        }
        probs_1x2 = self._apply_calibration(probs_1x2)

        tot_grid = sum(live_grid.values()) or 1.0
        norm_grid = {k: v / tot_grid for k, v in live_grid.items()}
        alt_markets = self._compute_alternative_markets(norm_grid, probs_1x2)

        return {
            **probs_1x2,
            "lambda_home_remaining": round(lh_rem, 2),
            "lambda_away_remaining": round(la_rem, 2),
            "home_score": home_score,
            "away_score": away_score,
            "minute": minute,
            "markets": alt_markets,
            "model": "live_intensity",
        }

    def predict(self, match_state: dict) -> dict:
        status = match_state.get("status", "SCHEDULED")
        minute = match_state.get("minute") or 0
        if status in ("IN_PLAY", "PAUSED") or minute > 0:
            return self.predict_live(match_state)
        return self.predict_prematch(
            match_state.get("home_team", ""),
            match_state.get("away_team", ""),
            comp_code=match_state.get("competition_code", "") or competition_code(match_state),
            match_date=match_state.get("match_date"),
            match=match_state,
        )

    def compute_ev(self, probs: dict, odds: dict) -> dict:
        ev = {}
        for outcome in ("home_win", "draw", "away_win"):
            p = probs.get(outcome, 0)
            o = odds.get(outcome, 0)
            ev[outcome] = round(p * o - 1, 3) if o > 0 else None
        valid = {k: v for k, v in ev.items() if v is not None}
        ev["best"] = max(valid, key=valid.get) if valid else None
        ev["best_ev"] = max(valid.values()) if valid else None
        return ev

    # ── Entraînement ──────────────────────────────────────────────────────

    def train(self, matches: list[dict], *, tune: bool = False, calibrate: bool = True) -> "FootballModel":
        self.history = sort_matches(matches)
        if tune:
            self._tune_elo_params(self.history)

        self.elo.ratings.clear()
        self.elo.train_on_history(self.history)
        self._fit_league_avgs()
        self._fit_attack_defense()
        self._fit_rho()
        if calibrate:
            self._fit_calibrator()

        logger.info(
            "Modèle football entraîné: %d matchs, %d équipes, ρ=%.3f, K=%s, tau=%s, T=%s",
            len(self.history), len(self.elo.get_all_ratings()), self.rho,
            self.elo.k_factor, self.elo.tau_days,
            round(self.calibrator.T, 3) if self.calibrator.fitted else 1.0,
        )
        return self

    def _fit_league_avgs(self):
        self.league_avgs = league_goal_averages(self.history, min_n=8)

    def _fit_attack_defense(self, n_iter: int = 28):
        matches = self.history
        teams: set[str] = set()
        self.games_played = {}
        for m in matches:
            teams.add(m["home_team"])
            teams.add(m["away_team"])
            self.games_played[m["home_team"]] = self.games_played.get(m["home_team"], 0) + 1
            self.games_played[m["away_team"]] = self.games_played.get(m["away_team"], 0) + 1

        self.attack = {t: 1.0 for t in teams}
        self.defense = {t: 1.0 for t in teams}
        if not matches:
            return

        ref = latest_date(matches)
        tau = self.elo.tau_days

        for _ in range(n_iter):
            scored = {t: 0.0 for t in teams}
            exp_att = {t: 0.0 for t in teams}
            conc = {t: 0.0 for t in teams}
            exp_def = {t: 0.0 for t in teams}

            for m in matches:
                h, a = m["home_team"], m["away_team"]
                hs, aws = float(m["home_score"]), float(m["away_score"])
                mu_h, mu_a = self._league_means(m)
                if is_neutral_venue(m):
                    mid = 0.5 * (mu_h + mu_a)
                    mu_h, mu_a = mid, mid
                d = parse_date(m.get("match_date"))
                w = exp_decay_weight((ref - d).days if ref and d else 0, tau)

                scored[h] += w * hs
                scored[a] += w * aws
                conc[h] += w * aws
                conc[a] += w * hs
                exp_att[h] += w * (mu_h * self.defense[a])
                exp_att[a] += w * (mu_a * self.defense[h])
                exp_def[a] += w * (mu_h * self.attack[h])
                exp_def[h] += w * (mu_a * self.attack[a])

            for t in teams:
                self.attack[t] = (scored[t] + SHRINK_K) / (exp_att[t] + SHRINK_K)
                self.defense[t] = (conc[t] + SHRINK_K) / (exp_def[t] + SHRINK_K)
                self.attack[t] = max(0.45, min(2.4, self.attack[t]))
                self.defense[t] = max(0.45, min(2.4, self.defense[t]))

            mean_att = sum(self.attack.values()) / len(self.attack)
            mean_def = sum(self.defense.values()) / len(self.defense)
            for t in teams:
                self.attack[t] /= mean_att
                self.defense[t] /= mean_def

    def _match_nll(self, m: dict, rho: float) -> float:
        lh, la = self._expected_goals(
            m["home_team"], m["away_team"],
            competition_code(m), m.get("match_date"), m,
        )
        hs, aws = int(m["home_score"]), int(m["away_score"])
        if hs > MAX_GOALS:
            hs = MAX_GOALS
        if aws > MAX_GOALS:
            aws = MAX_GOALS
        p = _tau(hs, aws, lh, la, rho) * poisson.pmf(hs, lh) * poisson.pmf(aws, la)
        return -math.log(max(p, 1e-12))

    def _fit_rho(self):
        matches = self.history[-250:]
        if len(matches) < 20:
            return
        best_rho, best_nll = self.rho, float("inf")
        for rho in (0.0, 0.04, 0.08, 0.10, 0.13, 0.16, 0.20):
            nll = sum(self._match_nll(m, rho) for m in matches)
            if nll < best_nll:
                best_nll, best_rho = nll, rho
        self.rho = best_rho
        logger.info("ρ Dixon-Coles ajusté: %.2f (nll=%.1f)", self.rho, best_nll)

    def _tune_elo_params(self, matches: list[dict]):
        if len(matches) < 50:
            return
        split = int(len(matches) * 0.72)
        train, test = matches[:split], matches[split:]
        if len(test) < 12:
            return

        best = (self.elo.k_factor, self.elo.tau_days, self.elo.home_advantage)
        best_bs = 1e9
        for k in (18, 24, 32):
            for tau in (90, 150):
                for adv in (70, 100):
                    elo = EloRating(k_factor=k, tau_days=tau, home_advantage=adv)
                    elo.train_on_history(train)
                    bs_acc = []
                    for m in test:
                        tmp = FootballModel(elo=elo, rho=self.rho)
                        tmp.history = train
                        tmp.league_avgs = league_goal_averages(train)
                        tmp.attack, tmp.defense = {}, {}
                        p = tmp._dixon_coles_probs(*tmp._elo_lambdas(
                            m["home_team"], m["away_team"], *tmp._league_means(m),
                        ))
                        y = [0, 0, 0]
                        hs, aws = m["home_score"], m["away_score"]
                        y[0 if hs > aws else (2 if hs < aws else 1)] = 1
                        pred = [p["home_win"], p["draw"], p["away_win"]]
                        bs_acc.append(sum((a - b) ** 2 for a, b in zip(y, pred)) / 3)
                    bs = sum(bs_acc) / len(bs_acc)
                    if bs < best_bs:
                        best_bs = bs
                        best = (k, tau, adv)

        self.elo.k_factor, self.elo.tau_days, self.elo.home_advantage = best
        logger.info(
            "Hyperparams Elo: K=%s tau=%s home_adv=%s (Brier=%.4f)",
            *best, best_bs,
        )

    def _fit_calibrator(self):
        matches = self.history
        if len(matches) < 40:
            return
        split = int(len(matches) * 0.75)
        shadow = FootballModel(
            elo=EloRating(
                k_factor=self.elo.k_factor,
                tau_days=self.elo.tau_days,
                home_advantage=self.elo.home_advantage,
            ),
            rho=self.rho,
        )
        shadow.train(matches[:split], tune=False, calibrate=False)
        probs, y = [], []
        for m in matches[split:]:
            p = shadow.predict_prematch(
                m["home_team"], m["away_team"],
                competition_code(m), m.get("match_date"), m,
                calibrate=False,
            )
            probs.append([p["home_win"], p["draw"], p["away_win"]])
            hs, aws = m["home_score"], m["away_score"]
            y.append("H" if hs > aws else ("A" if hs < aws else "D"))
        self.calibrator.fit(np.array(probs), y)
