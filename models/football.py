# -*- coding: utf-8 -*-
"""
models/football.py - Modèle de prédiction pour le football

Avant-match : Rating Elo + modèle de Poisson Dixon-Coles (1997)
En direct   : Ajustement bayésien selon score, minute et cartons rouges

Références :
- Dixon & Coles (1997) "Modelling Association Football Scores and Inefficiencies
  in the Football Betting Market" — correction τ pour les scores faibles (0-0, 1-0, etc.)
- Lee (1997) pour la modélisation Poisson des buts
"""

import math
import logging
import numpy as np
from scipy.stats import poisson
from models.elo import EloRating

logger = logging.getLogger(__name__)

# ─────────────────── Paramètres globaux ────────────────────────────────────

LEAGUE_AVG_GOALS_HOME = 1.55   # Buts/match à domicile (Europe, toutes compétitions)
LEAGUE_AVG_GOALS_AWAY = 1.15   # Buts/match à l'extérieur
MAX_GOALS = 10                  # Plafond pour le calcul Poisson
RED_CARD_PENALTY = 0.15        # -15% de λ par carton rouge pour l'équipe pénalisée

# Facteurs d'avantage domicile spécifiques par compétition
LEAGUE_HOME_FACTORS = {
    "PL": 1.12,   # Premier League
    "PD": 1.16,   # La Liga (avantage domicile fort en Espagne)
    "FL1": 1.14,  # Ligue 1
    "BL1": 1.15,  # Bundesliga (ligue prolifique)
    "SA": 1.13,   # Serie A
    "CL": 1.10,   # Champions League (équipes élites, écart réduit)
    "WC": 1.00,   # Coupe du Monde (terrain neutre par défaut)
    "CAN": 1.00,  # CAN (terrain neutre par défaut)
}

# Paramètre de sur-corrélation Dixon-Coles (ajusté empiriquement sur données européennes)
# ρ > 0 → les scores {0-0, 1-0, 0-1, 1-1} sont plus probables qu'un Poisson indépendant
RHO = 0.10


def _tau(home_goals: int, away_goals: int, lh: float, la: float, rho: float) -> float:
    """
    Facteur de correction τ de Dixon-Coles pour les scores ≤ 1-1.
    Corrige la sur/sous-représentation empirique des scores faibles.

    τ(0,0) = 1 − λH·λA·ρ
    τ(1,0) = 1 + λA·ρ
    τ(0,1) = 1 + λH·ρ
    τ(1,1) = 1 − ρ
    autres  = 1  (pas de correction)
    """
    if home_goals == 0 and away_goals == 0:
        return 1 - lh * la * rho
    elif home_goals == 1 and away_goals == 0:
        return 1 + la * rho
    elif home_goals == 0 and away_goals == 1:
        return 1 + lh * rho
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


class FootballModel:
    """
    Modèle de prédiction football combinant Elo et Dixon-Coles (1997).

    Flux de prédiction :
    1. Elo          → force relative des équipes → λ home, λ away
    2. Dixon-Coles  → P(score h-a) avec correction τ sur les scores ≤ 1-1
    3. Agrégation   → P(victoire home), P(nul), P(victoire away)
    4. Live         → buts restants proportionnels + malus cartons rouges
    """

    def __init__(self, elo: EloRating | None = None, rho: float = RHO):
        self.elo = elo or EloRating()
        self.rho = rho  # Paramètre de correction Dixon-Coles

    # ── Calcul des λ via Elo ───────────────────────────────────────────────

    def _expected_goals(self, home_team: str, away_team: str, comp_code: str = "") -> tuple[float, float]:
        """
        Convertit la différence Elo en buts attendus (λ) via une fonction puissance.
        Prend en compte l'avantage domicile spécifique à chaque championnat (ex: neutral en CAN/WC).
        """
        home_rating = self.elo.get_rating(home_team)
        away_rating = self.elo.get_rating(away_team)

        delta = (home_rating - away_rating) / 400.0
        strength_factor = math.pow(10, delta)

        home_factor = LEAGUE_HOME_FACTORS.get(comp_code, 1.12)
        base_h = LEAGUE_AVG_GOALS_HOME * (home_factor / 1.12)
        base_a = LEAGUE_AVG_GOALS_AWAY / (home_factor / 1.12)

        lambda_home = base_h * math.sqrt(strength_factor)
        lambda_away = base_a / math.sqrt(strength_factor)

        return max(0.3, min(lambda_home, 5.0)), max(0.3, min(lambda_away, 5.0))

    def _compute_alternative_markets(self, grid: dict[tuple[int, int], float], probs_1x2: dict) -> dict:
        """
        Calcule les probabilités des marchés alternatifs depuis la distribution de Poisson Dixon-Coles :
        - Over / Under 1.5, 2.5, 3.5 buts
        - Both Teams to Score (BTTS) Oui / Non
        - Double Chance (1X, 12, X2)
        """
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

    # ── Modèle Dixon-Coles ─────────────────────────────────────────────────

    def _dixon_coles_probs(
        self,
        lambda_home: float,
        lambda_away: float,
        rho: float | None = None,
    ) -> dict[str, float]:
        """
        Calcule P(V), P(N), P(D) via le modèle Dixon-Coles.

        Formule :
            P(H=h, A=a) = τ(h,a,λH,λA,ρ) × Poisson(h|λH) × Poisson(a|λA)

        La correction τ augmente légèrement P(0-0), P(1-0), P(0-1) et diminue
        P(1-1) pour coller aux statistiques empiriques réelles.
        """
        rho = rho if rho is not None else self.rho
        p_home_win = 0.0
        p_draw = 0.0
        p_away_win = 0.0

        for h in range(MAX_GOALS + 1):
            p_h = poisson.pmf(h, lambda_home)
            for a in range(MAX_GOALS + 1):
                p_a = poisson.pmf(a, lambda_away)
                # Correction Dixon-Coles pour les faibles scores
                tau = _tau(h, a, lambda_home, lambda_away, rho)
                joint = tau * p_h * p_a

                if h > a:
                    p_home_win += joint
                elif h == a:
                    p_draw += joint
                else:
                    p_away_win += joint

        total = p_home_win + p_draw + p_away_win
        if total <= 0:
            return {"home_win": 1/3, "draw": 1/3, "away_win": 1/3}

        return {
            "home_win": round(p_home_win / total, 4),
            "draw":     round(p_draw     / total, 4),
            "away_win": round(p_away_win / total, 4),
        }

    def _score_grid(
        self,
        lambda_home: float,
        lambda_away: float,
        max_goals: int = 6,
    ) -> dict[tuple[int, int], float]:
        """
        Retourne la grille de probabilités de scores (h, a) → P.
        Utile pour l'affichage des scores les plus probables.
        """
        grid = {}
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                p_h = poisson.pmf(h, lambda_home)
                p_a = poisson.pmf(a, lambda_away)
                tau = _tau(h, a, lambda_home, lambda_away, self.rho)
                grid[(h, a)] = tau * p_h * p_a
        # Normalisation
        total = sum(grid.values())
        if total > 0:
            grid = {k: round(v / total, 4) for k, v in grid.items()}
        return grid

    # ── Prédiction avant-match ─────────────────────────────────────────────

    def predict_prematch(self, home_team: str, away_team: str, comp_code: str = "") -> dict:
        """
        Prédiction avant le coup d'envoi via Elo + Dixon-Coles.

        Retourne :
            home_win, draw, away_win    — probabilités 1X2
            lambda_home, lambda_away    — buts attendus
            home_elo, away_elo          — ratings Elo
            top_scores                  — 5 scores les plus probables
            markets                     — marchés alternatifs (Over/Under, BTTS, Double Chance)
            model                       — "prematch_dc" (Dixon-Coles)
        """
        lh, la = self._expected_goals(home_team, away_team, comp_code=comp_code)
        probs = self._dixon_coles_probs(lh, la)

        # Grille de scores et marchés alternatifs
        grid = self._score_grid(lh, la)
        top5 = sorted(grid.items(), key=lambda x: x[1], reverse=True)[:5]
        top_scores = [{"score": f"{h}-{a}", "prob": round(p * 100, 1)} for (h, a), p in top5]
        alt_markets = self._compute_alternative_markets(grid, probs)

        return {
            **probs,
            "lambda_home": round(lh, 2),
            "lambda_away": round(la, 2),
            "home_elo": round(self.elo.get_rating(home_team), 0),
            "away_elo": round(self.elo.get_rating(away_team), 0),
            "top_scores": top_scores,
            "markets": alt_markets,
            "model": "prematch_dc",
        }

    # ── Ajustement en direct ───────────────────────────────────────────────

    def predict_live(self, match_state: dict) -> dict:
        """
        Ajuste les probabilités en temps réel.
        Les λ restants sont proratisés selon le temps restant + malus cartons rouges.
        On recalcule la distribution de Poisson depuis le score actuel.
        """
        home_team  = match_state.get("home_team", "")
        away_team  = match_state.get("away_team", "")
        home_score = match_state.get("home_score") or 0
        away_score = match_state.get("away_score") or 0
        minute     = match_state.get("minute") or 0
        red_home   = match_state.get("red_cards_home", 0)
        red_away   = match_state.get("red_cards_away", 0)
        comp_code  = match_state.get("competition_code", "")

        if minute == 0 or (home_score == 0 and away_score == 0 and minute < 5):
            result = self.predict_prematch(home_team, away_team, comp_code=comp_code)
            result["model"] = "live_prematch"
            return result

        time_remaining_frac = max(0.0, (90 - minute) / 90.0)
        lh_full, la_full = self._expected_goals(home_team, away_team, comp_code=comp_code)

        lh_rem = lh_full * time_remaining_frac * ((1 - RED_CARD_PENALTY) ** red_home)
        la_rem = la_full * time_remaining_frac * ((1 - RED_CARD_PENALTY) ** red_away)

        p_home_win = 0.0
        p_draw     = 0.0
        p_away_win = 0.0
        live_grid  = {}

        for h_add in range(MAX_GOALS + 1):
            p_h = poisson.pmf(h_add, max(0.001, lh_rem))
            for a_add in range(MAX_GOALS + 1):
                p_a   = poisson.pmf(a_add, max(0.001, la_rem))
                tau   = _tau(h_add, a_add, lh_rem, la_rem, self.rho)
                joint = tau * p_h * p_a

                fh = home_score + h_add
                fa = away_score + a_add
                live_grid[(fh, fa)] = live_grid.get((fh, fa), 0.0) + joint

                if fh > fa:
                    p_home_win += joint
                elif fh == fa:
                    p_draw += joint
                else:
                    p_away_win += joint

        total = p_home_win + p_draw + p_away_win or 1.0
        probs_1x2 = {
            "home_win": round(p_home_win / total, 4),
            "draw":     round(p_draw     / total, 4),
            "away_win": round(p_away_win / total, 4),
        }

        # Normalisation de la grille en direct pour marchés alternatifs
        tot_grid = sum(live_grid.values()) or 1.0
        norm_grid = {k: v / tot_grid for k, v in live_grid.items()}
        alt_markets = self._compute_alternative_markets(norm_grid, probs_1x2)

        return {
            **probs_1x2,
            "lambda_home_remaining":  round(lh_rem, 2),
            "lambda_away_remaining":  round(la_rem, 2),
            "home_score":             home_score,
            "away_score":             away_score,
            "minute":                 minute,
            "markets":                alt_markets,
            "model":                  "live_dc",
        }

    # ── Point d'entrée unifié ──────────────────────────────────────────────

    def predict(self, match_state: dict) -> dict:
        """Sélectionne automatiquement prematch ou live selon le statut."""
        status = match_state.get("status", "SCHEDULED")
        minute = match_state.get("minute") or 0

        if status in ("IN_PLAY", "PAUSED") or minute > 0:
            return self.predict_live(match_state)
        return self.predict_prematch(
            match_state.get("home_team", ""),
            match_state.get("away_team", ""),
            comp_code=match_state.get("competition_code", ""),
        )

    def compute_ev(self, probs: dict, odds: dict) -> dict:
        """
        Calcule la valeur espérée (Expected Value) pour chaque issue.
        EV = prob_modèle × cote_décimale − 1

        Positif → la cote bookmaker sous-estime notre probabilité → value bet !

        Args:
            probs: {"home_win": 0.52, "draw": 0.28, "away_win": 0.20}
            odds:  {"home_win": 2.10, "draw": 3.40, "away_win": 4.50}

        Returns:
            {"home_win": 0.092, "draw": -0.048, "away_win": -0.10, "best": "home_win"}
        """
        ev = {}
        for outcome in ("home_win", "draw", "away_win"):
            p  = probs.get(outcome, 0)
            o  = odds.get(outcome, 0)
            ev[outcome] = round(p * o - 1, 3) if o > 0 else None

        valid = {k: v for k, v in ev.items() if v is not None}
        ev["best"] = max(valid, key=valid.get) if valid else None
        ev["best_ev"] = max(valid.values()) if valid else None
        return ev

    # ── Entraînement ──────────────────────────────────────────────────────

    def train(self, matches: list[dict]) -> "FootballModel":
        """Entraîne le modèle Elo sur un historique de matchs."""
        self.elo.train_on_history(matches)
        logger.info(
            "Modèle football (Dixon-Coles) entraîné sur %d équipes.",
            len(self.elo.get_all_ratings()),
        )
        return self
