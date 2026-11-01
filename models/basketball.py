# -*- coding: utf-8 -*-
"""
models/basketball.py - Modèle de prédiction basketball (NBA)

Avant-match : Rating Elo adapté au basketball
En direct   : Approximation par loi normale (Berry & Wood, 2004)
              → P(victoire) ≈ Φ((score_diff) / σ_restant)
              où σ_restant décroît avec le temps restant

Logique :
- En basketball, l'écart de score prédit fortement le résultat final
- L'incertitude (σ) diminue à mesure que le temps passe
- On combine la prédiction Elo (avant-match) avec l'ajustement en direct
"""

import math
import logging
from scipy.stats import norm
from models.elo import EloRating

logger = logging.getLogger(__name__)

# ─────────────────── Paramètres du modèle Basketball ───────────────────────

# Écart-type empirique des scores NBA en fin de match (en points)
# Source : analyse historique NBA, ~10-12 pts en début de match
SIGMA_PER_MINUTE_NBA = 2.5    # Points d'incertitude par minute restante (approx)
TOTAL_GAME_MINUTES = 48       # Durée standard d'un match NBA (4 × 12 min)

# Avantage domicile NBA (en points)
NBA_HOME_ADVANTAGE_POINTS = 3.5

# Facteur de conversion Elo → points d'avantage
# Une différence de 100 Elo ≈ 3 points d'avantage
ELO_TO_POINTS = 100 / 3.0


class BasketballModel:
    """
    Modèle de prédiction basketball.

    La probabilité de victoire en cours de match est calculée via
    l'approximation normale sur l'écart de score attendu en fin de match.

    P(home_win) = Φ( (expected_margin + current_margin) / σ_remaining )
    """

    def __init__(self, elo: EloRating | None = None):
        # K-factor plus élevé pour le basket (résultats plus volatils)
        self.elo = elo or EloRating(k_factor=20, home_advantage=NBA_HOME_ADVANTAGE_POINTS * ELO_TO_POINTS)

    # ── Utilitaires ────────────────────────────────────────────────────────

    def _elo_to_point_spread(self, home_team: str, away_team: str) -> float:
        """
        Convertit la différence Elo en avantage de points attendu pour home.
        Positif = home favori, négatif = away favori.
        """
        home_elo = self.elo.get_rating(home_team) + self.elo.home_advantage
        away_elo = self.elo.get_rating(away_team)
        elo_diff = home_elo - away_elo
        return elo_diff / ELO_TO_POINTS

    def _minutes_remaining(self, match_state: dict) -> float:
        """Calcule les minutes restantes dans le match."""
        time_remaining_secs = match_state.get("time_remaining_secs")
        if time_remaining_secs is not None:
            return time_remaining_secs / 60.0

        # Fallback : depuis period + time string
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

    # ── Prédiction avant-match ─────────────────────────────────────────────

    def predict_prematch(self, home_team: str, away_team: str) -> dict:
        """
        Prédiction avant le début du match via Elo.
        Pas de nul en basketball → seulement home_win / away_win.
        """
        point_spread = self._elo_to_point_spread(home_team, away_team)
        sigma = SIGMA_PER_MINUTE_NBA * math.sqrt(TOTAL_GAME_MINUTES)

        # Probabilité que home gagne (score_diff final > 0)
        p_home = norm.cdf(point_spread / sigma)
        p_away = 1 - p_home

        return {
            "home_win": round(p_home, 4),
            "draw": 0.0,       # Pas de nul en basketball
            "away_win": round(p_away, 4),
            "point_spread": round(point_spread, 1),
            "home_elo": round(self.elo.get_rating(home_team), 0),
            "away_elo": round(self.elo.get_rating(away_team), 0),
            "model": "prematch",
        }

    # ── Ajustement en direct ───────────────────────────────────────────────

    def predict_live(self, match_state: dict) -> dict:
        """
        Ajuste les probabilités en cours de match.

        Formule :
            margin_actuelle = home_score - away_score
            margin_attendue = point_spread_elo
            σ_restant       = SIGMA × √(minutes_restantes)

            z = (margin_attendue + margin_actuelle) / σ_restant
            P(home_win) = Φ(z)

        Plus le temps est court et l'écart grand, plus la probabilité
        tend vers 0 ou 1.
        """
        home_team = match_state.get("home_team", "")
        away_team = match_state.get("away_team", "")
        home_score = match_state.get("home_score", 0) or 0
        away_score = match_state.get("away_score", 0) or 0

        minutes_remaining = self._minutes_remaining(match_state)
        current_margin = home_score - away_score

        # Si le match est terminé
        if minutes_remaining <= 0:
            if current_margin > 0:
                return {"home_win": 1.0, "draw": 0.0, "away_win": 0.0, "model": "final"}
            else:
                return {"home_win": 0.0, "draw": 0.0, "away_win": 1.0, "model": "final"}

        # Avantage Elo en points (marge attendue sans tenir compte du score actuel)
        expected_margin = self._elo_to_point_spread(home_team, away_team)

        # Incertitude restante : σ ∝ √(minutes_restantes)
        # (marche aléatoire dans l'espace des points)
        sigma_remaining = SIGMA_PER_MINUTE_NBA * math.sqrt(max(0.1, minutes_remaining))

        # Score z combiné : avantage structurel + avantage de score actuel
        z = (expected_margin + current_margin) / sigma_remaining

        p_home = float(norm.cdf(z))
        p_away = 1 - p_home

        # Clamp pour éviter les certitudes absolues avant la fin
        p_home = max(0.01, min(0.99, p_home))
        p_away = max(0.01, min(0.99, p_away))

        return {
            "home_win": round(p_home, 4),
            "draw": 0.0,
            "away_win": round(p_away, 4),
            "current_margin": current_margin,
            "expected_margin": round(expected_margin, 1),
            "minutes_remaining": round(minutes_remaining, 1),
            "sigma_remaining": round(sigma_remaining, 2),
            "home_score": home_score,
            "away_score": away_score,
            "model": "live",
        }

    def predict(self, match_state: dict) -> dict:
        """
        Point d'entrée unifié.
        Détecte automatiquement si le match est en cours ou à venir.
        """
        status = match_state.get("status", "")
        home_score = match_state.get("home_score", 0) or 0
        away_score = match_state.get("away_score", 0) or 0
        period = match_state.get("period", 0)

        # Match commencé si au moins un point marqué ou period > 0
        if period > 0 or home_score > 0 or away_score > 0:
            return self.predict_live(match_state)
        else:
            return self.predict_prematch(
                match_state.get("home_team", ""),
                match_state.get("away_team", ""),
            )

    def train(self, matches: list[dict]) -> "BasketballModel":
        """
        Entraîne le modèle Elo sur un historique de matchs NBA.
        """
        self.elo.train_on_history(matches)
        logger.info(
            "Modèle basketball entraîné. %d équipes référencées.",
            len(self.elo.get_all_ratings())
        )
        return self
