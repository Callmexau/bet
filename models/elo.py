# -*- coding: utf-8 -*-
"""
models/elo.py - Système de rating Elo pour le calcul des probabilités avant-match.

Le rating Elo est un système de classement relatif initialement créé pour les échecs,
adapté ici aux sports d'équipe. La probabilité de victoire est dérivée de la
différence de ratings via une fonction logistique.
"""

import math
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# ─────────────────────── Paramètres du modèle Elo ──────────────────────────

ELO_DEFAULT = 1500          # Rating initial pour toute nouvelle équipe
ELO_K_FACTOR = 32           # Sensibilité aux mises à jour (plus élevé = plus réactif)
ELO_HOME_ADVANTAGE = 100    # Bonus Elo pour l'équipe à domicile
ELO_SCALE = 400             # Échelle de conversion (standard pour les jeux)


class EloRating:
    """
    Système Elo pour équipes sportives.
    Maintient un dictionnaire de ratings mis à jour match par match.
    """

    def __init__(self, k_factor: float = ELO_K_FACTOR, home_advantage: float = ELO_HOME_ADVANTAGE):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings: dict[str, float] = defaultdict(lambda: ELO_DEFAULT)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calcule le score attendu pour l'équipe A contre l'équipe B.
        Retourne une valeur entre 0 et 1 (probabilité de victoire de A).
        """
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / ELO_SCALE))

    def win_probability(self, home_team: str, away_team: str) -> dict[str, float]:
        """
        Calcule la probabilité de victoire avant match (sans ajustement en direct).
        Retourne un dict avec home_win, draw, away_win.
        """
        home_rating = self.ratings[home_team] + self.home_advantage
        away_rating = self.ratings[away_team]

        # Probabilité brute de victoire à domicile
        p_home = self.expected_score(home_rating, away_rating)
        p_away = self.expected_score(away_rating, home_rating)

        # Pour le football, on réserve une part pour le nul
        # Estimation empirique : le nul représente ~25-28% des matchs en Europe
        # On utilise la formule de Dixon-Coles simplifiée
        draw_factor = 0.26  # Proportion historique de nuls en Europe
        adjusted_home = p_home * (1 - draw_factor)
        adjusted_draw = draw_factor
        adjusted_away = p_away * (1 - draw_factor)

        # Normalisation pour que la somme = 1
        total = adjusted_home + adjusted_draw + adjusted_away
        return {
            "home_win": round(adjusted_home / total, 4),
            "draw": round(adjusted_draw / total, 4),
            "away_win": round(adjusted_away / total, 4),
        }

    def update(self, home_team: str, away_team: str, result: str):
        """
        Met à jour les ratings après un match.
        result: 'H' (home win), 'D' (draw), 'A' (away win)
        """
        home_rating = self.ratings[home_team] + self.home_advantage
        away_rating = self.ratings[away_team]

        exp_home = self.expected_score(home_rating, away_rating)
        exp_away = 1 - exp_home

        # Scores réels selon le résultat
        if result == "H":
            score_home, score_away = 1.0, 0.0
        elif result == "D":
            score_home, score_away = 0.5, 0.5
        else:  # 'A'
            score_home, score_away = 0.0, 1.0

        # Mise à jour des ratings (sans le bonus domicile dans le storage)
        self.ratings[home_team] += self.k_factor * (score_home - exp_home)
        self.ratings[away_team] += self.k_factor * (score_away - exp_away)

    def train_on_history(self, matches: list[dict]) -> "EloRating":
        """
        Entraîne le modèle Elo sur un historique de matchs.
        Chaque match doit avoir : home_team, away_team, home_score, away_score.
        Retourne self pour le chaînage.
        """
        # Tri chronologique pour rejouer dans l'ordre
        sorted_matches = sorted(
            [m for m in matches if m.get("home_score") is not None and m.get("away_score") is not None],
            key=lambda x: x.get("match_date", "")
        )

        for match in sorted_matches:
            home = match["home_team"]
            away = match["away_team"]
            hs = match["home_score"]
            as_ = match["away_score"]

            if hs > as_:
                result = "H"
            elif hs < as_:
                result = "A"
            else:
                result = "D"

            self.update(home, away, result)

        logger.debug("Elo entraîné sur %d matchs. Équipes: %d", len(sorted_matches), len(self.ratings))
        return self

    def get_rating(self, team: str) -> float:
        """Retourne le rating Elo d'une équipe (défaut 1500)."""
        return self.ratings[team]

    def get_all_ratings(self) -> dict[str, float]:
        """Retourne le classement complet trié par rating décroissant."""
        return dict(sorted(self.ratings.items(), key=lambda x: x[1], reverse=True))
