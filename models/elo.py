# -*- coding: utf-8 -*-
"""
models/elo.py - Système de rating Elo pour le calcul des probabilités avant-match.

Le rating Elo est un système de classement relatif initialement créé pour les échecs,
adapté ici aux sports d'équipe. La pondération des matchs d'entraînement est
exponentielle dans le temps (les résultats récents pèsent plus).
"""

import math
import logging
from collections import defaultdict

from models.features import parse_date, latest_date, exp_decay_weight, is_neutral_venue

logger = logging.getLogger(__name__)

ELO_DEFAULT = 1500
ELO_K_FACTOR = 24
ELO_HOME_ADVANTAGE = 80
ELO_SCALE = 400
ELO_TAU_DAYS = 120


class EloRating:
    """
    Système Elo pour équipes sportives.
    Maintient un dictionnaire de ratings mis à jour match par match.
    """

    def __init__(
        self,
        k_factor: float = ELO_K_FACTOR,
        home_advantage: float = ELO_HOME_ADVANTAGE,
        tau_days: float = ELO_TAU_DAYS,
    ):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.tau_days = tau_days
        self.ratings: dict[str, float] = defaultdict(lambda: ELO_DEFAULT)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / ELO_SCALE))

    def win_probability(self, home_team: str, away_team: str) -> dict[str, float]:
        home_rating = self.ratings[home_team] + self.home_advantage
        away_rating = self.ratings[away_team]

        p_home = self.expected_score(home_rating, away_rating)
        p_away = self.expected_score(away_rating, home_rating)

        draw_factor = 0.26
        adjusted_home = p_home * (1 - draw_factor)
        adjusted_draw = draw_factor
        adjusted_away = p_away * (1 - draw_factor)

        total = adjusted_home + adjusted_draw + adjusted_away
        return {
            "home_win": round(adjusted_home / total, 4),
            "draw": round(adjusted_draw / total, 4),
            "away_win": round(adjusted_away / total, 4),
        }

    def update(
        self,
        home_team: str,
        away_team: str,
        result: str,
        home_score: int = 0,
        away_score: int = 0,
        weight: float = 1.0,
        home_advantage: float | None = None,
    ):
        adv = home_advantage if home_advantage is not None else self.home_advantage
        home_rating = self.ratings[home_team] + adv
        away_rating = self.ratings[away_team]

        exp_home = self.expected_score(home_rating, away_rating)
        exp_away = 1 - exp_home

        if result == "H":
            score_home, score_away = 1.0, 0.0
        elif result == "D":
            score_home, score_away = 0.5, 0.5
        else:
            score_home, score_away = 0.0, 1.0

        diff = abs(home_score - away_score)
        if diff <= 1:
            margin_mult = 1.0
        elif diff == 2:
            margin_mult = 1.5
        else:
            margin_mult = (11.0 + diff) / 8.0

        k_effective = self.k_factor * margin_mult * max(0.15, min(weight, 1.5))

        self.ratings[home_team] += k_effective * (score_home - exp_home)
        self.ratings[away_team] += k_effective * (score_away - exp_away)

    def train_on_history(self, matches: list[dict]) -> "EloRating":
        """
        Entraîne l'Elo dans l'ordre chronologique.
        Poids = exp(-jours / tau) par rapport au match le plus récent.
        """
        from models.features import sort_matches

        sorted_matches = sort_matches(matches)
        ref = latest_date(sorted_matches)

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

            d = parse_date(match.get("match_date"))
            days_ago = (ref - d).days if ref and d else 0
            recency_weight = exp_decay_weight(days_ago, self.tau_days)

            adv = 0.0 if is_neutral_venue(match) else self.home_advantage

            self.update(
                home, away, result,
                home_score=hs, away_score=as_,
                weight=recency_weight,
                home_advantage=adv,
            )

        logger.debug(
            "Elo entraîné sur %d matchs (tau=%s j, K=%s). Équipes: %d",
            len(sorted_matches), self.tau_days, self.k_factor, len(self.ratings),
        )
        return self

    def get_rating(self, team: str) -> float:
        return self.ratings[team]

    def get_all_ratings(self) -> dict[str, float]:
        return dict(sorted(self.ratings.items(), key=lambda x: x[1], reverse=True))
