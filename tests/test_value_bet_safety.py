import unittest

from app import (
    cap_portfolio_stakes,
    conservative_match_ev,
    is_value_bet_eligible,
    kelly_fraction,
)
from models.football import FootballModel


class ValueBetSafetyTests(unittest.TestCase):
    def test_untrained_model_cannot_generate_an_automatic_value_signal(self):
        model = FootballModel()
        odds_event = {
            "avg_odds": {"home_win": 2.0, "draw": 3.5, "away_win": 4.0},
            "best_odds": {"home_win": 2.1, "draw": 3.6, "away_win": 4.2},
        }
        probs = {"home_win": 0.6, "draw": 0.2, "away_win": 0.2}

        self.assertIsNone(conservative_match_ev(model, probs, odds_event))

    def test_conservative_probabilities_shrink_model_edge_toward_market(self):
        model_probs = {
            "home_win": 0.70,
            "draw": 0.20,
            "away_win": 0.10,
        }
        average_odds = {
            "home_win": 2.0,
            "draw": 3.5,
            "away_win": 4.0,
        }

        market_probs = FootballModel.market_implied_probabilities(average_odds)
        conservative = FootballModel.conservative_probabilities(model_probs, average_odds)

        self.assertIsNotNone(market_probs)
        self.assertIsNotNone(conservative)
        self.assertAlmostEqual(sum(conservative.values()), 1.0)
        self.assertLess(
            abs(conservative["home_win"] - market_probs["home_win"]),
            abs(model_probs["home_win"] - market_probs["home_win"]),
        )

    def test_missing_market_prices_do_not_produce_conservative_probabilities(self):
        self.assertIsNone(FootballModel.market_implied_probabilities({"home_win": 2.0}))
        self.assertIsNone(
            FootballModel.conservative_probabilities(
                {"home_win": 0.5, "draw": 0.3, "away_win": 0.2},
                {"home_win": 2.0, "draw": float("nan"), "away_win": 4.0},
            )
        )

    def test_kelly_stake_is_capped_and_total_exposure_is_limited(self):
        self.assertEqual(kelly_fraction(0.99, 10.0), 0.5)

        bets = [
            {"stake": 1000, "kelly_pct": 1.0},
            {"stake": 1000, "kelly_pct": 1.0},
            {"stake": 1000, "kelly_pct": 1.0},
        ]
        cap_portfolio_stakes(bets, bankroll=100_000, currency="FCFA")

        self.assertLessEqual(sum(bet["stake"] for bet in bets), 1000)
        self.assertTrue(all(bet["stake"] <= 500 for bet in bets))

    def test_strict_filter_requires_high_probability_and_low_odds(self):
        self.assertTrue(is_value_bet_eligible(0.65, 1.80, 0.08, 0.08))
        self.assertFalse(is_value_bet_eligible(0.649, 1.80, 0.10, 0.08))
        self.assertFalse(is_value_bet_eligible(0.70, 1.81, 0.10, 0.08))
        self.assertFalse(is_value_bet_eligible(0.70, 1.80, 0.079, 0.08))


if __name__ == "__main__":
    unittest.main()
