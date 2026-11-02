import json
import unittest

from backtest import simulate_opening_odds_bet


class OpeningOddsBacktestTests(unittest.TestCase):
    def test_selects_positive_ev_and_calculates_unit_profit(self):
        prediction = {"home_win": 0.6, "draw": 0.2, "away_win": 0.2}
        raw_data = json.dumps({
            "historical_odds": {
                "average_opening": {
                    "home_win": 2.0,
                    "draw": 4.0,
                    "away_win": 5.9,
                }
            }
        })

        result = simulate_opening_odds_bet(prediction, "H", raw_data)

        self.assertIsNotNone(result)
        self.assertEqual(result["outcome"], "H")
        self.assertEqual(result["profit_units"], 1.0)

    def test_ignores_missing_or_below_threshold_odds(self):
        prediction = {"home_win": 0.4, "draw": 0.3, "away_win": 0.3}
        no_odds = json.dumps({"historical_odds": {}})
        low_ev = json.dumps({
            "historical_odds": {
                "average_opening": {
                    "home_win": 2.0,
                    "draw": 2.0,
                    "away_win": 2.0,
                }
            }
        })

        self.assertIsNone(simulate_opening_odds_bet(prediction, "H", no_odds))
        self.assertIsNone(simulate_opening_odds_bet(prediction, "H", low_ev))


if __name__ == "__main__":
    unittest.main()
