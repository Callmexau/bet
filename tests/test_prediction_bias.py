import json
import unittest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import app
from data_client import BasketballClient, OddsClient
from models.basketball import BasketballModel
from models.football import FootballModel


class PredictionBiasTests(unittest.TestCase):
    def test_football_prematch_blends_model_and_no_vig_market_probabilities(self):
        model_probs = {
            "home_win": 0.70,
            "draw": 0.20,
            "away_win": 0.10,
            "markets": {},
        }
        odds_event = {
            "avg_odds": {
                "home_win": 4.0,
                "draw": 2.0,
                "away_win": 2.0,
            },
        }

        combined = app.blend_prematch_probabilities(
            model_probs, odds_event, "football"
        )

        self.assertAlmostEqual(
            sum(combined[outcome] for outcome in ("home_win", "draw", "away_win")),
            1.0,
        )
        self.assertAlmostEqual(combined["home_win"], 0.325)
        self.assertAlmostEqual(combined["draw"], 0.35)
        self.assertAlmostEqual(combined["markets"]["dc_1x"], 0.675)
        self.assertIn("marché sans marge 75%", combined["probability_source"])

    def test_basketball_prematch_blends_two_way_market_probabilities(self):
        combined = app.blend_prematch_probabilities(
            {"home_win": 0.5, "draw": 0.0, "away_win": 0.5},
            {"avg_odds": {"home_win": 2.0, "away_win": 4.0}},
            "basketball",
        )

        self.assertAlmostEqual(combined["home_win"], 0.625)
        self.assertAlmostEqual(combined["away_win"], 0.375)
        self.assertEqual(combined["draw"], 0.0)

    def test_missing_market_prices_keep_model_probabilities_and_explain_source(self):
        model_probs = {"home_win": 0.6, "draw": 0.25, "away_win": 0.15}

        combined = app.blend_prematch_probabilities(
            model_probs, None, "football"
        )

        self.assertEqual(
            {key: combined[key] for key in model_probs},
            model_probs,
        )
        self.assertIn("Modèle seul", combined["probability_source"])

    def test_odds_matcher_rejects_same_teams_on_different_dates(self):
        match = {
            "home_team": "Home United",
            "away_team": "Away City",
            "match_date": "2026-11-02",
        }
        odds_events = [{
            "home_team": "Home United",
            "away_team": "Away City",
            "commence_time": "2026-11-12T19:00:00Z",
        }]

        self.assertIsNone(OddsClient().match_odds_to_match(match, odds_events))

    def test_untrained_models_do_not_automatically_favor_the_home_team(self):
        basketball = BasketballModel()
        home_prediction = basketball.predict_prematch("Home", "Away")
        reverse_prediction = basketball.predict_prematch("Away", "Home")

        self.assertEqual(home_prediction["home_win"], 0.5)
        self.assertEqual(home_prediction["away_win"], 0.5)
        self.assertEqual(reverse_prediction["home_win"], 0.5)
        self.assertEqual(reverse_prediction["away_win"], 0.5)

        football = FootballModel()
        home_prediction = football.predict_prematch("Home", "Away")
        reverse_prediction = football.predict_prematch("Away", "Home")
        self.assertEqual(home_prediction["home_win"], home_prediction["away_win"])
        self.assertEqual(reverse_prediction["home_win"], reverse_prediction["away_win"])
        recommendation = app.get_recommendation(
            home_prediction,
            "football",
            {"home_team": "Home", "away_team": "Away"},
        )
        self.assertEqual(recommendation["outcome"], "skip")
        self.assertEqual(recommendation["label"], "Match équilibré")

    def test_equal_probabilities_do_not_tie_break_to_home(self):
        recommendation = app.get_recommendation(
            {"home_win": 0.5, "draw": 0.0, "away_win": 0.5},
            "basketball",
            {"home_team": "Home", "away_team": "Away"},
        )

        self.assertEqual(recommendation["outcome"], "skip")
        self.assertEqual(recommendation["label"], "Match équilibré")

    def test_recommendation_can_select_away_team(self):
        recommendation = app.get_recommendation(
            {"home_win": 0.3, "draw": 0.0, "away_win": 0.7},
            "basketball",
            {"home_team": "Home", "away_team": "Away"},
        )

        self.assertEqual(recommendation["outcome"], "away_win")
        self.assertEqual(recommendation["label"], "Away")

    def test_trained_team_strength_can_make_the_away_team_favorite(self):
        today = date.today()
        history = [
            {
                "home_team": "Strong",
                "away_team": "Weak",
                "home_score": 120,
                "away_score": 90,
                "match_date": (today - timedelta(days=20 - index)).isoformat(),
            }
            for index in range(20)
        ]
        model = BasketballModel().train(history, calibrate=False)
        match = {"home_team": "Weak", "away_team": "Strong"}

        prediction = model.predict_prematch(
            match["home_team"], match["away_team"], calibrate=False
        )
        recommendation = app.get_recommendation(prediction, "basketball", match)

        self.assertGreater(prediction["away_win"], prediction["home_win"])
        self.assertEqual(recommendation["outcome"], "away_win")

    def test_fetch_requests_api_each_time_and_returns_only_completed_past_games(self):
        today = date.today().isoformat()
        yesterday = date.fromordinal(date.today().toordinal() - 1).isoformat()
        client = MagicMock()
        client.get_history.return_value = [
            {
                "id": 101,
                "status": "Final",
                "match_date": yesterday,
                "home_team": "Home",
                "away_team": "Opponent",
                "home_score": 90,
                "away_score": 100,
            },
            {
                "id": 102,
                "status": "Final",
                "match_date": today,
                "home_team": "Home",
                "away_team": "Opponent",
                "home_score": 90,
                "away_score": 100,
            },
            {
                "id": 103,
                "status": "Scheduled",
                "match_date": yesterday,
                "home_team": "Home",
                "away_team": "Opponent",
                "home_score": 0,
                "away_score": 0,
            },
        ]
        matches = [{
            "status": "Scheduled",
            "home_team": "Home",
            "home_team_id": 1,
            "away_team": "Away",
            "away_team_id": 2,
        }]

        first_fetch = app._fetch_basketball_history(client, matches)
        second_fetch = app._fetch_basketball_history(client, matches)

        self.assertEqual(client.get_history.call_count, 4)
        self.assertEqual(first_fetch, [client.get_history.return_value[0]])
        self.assertEqual(second_fetch, first_fetch)

    def test_basketball_client_does_not_read_or_write_history_in_sqlite(self):
        import data_client

        data_client._BASKETBALL_HISTORY_API_CACHE.clear()
        client = BasketballClient()
        response = {
            "data": [{
                "id": 101,
                "status": "Final",
                "date": "2026-10-30",
                "home_team": {"id": 1, "full_name": "Home"},
                "visitor_team": {"id": 2, "full_name": "Away"},
                "home_team_score": 90,
                "visitor_team_score": 100,
                "period": 4,
                "time": "0:00",
            }]
        }

        with (
            patch.object(client, "_safe_request", return_value=response) as request,
            patch("data_client.cache_get") as cache_get,
            patch("data_client.cache_set") as cache_set,
        ):
            first_fetch = client.get_history(1)
            second_fetch = client.get_history(1)

        self.assertEqual(first_fetch, second_fetch)
        self.assertEqual(request.call_count, 1)
        cache_get.assert_not_called()
        cache_set.assert_not_called()

    def test_basketball_today_games_are_not_saved_as_local_history(self):
        client = BasketballClient()
        response = {
            "data": [{
                "id": 101,
                "status": "Final",
                "date": date.today().isoformat(),
                "home_team": {"id": 1, "full_name": "Home"},
                "visitor_team": {"id": 2, "full_name": "Away"},
                "home_team_score": 90,
                "visitor_team_score": 100,
                "period": 4,
                "time": "0:00",
            }]
        }

        with (
            patch.object(client, "_safe_request", return_value=response),
            patch("data_client.cache_get", return_value=None),
            patch("data_client.cache_set"),
            patch("data_client.save_match", create=True) as save_match,
        ):
            matches = client.get_today_matches()

        self.assertEqual(matches[0]["status"], "Final")
        save_match.assert_not_called()


if __name__ == "__main__":
    unittest.main()
