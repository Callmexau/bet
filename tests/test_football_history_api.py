import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import db
import app
from data_client import FootballClient


class FootballHistoryApiTests(unittest.TestCase):
    def setUp(self):
        import data_client

        data_client._FOOTBALL_HISTORY_API_CACHE.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = db.DB_PATH
        db.DB_PATH = Path(self.temp_dir.name) / "history.db"
        db.init_db()

    def tearDown(self):
        db.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def test_fetches_only_finished_results_from_api_without_local_persistence(self):
        response = {
            "matches": [
                {
                    "id": 101,
                    "status": "FINISHED",
                    "homeTeam": {"name": "Home", "id": 1},
                    "awayTeam": {"name": "Away", "id": 2},
                    "score": {"fullTime": {"home": 2, "away": 1}},
                    "utcDate": "2025-08-15T19:00:00Z",
                    "competition": {"name": "Premier League", "code": "PL"},
                },
                {
                    "id": 102,
                    "status": "SCHEDULED",
                    "homeTeam": {"name": "Future Home"},
                    "awayTeam": {"name": "Future Away"},
                    "score": {"fullTime": {"home": None, "away": None}},
                    "utcDate": "2025-08-20T19:00:00Z",
                    "competition": {"name": "Premier League", "code": "PL"},
                },
            ],
        }
        with patch.dict(os.environ, {"FOOTBALL_DATA_API_KEY": "test-key"}):
            client = FootballClient()
        with (
            patch.object(client, "_safe_request", return_value=response) as request,
            patch("data_client.cache_get", return_value=None) as cache_get,
            patch("data_client.cache_set") as cache_set,
            patch("data_client.save_matches", create=True) as save_matches,
        ):
            matches = client.get_competition_history("PL")
            refreshed_matches = client.get_competition_history("PL")

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches, refreshed_matches)
        self.assertEqual(matches[0]["home_score"], 2)
        self.assertEqual(request.call_count, 1)
        cache_get.assert_not_called()
        cache_set.assert_not_called()
        save_matches.assert_not_called()

    def test_api_history_does_not_replace_imported_match_data(self):
        imported_match = {
            "id": "101",
            "home_team": "Home",
            "away_team": "Away",
            "home_score": 2,
            "away_score": 1,
            "match_date": "2025-08-15",
            "competition": "Premier League",
            "historical_odds": {"average_opening": {"home_win": 1.8}},
        }
        db.save_matches([imported_match], "football")
        db.save_matches(
            [{**imported_match, "historical_odds": None}],
            "football",
            replace=False,
        )

        stored = db.get_history_from_db("football", limit=1)[0]
        self.assertEqual(
            json.loads(stored["raw_data"])["historical_odds"]["average_opening"]["home_win"],
            1.8,
        )

    def test_failed_history_request_is_cached_in_memory(self):
        with patch.dict(os.environ, {"FOOTBALL_DATA_API_KEY": "test-key"}):
            client = FootballClient()
        with patch.object(client, "_safe_request", return_value=None) as request:
            self.assertEqual(client.get_competition_history("PL"), [])
            self.assertEqual(client.get_competition_history("PL"), [])

        self.assertEqual(request.call_count, 1)

    def test_model_fetches_api_history_every_time_without_reading_local_history(self):
        history = [{
            "id": "match-1",
            "home_team": "Home",
            "away_team": "Away",
            "home_score": 2,
            "away_score": 1,
            "match_date": "2025-08-15",
            "competition": "Premier League",
        }]
        with (
            patch.object(app, "get_football_client") as get_client,
            patch.object(app, "_train_football_model", return_value="trained") as train,
            patch.object(db, "get_history_from_db", side_effect=AssertionError),
        ):
            get_client.return_value.get_competition_history.return_value = history
            first_model = app.get_football_model()
            second_model = app.get_football_model()

        self.assertEqual(first_model, "trained")
        self.assertEqual(second_model, "trained")
        self.assertEqual(
            get_client.return_value.get_competition_history.call_count,
            2 * len(app.FOOTBALL_HISTORY_COMPETITIONS),
        )
        self.assertEqual(train.call_count, 2)
        self.assertEqual(json.loads(train.call_args.args[0]), history)


if __name__ == "__main__":
    unittest.main()
