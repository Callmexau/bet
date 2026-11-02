import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import db
import app
from data_client import FootballClient


class FootballHistoryApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = db.DB_PATH
        db.DB_PATH = Path(self.temp_dir.name) / "history.db"
        db.init_db()

    def tearDown(self):
        db.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def test_imports_only_finished_results_and_caches_them(self):
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
        with patch.object(client, "_safe_request", return_value=response) as request:
            matches = client.get_competition_history("PL")
            cached_matches = client.get_competition_history("PL")

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches, cached_matches)
        self.assertEqual(matches[0]["home_score"], 2)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(len(db.get_history_from_db("football", limit=10)), 1)

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

    def test_failed_history_request_is_cached_to_avoid_repeating_it(self):
        with patch.dict(os.environ, {"FOOTBALL_DATA_API_KEY": "test-key"}):
            client = FootballClient()
        with patch.object(client, "_safe_request", return_value=None) as request:
            self.assertEqual(client.get_competition_history("PL"), [])
            self.assertEqual(client.get_competition_history("PL"), [])

        request.assert_called_once()

    def test_model_refreshes_history_before_training_when_database_is_small(self):
        initial_history = json.dumps([])
        refreshed_history = json.dumps([{"home_team": "Home", "away_team": "Away"}])
        with (
            patch.object(
                app,
                "_training_history_json",
                side_effect=[initial_history, refreshed_history],
            ),
            patch.object(app, "get_football_client") as get_client,
            patch.object(app, "_train_football_model", return_value="trained") as train,
        ):
            model = app.get_football_model()

        get_client.return_value.sync_competition_history.assert_called_once_with()
        train.assert_called_once_with(refreshed_history)
        self.assertEqual(model, "trained")

    def test_model_uses_compatibility_sync_when_client_method_is_missing(self):
        api_match = {
            "id": 101,
            "home_team": "Home",
            "away_team": "Away",
            "home_score": 2,
            "away_score": 1,
            "match_date": "2025-08-15",
            "competition": "Premier League",
        }
        client = SimpleNamespace(
            api_key="test-key",
            headers={},
            _safe_request=lambda *args, **kwargs: {
                "matches": [{
                    "status": "FINISHED",
                    "id": 101,
                }],
            },
            _normalize_match=lambda _: api_match,
        )
        with (
            patch.object(app, "get_football_client", return_value=client),
            patch.object(app, "_train_football_model", return_value="trained") as train,
        ):
            model = app.get_football_model()

        train.assert_called_once()
        training_history = json.loads(train.call_args.args[0])
        self.assertEqual(len(training_history), 1)
        self.assertEqual(training_history[0]["home_team"], "Home")
        self.assertEqual(model, "trained")


if __name__ == "__main__":
    unittest.main()
