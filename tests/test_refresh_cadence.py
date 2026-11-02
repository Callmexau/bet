import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from data_client import (
    BasketballClient,
    FootballClient,
    LIVE_REFRESH_INTERVAL_SECONDS,
    UPCOMING_REFRESH_INTERVAL_SECONDS,
    match_list_refresh_interval_seconds,
)


class RefreshCadenceTests(unittest.TestCase):
    def test_live_football_matches_refresh_every_30_seconds(self):
        matches = [
            {"sport": "football", "status": "IN_PLAY"},
            {"sport": "football", "status": "SCHEDULED"},
        ]

        self.assertEqual(
            match_list_refresh_interval_seconds(matches),
            LIVE_REFRESH_INTERVAL_SECONDS,
        )

    def test_live_basketball_matches_refresh_every_30_seconds(self):
        match = {"sport": "basketball", "status": "8:42", "period": 2}

        self.assertEqual(
            match_list_refresh_interval_seconds([match]),
            LIVE_REFRESH_INTERVAL_SECONDS,
        )

    def test_upcoming_matches_refresh_after_35_minutes_by_default(self):
        matches = [{"sport": "football", "status": "SCHEDULED"}]

        self.assertEqual(
            match_list_refresh_interval_seconds(matches),
            UPCOMING_REFRESH_INTERVAL_SECONDS,
        )

    def test_upcoming_matches_are_rechecked_near_kickoff(self):
        kickoff = datetime.now(timezone.utc) + timedelta(minutes=10)
        matches = [{
            "sport": "football",
            "status": "TIMED",
            "kickoff_utc": kickoff.isoformat(),
        }]

        interval = match_list_refresh_interval_seconds(matches)

        self.assertGreaterEqual(interval, 590)
        self.assertLessEqual(interval, 600)

    def test_finished_basketball_games_do_not_trigger_live_refresh(self):
        match = {"sport": "basketball", "status": "Final", "period": 4}

        self.assertEqual(
            match_list_refresh_interval_seconds([match]),
            UPCOMING_REFRESH_INTERVAL_SECONDS,
        )

    def test_football_match_list_uses_live_cache_ttl(self):
        client = FootballClient()
        response = {
            "matches": [{
                "id": 1,
                "status": "IN_PLAY",
                "competition": {"code": "PL"},
                "homeTeam": {"name": "Home"},
                "awayTeam": {"name": "Away"},
                "score": {"fullTime": {}},
                "utcDate": datetime.now(timezone.utc).isoformat(),
            }]
        }
        with (
            patch("data_client.cache_get", return_value=None),
            patch("data_client.cache_set") as cache_set,
            patch.object(client, "_safe_request", return_value=response),
            patch.object(client, "_get_can_matches", return_value=[]),
        ):
            client.get_today_matches()

        self.assertEqual(
            cache_set.call_args.kwargs["ttl"],
            LIVE_REFRESH_INTERVAL_SECONDS,
        )

    def test_basketball_match_list_uses_live_cache_ttl(self):
        client = BasketballClient()
        response = {
            "data": [{
                "id": 1,
                "status": "8:42",
                "period": 2,
                "date": datetime.now(timezone.utc).isoformat(),
                "home_team": {"id": 1, "full_name": "Home"},
                "visitor_team": {"id": 2, "full_name": "Away"},
                "home_team_score": 50,
                "visitor_team_score": 48,
            }]
        }
        with (
            patch("data_client.cache_get", return_value=None),
            patch("data_client.cache_set") as cache_set,
            patch.object(client, "_safe_request", return_value=response),
        ):
            client.get_today_matches()

        self.assertEqual(
            cache_set.call_args.kwargs["ttl"],
            LIVE_REFRESH_INTERVAL_SECONDS,
        )


if __name__ == "__main__":
    unittest.main()
