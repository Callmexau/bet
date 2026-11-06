# -*- coding: utf-8 -*-
import os
import unittest
from unittest.mock import patch

from data_client import (
    FOOTBALL_COMPETITIONS,
    FOOTBALL_HISTORY_COMPETITIONS,
    ODDS_API_EXTRA_COMPETITIONS,
    FootballClient,
    OddsClient,
)
from models.features import competition_code


class ExtraCompetitionsTests(unittest.TestCase):
    def test_extra_competitions_configured(self):
        # Toutes les compétitions Option 1 sont dans FOOTBALL_COMPETITIONS
        for code in ODDS_API_EXTRA_COMPETITIONS:
            self.assertIn(code, FOOTBALL_COMPETITIONS)

        # Les compétitions Option 1 ne sont PAS dans FOOTBALL_HISTORY_COMPETITIONS
        # (protège le quota gratuit football-data.org)
        for code in ODDS_API_EXTRA_COMPETITIONS:
            self.assertNotIn(code, FOOTBALL_HISTORY_COMPETITIONS)

    def test_comp_name_to_code_mapping(self):
        self.assertEqual(competition_code({"competition": "UEFA Europa League"}), "UEL")
        self.assertEqual(competition_code({"competition": "Ligue Europa"}), "UEL")
        self.assertEqual(competition_code({"competition": "UEFA Europa Conference League"}), "UECL")
        self.assertEqual(competition_code({"competition": "Ligue 2"}), "FL2")
        self.assertEqual(competition_code({"competition": "La Liga 2"}), "SD")
        self.assertEqual(competition_code({"competition": "Serie B"}), "SB")
        self.assertEqual(competition_code({"competition": "2. Bundesliga"}), "BL2")
        self.assertEqual(competition_code({"competition": "Saudi Pro League"}), "SPL")
        self.assertEqual(competition_code({"competition": "Major League Soccer"}), "MLS")
        self.assertEqual(competition_code({"competition": "MLS"}), "MLS")
        self.assertEqual(competition_code({"competition": "Süper Lig"}), "TSL")
        self.assertEqual(competition_code({"competition": "Jupiler Pro League"}), "BPL")
        self.assertEqual(competition_code({"competition": "Premiership"}), "SCO")
        self.assertEqual(competition_code({"competition": "CAN"}), "CAN")

    def test_odds_client_active_sport_keys(self):
        with patch.dict(os.environ, {"ODDS_API_KEY": "test-odds-key"}):
            client = OddsClient()

        mock_sports = [
            {"key": "soccer_uefa_europa_league", "active": True},
            {"key": "soccer_france_ligue_two", "active": True},
            {"key": "soccer_inactive_league", "active": False},
        ]

        with patch.object(client, "_safe_request", return_value=mock_sports), \
             patch("data_client.cache_get", return_value=None), \
             patch("data_client.cache_set"):
            active_keys = client.get_active_sport_keys()
            self.assertIn("soccer_uefa_europa_league", active_keys)
            self.assertIn("soccer_france_ligue_two", active_keys)
            self.assertNotIn("soccer_inactive_league", active_keys)

    def test_football_client_get_odds_api_matches(self):
        with patch.dict(os.environ, {"FOOTBALL_DATA_API_KEY": "test-key", "ODDS_API_KEY": "test-odds-key"}):
            f_client = FootballClient()

        mock_events = [
            {
                "id": "match_uel_1",
                "home_team": "AS Roma",
                "away_team": "Athletic Bilbao",
                "commence_time": "2026-11-06T20:00:00Z",
            }
        ]

        with patch("data_client.OddsClient.get_active_sport_keys", return_value={"soccer_uefa_europa_league"}), \
             patch("data_client.OddsClient.get_odds_for_sport", return_value=mock_events):
            matches = f_client._get_odds_api_matches("2026-11-06", "2026-11-06")
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0]["competition_code"], "UEL")
            self.assertEqual(matches[0]["home_team"], "AS Roma")
            self.assertEqual(matches[0]["away_team"], "Athletic Bilbao")
            self.assertEqual(matches[0]["id"], "odds_match_uel_1")


if __name__ == "__main__":
    unittest.main()

