import csv
import json
import tempfile
import unittest
from pathlib import Path

import db
from import_history import read_matches


class ImportHistoryTests(unittest.TestCase):
    def test_imports_finished_matches_and_skips_unplayed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "E0.csv"
            columns = [
                "Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
                "AvgH", "AvgD", "AvgA", "AvgCH", "AvgCD", "AvgCA",
            ]
            with csv_path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                writer.writerow({
                    "Div": "E0",
                    "Date": "15/08/2025",
                    "HomeTeam": "Liverpool",
                    "AwayTeam": "Bournemouth",
                    "FTHG": "4",
                    "FTAG": "2",
                    "AvgH": "1.31",
                    "AvgD": "5.96",
                    "AvgA": "8.31",
                    "AvgCH": "1.29",
                    "AvgCD": "6.02",
                    "AvgCA": "8.68",
                })
                writer.writerow({
                    "Div": "E0",
                    "Date": "20/08/2025",
                    "HomeTeam": "Arsenal",
                    "AwayTeam": "Chelsea",
                })

            matches, skipped = read_matches(csv_path)

        self.assertEqual(skipped, 1)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["match_date"], "2025-08-15")
        self.assertEqual(matches[0]["competition_code"], "PL")
        self.assertEqual(
            matches[0]["historical_odds"]["average_opening"]["home_win"],
            1.31,
        )
        self.assertEqual(
            matches[0]["historical_odds"]["average_closing"]["away_win"],
            8.68,
        )

    def test_import_is_repeatable_and_stores_rows_in_application_database(self):
        with tempfile.TemporaryDirectory() as directory:
            original_db_path = db.DB_PATH
            db.DB_PATH = Path(directory) / "history.db"
            try:
                db.init_db()
                match = {
                    "id": "fdc_test_match",
                    "home_team": "Home",
                    "away_team": "Away",
                    "home_score": 1,
                    "away_score": 0,
                    "match_date": "2025-08-15",
                    "competition": "Premier League",
                    "historical_odds": {
                        "average_opening": {
                            "home_win": 1.8,
                            "draw": 3.5,
                            "away_win": 4.2,
                        }
                    },
                }
                db.save_matches([match], "football")
                db.save_matches([match], "football")

                history = db.get_history_from_db("football", limit=10)
                self.assertEqual(len(history), 1)
                raw_data = json.loads(history[0]["raw_data"])
                self.assertEqual(
                    raw_data["historical_odds"]["average_opening"]["home_win"],
                    1.8,
                )
            finally:
                db.DB_PATH = original_db_path


if __name__ == "__main__":
    unittest.main()
