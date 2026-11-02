import unittest
from unittest.mock import MagicMock, patch

import app
from models.football import FootballModel, MAX_GOALS


class FootballMarketsTests(unittest.TestCase):
    def test_prematch_prediction_includes_goal_and_btts_markets(self):
        prediction = FootballModel().predict_prematch("Home", "Away")
        markets = prediction["markets"]

        self.assertEqual(prediction["history_matches"], 0)
        for key in (
            "over_1_5", "under_1_5",
            "over_2_5", "under_2_5",
            "over_3_5", "under_3_5",
            "btts_yes", "btts_no",
        ):
            self.assertIn(key, markets)
            self.assertGreaterEqual(markets[key], 0)
            self.assertLessEqual(markets[key], 1)

        for over, under in (
            ("over_1_5", "under_1_5"),
            ("over_2_5", "under_2_5"),
            ("over_3_5", "under_3_5"),
            ("btts_yes", "btts_no"),
        ):
            self.assertAlmostEqual(markets[over] + markets[under], 1.0, places=3)

    def test_score_grid_covers_the_same_goal_range_as_1x2_model(self):
        grid = FootballModel()._score_grid(1.6, 1.2)
        self.assertEqual(len(grid), (MAX_GOALS + 1) ** 2)
        self.assertAlmostEqual(sum(grid.values()), 1.0)

    def test_market_forecast_renderer_shows_all_supported_markets(self):
        streamlit = MagicMock()

        class Column:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        def make_columns(count):
            return [Column() for _ in range(count)]

        streamlit.columns.side_effect = make_columns
        prediction = FootballModel().predict_prematch("Home", "Away")

        with patch.object(app, "st", streamlit):
            app.render_football_market_forecasts(prediction)

        rendered_labels = [
            call.args[0] for call in streamlit.metric.call_args_list
        ]
        for line in ("1.5", "2.5", "3.5"):
            self.assertTrue(
                any(line in label for label in rendered_labels),
            )
        self.assertTrue(
            any(label in ("Oui, les deux marquent", "Non, les deux marquent")
                for label in rendered_labels)
        )
        self.assertIn("Double chance 1X", rendered_labels)
        self.assertIn("Double chance X2", rendered_labels)
        self.assertIn("Double chance 12", rendered_labels)
        streamlit.caption.assert_any_call(
            "Modèle sans historique : estimations indicatives, non validées et "
            "à ne pas interpréter comme des conseils de pari."
        )


if __name__ == "__main__":
    unittest.main()
