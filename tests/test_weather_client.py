import unittest
from datetime import datetime, timezone

from weather_client import WeatherClient


class WeatherClientTests(unittest.TestCase):
    def test_selects_forecast_hour_nearest_to_match_kickoff(self):
        payload = {
            "hourly": {
                "time": ["2026-11-02T14:00", "2026-11-02T15:00"],
                "temperature_2m": [19, 20],
                "precipitation_probability": [10, 70],
                "precipitation": [0.0, 1.2],
                "wind_speed_10m": [8, 24],
                "weather_code": [1, 63],
            }
        }

        forecast = WeatherClient._hourly_forecast(
            payload,
            datetime(2026, 11, 2, 14, 40, tzinfo=timezone.utc),
        )

        self.assertIsNotNone(forecast)
        self.assertEqual(forecast["time"], "2026-11-02T15:00")
        self.assertEqual(forecast["condition"], "pluie modérée")
        self.assertEqual(forecast["precipitation_probability"], 70)

    def test_rejects_forecast_too_far_from_kickoff_and_missing_venue(self):
        payload = {"hourly": {"time": ["2026-11-02T10:00"]}}
        self.assertIsNone(
            WeatherClient._hourly_forecast(
                payload,
                datetime(2026, 11, 2, 14, 0, tzinfo=timezone.utc),
            )
        )
        self.assertIsNone(WeatherClient().get_match_weather({
            "kickoff_utc": "2026-11-02T14:00:00Z",
        }))


if __name__ == "__main__":
    unittest.main()
