"""Weather context for football fixtures using Open-Meteo's non-commercial API."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import requests

from db import cache_get, cache_set

logger = logging.getLogger(__name__)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_CODES = {
    0: "ciel dégagé",
    1: "principalement dégagé",
    2: "partiellement nuageux",
    3: "couvert",
    45: "brouillard",
    48: "brouillard givrant",
    51: "bruine légère",
    53: "bruine modérée",
    55: "bruine forte",
    61: "pluie faible",
    63: "pluie modérée",
    65: "pluie forte",
    71: "neige faible",
    73: "neige modérée",
    75: "neige forte",
    80: "averses faibles",
    81: "averses modérées",
    82: "averses fortes",
    95: "orage",
    96: "orage avec grêle",
    99: "orage fort avec grêle",
}


class WeatherClient:
    """Fetches a best-effort forecast for a fixture's geocodable venue."""

    @staticmethod
    def _request_json(url: str, params: dict) -> dict | None:
        try:
            response = requests.get(url, params=params, timeout=8)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            logger.warning("Open-Meteo request failed for %s: %s", url, exc)
            return None
        except ValueError as exc:
            logger.warning("Open-Meteo returned invalid JSON for %s: %s", url, exc)
            return None
        if not isinstance(payload, dict):
            logger.warning("Open-Meteo returned an unexpected response for %s", url)
            return None
        return payload

    @staticmethod
    def _normalize_place(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    def _coordinates_for_venue(self, venue: str) -> tuple[float, float] | None:
        cache_key = f"weather:geocode:{self._normalize_place(venue)}"
        cached = cache_get(cache_key)
        if cached is not None:
            try:
                return float(cached["latitude"]), float(cached["longitude"])
            except (KeyError, TypeError, ValueError):
                logger.warning("Ignoring invalid cached venue coordinates for %s", venue)

        payload = self._request_json(
            GEOCODING_URL,
            {"name": venue, "count": 5, "language": "en", "format": "json"},
        )
        results = payload.get("results", []) if payload else []
        venue_key = self._normalize_place(venue)
        exact_results = [
            item for item in results
            if isinstance(item, dict)
            and self._normalize_place(str(item.get("name", ""))) == venue_key
        ]
        if not exact_results:
            logger.info("No exact geocoding result for football venue %s", venue)
            return None

        place = exact_results[0]
        try:
            coordinates = {
                "latitude": float(place["latitude"]),
                "longitude": float(place["longitude"]),
            }
        except (KeyError, TypeError, ValueError):
            logger.warning("Invalid geocoding coordinates for football venue %s", venue)
            return None
        cache_set(cache_key, coordinates, ttl=30 * 24 * 60 * 60)
        return coordinates["latitude"], coordinates["longitude"]

    @staticmethod
    def _parse_kickoff(value: str) -> datetime | None:
        try:
            kickoff = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)
        return kickoff.astimezone(timezone.utc)

    @staticmethod
    def _hourly_forecast(payload: dict, kickoff: datetime) -> dict | None:
        hourly = payload.get("hourly")
        if not isinstance(hourly, dict):
            return None
        times = hourly.get("time", [])
        if not times:
            return None
        target = kickoff.strftime("%Y-%m-%dT%H:00")
        try:
            index = min(range(len(times)), key=lambda i: abs(
                datetime.fromisoformat(times[i]).replace(tzinfo=timezone.utc) - kickoff
            ))
            distance = abs(
                datetime.fromisoformat(times[index]).replace(tzinfo=timezone.utc) - kickoff
            )
        except (TypeError, ValueError, IndexError):
            return None
        if distance.total_seconds() > 90 * 60:
            return None

        def value(name: str):
            values = hourly.get(name, [])
            return values[index] if index < len(values) else None

        code = value("weather_code")
        condition = WEATHER_CODES.get(code, "conditions inconnues") if code is not None else "conditions inconnues"
        return {
            "time": times[index] if times[index] else target,
            "temperature": value("temperature_2m"),
            "precipitation_probability": value("precipitation_probability"),
            "precipitation_mm": value("precipitation"),
            "wind_kmh": value("wind_speed_10m"),
            "condition": condition,
        }

    def get_match_weather(self, match: dict) -> dict | None:
        venue = match.get("venue")
        kickoff = self._parse_kickoff(match.get("kickoff_utc", ""))
        if not isinstance(venue, str) or not venue.strip() or kickoff is None:
            return None

        coordinates = self._coordinates_for_venue(venue.strip())
        if coordinates is None:
            return None
        latitude, longitude = coordinates
        cache_key = (
            f"weather:forecast:{latitude:.3f}:{longitude:.3f}:"
            f"{kickoff.strftime('%Y-%m-%dT%H')}"
        )
        cached = cache_get(cache_key)
        if cached is not None:
            return cached

        payload = self._request_json(
            FORECAST_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": (
                    "temperature_2m,precipitation_probability,precipitation,"
                    "wind_speed_10m,weather_code"
                ),
                "forecast_days": 7,
                "timezone": "UTC",
            },
        )
        if payload is None:
            return None
        forecast = self._hourly_forecast(payload, kickoff)
        if forecast is not None:
            cache_set(cache_key, forecast, ttl=60 * 60)
        return forecast
