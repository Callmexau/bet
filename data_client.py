# -*- coding: utf-8 -*-
"""
data_client.py - Clients API pour football-data.org, balldontlie et The Odds API.
Interface commune : get_today_matches(), get_live_match(id), get_history(team, n=20).
Les listes sont mises en cache avec un TTL adaptatif ; l'historique API est en mémoire.
"""

import os
import logging
import requests
import time
from datetime import date, datetime, timedelta, timezone
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from db import cache_get, cache_set, CACHE_TTL

load_dotenv()
logger = logging.getLogger(__name__)

# ─────────────────────────── Constantes ────────────────────────────────────

FOOTBALL_BASE_URL   = "https://api.football-data.org/v4"
BALLDONTLIE_BASE_URL = "https://api.balldontlie.io/v1"
ODDS_BASE_URL        = "https://api.the-odds-api.com/v4"

# Compétitions suivies (football-data.org Tier One + The Odds API)
FOOTBALL_COMPETITIONS = [
    "PL",   # Premier League (Angleterre)
    "PD",   # La Liga (Espagne)
    "FL1",  # Ligue 1 (France)
    "BL1",  # Bundesliga (Allemagne)
    "SA",   # Serie A (Italie)
    "DED",  # Eredivisie (Pays-Bas)
    "PPL",  # Primeira Liga (Portugal)
    "ELC",  # Championship (Angleterre D2)
    "BSA",  # Campeonato Brasileiro Série A (Brésil)
    "CL",   # Ligue des Champions (UEFA)
    "CLI",  # Copa Libertadores (CONMEBOL)
    "WC",   # Coupe du Monde (FIFA World Cup)
    "EC",   # Championnat d'Europe (Euro UEFA)
    "CAN",  # Coupe d'Afrique des Nations (CAF)
]
FOOTBALL_HISTORY_COMPETITIONS = tuple(
    competition for competition in FOOTBALL_COMPETITIONS if competition != "CAN"
)
LIVE_REFRESH_INTERVAL_SECONDS = 30
UPCOMING_REFRESH_INTERVAL_SECONDS = 35 * 60

_FOOTBALL_HISTORY_API_CACHE: dict[str, tuple[float, list[dict]]] = {}
_BASKETBALL_HISTORY_API_CACHE: dict[
    tuple[str, int, int], tuple[float, list[dict]]
] = {}


def is_live_match(match: dict) -> bool:
    period = match.get("period")
    return (
        match.get("status") in ("IN_PLAY", "PAUSED")
        or (
            match.get("sport") == "basketball"
            and isinstance(period, (int, float))
            and period > 0
            and match.get("status") != "Final"
        )
    )


def match_list_refresh_interval_seconds(matches: list[dict]) -> int:
    if any(is_live_match(match) for match in matches):
        return LIVE_REFRESH_INTERVAL_SECONDS

    now = datetime.now(timezone.utc)
    upcoming_kickoffs = []
    for match in matches:
        if match.get("status") in ("FINISHED", "Final"):
            continue
        raw = match.get("raw") or {}
        if not isinstance(raw, dict):
            continue
        kickoff = (
            match.get("kickoff_utc")
            or raw.get("utcDate")
            or raw.get("commence_time")
            or raw.get("datetime")
        )
        if not isinstance(kickoff, str) or "T" not in kickoff:
            continue
        try:
            kickoff_at = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
        except ValueError:
            continue
        if kickoff_at.tzinfo is None:
            kickoff_at = kickoff_at.replace(tzinfo=timezone.utc)
        upcoming_kickoffs.append(kickoff_at.astimezone(timezone.utc))

    if not upcoming_kickoffs:
        return UPCOMING_REFRESH_INTERVAL_SECONDS

    seconds_to_next_kickoff = int((min(upcoming_kickoffs) - now).total_seconds())
    return min(
        UPCOMING_REFRESH_INTERVAL_SECONDS,
        max(LIVE_REFRESH_INTERVAL_SECONDS, seconds_to_next_kickoff),
    )


def _match_list_cache_ttl(matches: list[dict]) -> int:
    return match_list_refresh_interval_seconds(matches)

# ─────────────────────────── Classe de base ────────────────────────────────


class BaseClient(ABC):
    """Interface commune pour tous les clients API sportifs."""

    @abstractmethod
    def get_today_matches(self) -> list[dict]:
        """Retourne les matchs du jour normalisés."""

    @abstractmethod
    def get_live_match(self, match_id: int | str) -> dict | None:
        """Retourne l'état en direct d'un match (score, minute…)."""

    @abstractmethod
    def get_history(self, team: str, n: int = 20) -> list[dict]:
        """Retourne les n derniers matchs d'une équipe."""

    def _safe_request(
        self,
        url: str,
        headers: dict | None = None,
        params: dict | None = None,
        timeout: int = 10,
    ) -> dict | None:
        """
        Exécute un appel HTTP GET avec gestion des erreurs (timeout, quota…).
        Retourne le JSON parsé ou None en cas d'échec.
        """
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
            if resp.status_code == 429:
                logger.warning("Quota API dépassé (429) pour %s", url)
                return None
            if resp.status_code == 403:
                logger.error("Accès refusé (403) — vérifiez votre clé API pour %s", url)
                return None
            if resp.status_code == 404:
                logger.debug("Ressource non trouvée (404) pour %s", url)
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            logger.error("Timeout lors de l'appel à %s", url)
            return None
        except requests.exceptions.ConnectionError:
            logger.error("Erreur de connexion pour %s", url)
            return None
        except requests.exceptions.HTTPError as e:
            logger.error("Erreur HTTP %s pour %s", e.response.status_code, url)
            return None
        except Exception as e:
            logger.error("Erreur inattendue: %s", e)
            return None


# ──────────────────────── Client Football ──────────────────────────────────


class FootballClient(BaseClient):
    """
    Client pour l'API football-data.org (plan gratuit).
    Documentation : https://www.football-data.org/documentation/quickstart
    """

    def __init__(self):
        self.api_key = os.getenv("FOOTBALL_DATA_API_KEY", "")
        if not self.api_key:
            logger.warning("FOOTBALL_DATA_API_KEY non définie dans .env")
        self.headers = {"X-Auth-Token": self.api_key}

    # ── Normalisation ──────────────────────────────────────────────────────

    def _normalize_match(self, raw: dict) -> dict:
        """Normalise un match brut de l'API en format standard."""
        home = raw.get("homeTeam", {})
        away = raw.get("awayTeam", {})
        score = raw.get("score", {})
        ft = score.get("fullTime", {})
        ht = score.get("halfTime", {})

        # Extraction de la minute (uniquement pour les matchs en cours)
        minute = None
        if raw.get("status") == "IN_PLAY":
            # L'API gratuite ne fournit pas la minute directement ;
            # on l'estime depuis l'heure de début
            try:
                utc_date = datetime.fromisoformat(raw["utcDate"].replace("Z", "+00:00"))
                elapsed = (datetime.now(utc_date.tzinfo) - utc_date).seconds // 60
                minute = min(elapsed, 90)
            except Exception:
                minute = None

        return {
            "id": raw.get("id"),
            "sport": "football",
            "home_team": home.get("name", "?"),
            "away_team": away.get("name", "?"),
            "home_team_id": home.get("id"),
            "away_team_id": away.get("id"),
            "home_score": ft.get("home"),
            "away_score": ft.get("away"),
            "home_score_ht": ht.get("home"),
            "away_score_ht": ht.get("away"),
            "status": raw.get("status"),       # SCHEDULED / IN_PLAY / PAUSED / FINISHED
            "minute": minute,
            "match_date": raw.get("utcDate", "")[:10],
            "competition": raw.get("competition", {}).get("name", ""),
            "competition_code": raw.get("competition", {}).get("code", ""),
            "venue": raw.get("venue", ""),
            "kickoff_utc": raw.get("utcDate", ""),
            "red_cards_home": self._count_cards(raw, "home", "RED"),
            "red_cards_away": self._count_cards(raw, "away", "RED"),
            "raw": raw,
        }

    def _count_cards(self, raw: dict, side: str, card_type: str) -> int:
        """Compte les cartons d'un type donné pour un côté (home/away)."""
        bookings = raw.get("bookings", [])
        return sum(
            1 for b in bookings
            if b.get("card", "").upper() == card_type
            and b.get("team", {}).get("id") == raw.get(f"{side}Team", {}).get("id")
        )

    # ── Interface publique ─────────────────────────────────────────────────

    def get_today_matches(self) -> list[dict]:
        """Retourne tous les matchs du jour pour les compétitions suivies."""
        today = date.today().isoformat()
        cache_key = f"football:today:{today}"

        cached = cache_get(cache_key)
        if cached is not None:
            return cached

        all_matches = []
        target_comps = set(FOOTBALL_COMPETITIONS)

        # 1. Requête globale (récupère tous les matchs du jour en 1 seul appel API)
        url = f"{FOOTBALL_BASE_URL}/matches"
        data = self._safe_request(url, headers=self.headers)

        if data and "matches" in data:
            for m in data["matches"]:
                comp_code = m.get("competition", {}).get("code", "")
                if comp_code in target_comps:
                    normalized = self._normalize_match(m)
                    all_matches.append(normalized)
        else:
            # Fallback par compétition si la requête globale échoue
            fd_comps = [c for c in FOOTBALL_COMPETITIONS if c != "CAN"]
            for comp in fd_comps:
                url_c = f"{FOOTBALL_BASE_URL}/competitions/{comp}/matches"
                data_c = self._safe_request(url_c, headers=self.headers, params={"dateFrom": today, "dateTo": today})
                if data_c and "matches" in data_c:
                    for m in data_c["matches"]:
                        normalized = self._normalize_match(m)
                        all_matches.append(normalized)

        # 2. Matchs CAN (via The Odds API si active)
        if "CAN" in FOOTBALL_COMPETITIONS:
            can_matches = self._get_can_matches(today, today)
            all_matches.extend(can_matches)

        cache_set(cache_key, all_matches, ttl=_match_list_cache_ttl(all_matches))
        logger.info("Football: %d matchs aujourd'hui", len(all_matches))
        return all_matches

    def get_week_matches(self, days: int = 7) -> list[dict]:
        """
        Retourne les matchs programmés pour les N prochains jours.
        Utile pour l'onglet 'Semaine' et la préparation des pronostics.
        """
        today = date.today()
        end   = today + timedelta(days=days)
        cache_key = f"football:week:{today.isoformat()}:{days}"

        cached = cache_get(cache_key)
        if cached is not None:
            return cached

        all_matches = []
        target_comps = set(FOOTBALL_COMPETITIONS)

        # 1. Requête globale sur la plage de dates (1 seul appel)
        url = f"{FOOTBALL_BASE_URL}/matches"
        params = {"dateFrom": today.isoformat(), "dateTo": end.isoformat()}
        data = self._safe_request(url, headers=self.headers, params=params)

        if data and "matches" in data:
            for m in data["matches"]:
                comp_code = m.get("competition", {}).get("code", "")
                if comp_code in target_comps:
                    normalized = self._normalize_match(m)
                    all_matches.append(normalized)
        else:
            # Fallback par compétition
            fd_comps = [c for c in FOOTBALL_COMPETITIONS if c != "CAN"]
            for comp in fd_comps:
                url_c = f"{FOOTBALL_BASE_URL}/competitions/{comp}/matches"
                data_c = self._safe_request(url_c, headers=self.headers, params={"dateFrom": today.isoformat(), "dateTo": end.isoformat()})
                if data_c and "matches" in data_c:
                    for m in data_c["matches"]:
                        normalized = self._normalize_match(m)
                        all_matches.append(normalized)

        # 2. Matchs CAN (via The Odds API si active)
        if "CAN" in FOOTBALL_COMPETITIONS:
            can_matches = self._get_can_matches(today.isoformat(), end.isoformat())
            all_matches.extend(can_matches)

        cache_set(cache_key, all_matches, ttl=_match_list_cache_ttl(all_matches))
        logger.info("Football: %d matchs cette semaine", len(all_matches))
        return all_matches

    def _get_can_matches(self, date_from: str, date_to: str) -> list[dict]:
        """Tente de récupérer les matchs de la CAN via The Odds API si la compétition est active."""
        can_matches = []
        try:
            odds_client = OddsClient()
            if odds_client._is_available():
                events = odds_client.get_odds_for_sport("soccer_caf_nations_cup")
                for ev in events:
                    match_date = ev.get("commence_time", "")[:10]
                    if date_from <= match_date <= date_to:
                        can_matches.append({
                            "id": f"can_{ev.get('id')}",
                            "sport": "football",
                            "home_team": ev.get("home_team", "?"),
                            "away_team": ev.get("away_team", "?"),
                            "home_score": None,
                            "away_score": None,
                            "home_score_ht": None,
                            "away_score_ht": None,
                            "status": "SCHEDULED",
                            "minute": None,
                            "match_date": match_date,
                            "competition": "Coupe d'Afrique des Nations (CAN)",
                            "competition_code": "CAN",
                            "red_cards_home": 0,
                            "red_cards_away": 0,
                            "raw": ev,
                        })
        except Exception as e:
            logger.debug("Pas de matchs CAN disponibles: %s", e)
        return can_matches

    def get_live_match(self, match_id: int | str) -> dict | None:
        """Retourne l'état actuel d'un match en cours (données fraîches)."""
        cache_key = f"football:live:{match_id}"
        cached = cache_get(cache_key)
        if cached is not None:
            return cached

        url = f"{FOOTBALL_BASE_URL}/matches/{match_id}"
        data = self._safe_request(url, headers=self.headers)

        if data:
            normalized = self._normalize_match(data)
            cache_set(cache_key, normalized, ttl=CACHE_TTL["live_match"])
            return normalized
        return None

    def get_history(self, team_id: int | str, n: int = 20) -> list[dict]:
        """Retourne les n derniers matchs d'une équipe (pour calcul Elo)."""
        cache_key = f"football:history:{team_id}:{n}"
        cached = cache_get(cache_key)
        if cached is not None:
            return cached

        url = f"{FOOTBALL_BASE_URL}/teams/{team_id}/matches"
        params = {"status": "FINISHED", "limit": n}
        data = self._safe_request(url, headers=self.headers, params=params)

        if data and "matches" in data:
            matches = [self._normalize_match(m) for m in data["matches"][-n:]]
            cache_set(cache_key, matches, ttl=CACHE_TTL["history"])
            return matches
        return []

    def get_team_matches_bulk(self, team_id: int | str, limit: int = 50) -> list[dict]:
        """Récupère un grand volume de matchs pour l'entraînement Elo."""
        cache_key = f"football:bulk:{team_id}:{limit}"
        cached = cache_get(cache_key)
        if cached is not None:
            return cached

        url = f"{FOOTBALL_BASE_URL}/teams/{team_id}/matches"
        params = {"status": "FINISHED", "limit": limit}
        data = self._safe_request(url, headers=self.headers, params=params)

        if data and "matches" in data:
            matches = [self._normalize_match(m) for m in data["matches"]]
            cache_set(cache_key, matches, ttl=CACHE_TTL["history"])
            return matches
        return []

    def get_competition_history(self, competition_code: str) -> list[dict]:
        """Récupère directement depuis l'API les résultats terminés d'une compétition."""
        code = competition_code.upper()
        if code not in FOOTBALL_HISTORY_COMPETITIONS or not self.api_key:
            return []

        cached = _FOOTBALL_HISTORY_API_CACHE.get(code)
        if cached and time.monotonic() - cached[0] < UPCOMING_REFRESH_INTERVAL_SECONDS:
            return cached[1]

        url = f"{FOOTBALL_BASE_URL}/competitions/{code}/matches"
        data = self._safe_request(
            url,
            headers=self.headers,
            params={"status": "FINISHED"},
        )
        if data is None or "matches" not in data:
            _FOOTBALL_HISTORY_API_CACHE[code] = (time.monotonic(), [])
            return []

        matches = []
        for raw_match in data["matches"]:
            if raw_match.get("status") != "FINISHED":
                continue
            match = self._normalize_match(raw_match)
            if match["home_score"] is not None and match["away_score"] is not None:
                matches.append(match)

        _FOOTBALL_HISTORY_API_CACHE[code] = (time.monotonic(), matches)
        logger.info(
            "Football: %d résultats historiques récupérés pour %s",
            len(matches),
            code,
        )
        return matches

# ──────────────────────── Client Basketball ────────────────────────────────


class BasketballClient(BaseClient):
    """
    Client pour l'API balldontlie (NBA).
    Documentation : https://www.balldontlie.io/
    Note: Le plan gratuit ne nécessite pas de clé API (limite: 30 req/min).
    """

    def __init__(self):
        self.api_key = os.getenv("BALLDONTLIE_API_KEY", "")
        self.headers = {}
        if self.api_key:
            self.headers["Authorization"] = self.api_key

    # ── Normalisation ──────────────────────────────────────────────────────

    def _normalize_game(self, raw: dict) -> dict:
        """Normalise un match brut de balldontlie en format standard."""
        home = raw.get("home_team", {})
        away = raw.get("visitor_team", {})

        # Calcul du temps restant (quarter + time)
        period = raw.get("period", 0)
        time_str = raw.get("time", "")
        time_remaining = self._parse_time_remaining(period, time_str)

        return {
            "id": raw.get("id"),
            "sport": "basketball",
            "home_team": home.get("full_name", home.get("name", "?")),
            "away_team": away.get("full_name", away.get("name", "?")),
            "home_team_id": home.get("id"),
            "away_team_id": away.get("id"),
            "home_score": raw.get("home_team_score", 0),
            "away_score": raw.get("visitor_team_score", 0),
            "status": raw.get("status", ""),
            "period": period,
            "time": time_str,
            "time_remaining_secs": time_remaining,
            "match_date": raw.get("date", "")[:10],
            "competition": "NBA",
            "raw": raw,
        }

    def _parse_time_remaining(self, period: int, time_str: str) -> int:
        """
        Calcule le temps total restant en secondes dans un match NBA.
        Un match NBA dure 4 quarts de 12 minutes (+ OT si besoin).
        """
        QUARTER_SECS = 12 * 60   # 720 secondes par quart
        TOTAL_SECS = 4 * QUARTER_SECS  # 2880 secondes au total

        if period == 0 or not time_str:
            return TOTAL_SECS  # Match pas encore commencé

        # Parse "MM:SS"
        try:
            parts = time_str.split(":")
            mins = int(parts[0]) if len(parts) > 0 else 0
            secs = int(parts[1]) if len(parts) > 1 else 0
            current_quarter_remaining = mins * 60 + secs
        except (ValueError, IndexError):
            current_quarter_remaining = 0

        # Quarts restants après le quart actuel
        quarters_left = max(0, 4 - period)
        return quarters_left * QUARTER_SECS + current_quarter_remaining

    # ── Interface publique ─────────────────────────────────────────────────

    def get_today_matches(self) -> list[dict]:
        """Retourne les matchs NBA du jour."""
        today = date.today().isoformat()
        cache_key = f"basketball:today:{today}"

        cached = cache_get(cache_key)
        if cached is not None:
            return cached

        url = f"{BALLDONTLIE_BASE_URL}/games"
        params = {"dates[]": today, "per_page": 100}
        data = self._safe_request(url, headers=self.headers, params=params)

        if data and "data" in data:
            matches = [self._normalize_game(g) for g in data["data"]]
            cache_set(cache_key, matches, ttl=_match_list_cache_ttl(matches))
            logger.info("Basketball: %d matchs aujourd'hui", len(matches))
            return matches
        return []

    def get_live_match(self, match_id: int | str) -> dict | None:
        """Retourne l'état actuel d'un match NBA."""
        cache_key = f"basketball:live:{match_id}"
        cached = cache_get(cache_key)
        if cached is not None:
            return cached

        url = f"{BALLDONTLIE_BASE_URL}/games/{match_id}"
        data = self._safe_request(url, headers=self.headers)

        if data and "data" in data:
            normalized = self._normalize_game(data["data"])
            cache_set(cache_key, normalized, ttl=CACHE_TTL["live_match"])
            return normalized
        return None

    def get_history(self, team_id: int | str, n: int = 20) -> list[dict]:
        """Récupère directement les n derniers matchs d'une équipe NBA via l'API."""
        # Saison en cours (NBA saison commence en octobre)
        current_year = date.today().year
        season = current_year if date.today().month >= 10 else current_year - 1
        cache_key = (str(team_id), season, n)
        cached = _BASKETBALL_HISTORY_API_CACHE.get(cache_key)
        if cached and time.monotonic() - cached[0] < UPCOMING_REFRESH_INTERVAL_SECONDS:
            return cached[1]

        url = f"{BALLDONTLIE_BASE_URL}/games"
        params = {
            "team_ids[]": team_id,
            "seasons[]": season,
            "per_page": n,
        }
        data = self._safe_request(url, headers=self.headers, params=params)

        if data and "data" in data:
            matches = [self._normalize_game(g) for g in data["data"][:n]]
        else:
            matches = []
        _BASKETBALL_HISTORY_API_CACHE[cache_key] = (time.monotonic(), matches)
        return matches

    def get_all_teams(self) -> list[dict]:
        """Retourne la liste de toutes les équipes NBA."""
        cache_key = "basketball:teams"
        cached = cache_get(cache_key)
        if cached is not None:
            return cached

        url = f"{BALLDONTLIE_BASE_URL}/teams"
        params = {"per_page": 100}
        data = self._safe_request(url, headers=self.headers, params=params)

        if data and "data" in data:
            teams = data["data"]
            cache_set(cache_key, teams, ttl=86400)  # 24h
            return teams
        return []


# ──────────────────────── Client Odds (Cotes) ──────────────────────────────

# Mapping des compétitions football-data.org vers les sport-keys The Odds API
FOOTBALL_TO_ODDS_KEY = {
    "PL":  "soccer_epl",
    "PD":  "soccer_spain_la_liga",
    "FL1": "soccer_france_ligue_one",
    "BL1": "soccer_germany_bundesliga",
    "SA":  "soccer_italy_serie_a",
    "DED": "soccer_netherlands_eredivisie",
    "PPL": "soccer_portugal_primeira_liga",
    "ELC": "soccer_efl_champ",
    "BSA": "soccer_brazil_campeonato",
    "CL":  "soccer_uefa_champs_league",
    "CLI": "soccer_conmebol_copa_libertadores",
    "WC":  "soccer_fifa_world_cup",
    "EC":  "soccer_uefa_european_championship",
    "CAN": "soccer_caf_nations_cup",
}


class OddsClient(BaseClient):
    """
    Client pour The Odds API (https://the-odds-api.com/).
    Plan gratuit : 500 requêtes/mois. Récupère les cotes décimales
    des bookmakers pour calculer la valeur espérée (EV).
    """

    def __init__(self):
        self.api_key = os.getenv("ODDS_API_KEY", "")
        if not self.api_key:
            logger.warning("ODDS_API_KEY non définie dans .env — les cotes ne seront pas disponibles.")

    def _is_available(self) -> bool:
        """Vérifie si la clé API est configurée."""
        return bool(self.api_key)

    # ── Interface publique ─────────────────────────────────────────────────

    def get_today_matches(self) -> list[dict]:
        """Non utilisé directement — on passe par get_odds_for_sport."""
        return []

    def get_live_match(self, match_id: int | str) -> dict | None:
        """Non disponible via Odds API."""
        return None

    def get_history(self, team: str, n: int = 20) -> list[dict]:
        """Non disponible via Odds API."""
        return []

    def get_odds_for_sport(self, sport_key: str, regions: str = "eu") -> list[dict]:
        """
        Récupère les cotes des prochains matchs pour un sport donné.

        Args:
            sport_key: ex. 'soccer_epl', 'basketball_nba'
            regions: 'eu' (cotes européennes décimales) ou 'us' / 'uk'

        Returns:
            Liste de matchs avec cotes normalisées :
            [
                {
                    "id": "abc123",
                    "home_team": "...", "away_team": "...",
                    "commence_time": "...",
                    "bookmakers": [...],
                    "best_odds": {"home_win": 2.10, "draw": 3.40, "away_win": 4.50},
                    "avg_odds": {"home_win": 2.05, "draw": 3.30, "away_win": 4.20},
                }
            ]
        """
        if not self._is_available():
            return []

        cache_key = f"odds:{sport_key}:{regions}"
        cached = cache_get(cache_key)
        if cached is not None:
            return cached

        url = f"{ODDS_BASE_URL}/sports/{sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": "h2h",         # 1X2 (home/draw/away)
            "oddsFormat": "decimal",
        }
        data = self._safe_request(url, params=params)

        if not data or not isinstance(data, list):
            return []

        results = []
        for event in data:
            normalized = self._normalize_odds(event)
            if normalized:
                results.append(normalized)

        cache_set(cache_key, results, ttl=600)  # 10 minutes
        logger.info("Odds: %d événements pour %s", len(results), sport_key)
        return results

    def get_football_odds(self) -> dict[str, list[dict]]:
        """
        Récupère les cotes de toutes les compétitions de football suivies.
        Retourne un dict {competition_code: [odds_events]}.
        """
        all_odds = {}
        for comp_code, odds_key in FOOTBALL_TO_ODDS_KEY.items():
            events = self.get_odds_for_sport(odds_key)
            if events:
                all_odds[comp_code] = events
        return all_odds

    def get_basketball_odds(self) -> list[dict]:
        """Récupère les cotes NBA."""
        return self.get_odds_for_sport("basketball_nba")

    # ── Normalisation ──────────────────────────────────────────────────────

    def _normalize_odds(self, event: dict) -> dict | None:
        """Normalise un événement brut de The Odds API."""
        bookmakers = event.get("bookmakers", [])
        if not bookmakers:
            return None

        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")

        # Extraire les meilleures cotes et les cotes moyennes
        all_home, all_draw, all_away = [], [], []

        for bk in bookmakers:
            for market in bk.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                if home_team in outcomes:
                    all_home.append(outcomes[home_team])
                if away_team in outcomes:
                    all_away.append(outcomes[away_team])
                if "Draw" in outcomes:
                    all_draw.append(outcomes["Draw"])

        best_odds = {
            "home_win": max(all_home) if all_home else 0,
            "draw":     max(all_draw) if all_draw else 0,
            "away_win": max(all_away) if all_away else 0,
        }
        avg_odds = {
            "home_win": round(sum(all_home) / len(all_home), 2) if all_home else 0,
            "draw":     round(sum(all_draw) / len(all_draw), 2) if all_draw else 0,
            "away_win": round(sum(all_away) / len(all_away), 2) if all_away else 0,
        }

        return {
            "id":              event.get("id"),
            "home_team":       home_team,
            "away_team":       away_team,
            "commence_time":   event.get("commence_time", ""),
            "sport_key":       event.get("sport_key", ""),
            "n_bookmakers":    len(bookmakers),
            "best_odds":       best_odds,
            "avg_odds":        avg_odds,
        }

    def match_odds_to_match(
        self,
        match: dict,
        odds_events: list[dict],
        threshold: float = 0.8,
    ) -> dict | None:
        """
        Associe un match (football-data.org) à ses cotes (The Odds API)
        par fuzzy matching sur les noms d'équipes.

        Args:
            match: match normalisé depuis FootballClient
            odds_events: liste d'odds depuis OddsClient
            threshold: seuil de similarité (0 à 1)

        Returns:
            Le dict odds correspondant, ou None si pas de match.
        """
        home = match.get("home_team", "").lower()
        away = match.get("away_team", "").lower()

        best_match = None
        best_score = 0

        for odds_event in odds_events:
            oh = odds_event.get("home_team", "").lower()
            oa = odds_event.get("away_team", "").lower()

            match_time = (
                match.get("kickoff_utc")
                or match.get("match_date")
            )
            odds_time = odds_event.get("commence_time")
            match_date = self._event_date(match_time)
            odds_date = self._event_date(odds_time)
            if (
                match_date is not None
                and odds_date is not None
                and abs((match_date - odds_date).days) > 1
            ):
                continue

            # Score de similarité simple basé sur les mots communs
            score = self._name_similarity(home, oh) + self._name_similarity(away, oa)

            if score > best_score:
                best_score = score
                best_match = odds_event

        if best_score >= threshold:
            return best_match
        return None

    @staticmethod
    def _event_date(value: str | None) -> date | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            try:
                return date.fromisoformat(value[:10])
            except ValueError:
                return None
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc)
        return parsed.date()

    @staticmethod
    def _name_similarity(name_a: str, name_b: str) -> float:
        """
        Similarité entre deux noms d'équipes basée sur les mots en commun.
        Retourne un score de 0 à 1.
        """
        words_a = set(name_a.lower().replace("fc", "").replace("cf", "").split())
        words_b = set(name_b.lower().replace("fc", "").replace("cf", "").split())

        if not words_a or not words_b:
            return 0.0

        common = words_a & words_b
        return len(common) / max(len(words_a), len(words_b))
