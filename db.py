# -*- coding: utf-8 -*-
"""
db.py - Gestion de la base de données SQLite
Cache des réponses API pour limiter les appels et améliorer les performances.
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "cache.db"

# Durées de cache par type de données (en secondes)
CACHE_TTL = {
    "today_matches": 300,       # 5 minutes
    "live_match": 30,           # 30 secondes (données en direct)
    "history": 3600,            # 1 heure
    "standings": 1800,          # 30 minutes
}


def get_connection() -> sqlite3.Connection:
    """Retourne une connexion SQLite avec Row Factory activée."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def managed_connection():
    """Context manager pour les connexions SQLite."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Erreur DB: {e}")
        raise
    finally:
        conn.close()


def init_db():
    """Initialise les tables SQLite si elles n'existent pas."""
    with managed_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS api_cache (
                cache_key   TEXT PRIMARY KEY,
                data        TEXT NOT NULL,
                cached_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                ttl_seconds INTEGER NOT NULL DEFAULT 300
            );

            CREATE TABLE IF NOT EXISTS match_history (
                id          TEXT PRIMARY KEY,
                sport       TEXT NOT NULL,
                home_team   TEXT NOT NULL,
                away_team   TEXT NOT NULL,
                home_score  INTEGER,
                away_score  INTEGER,
                match_date  TEXT NOT NULL,
                competition TEXT,
                raw_data    TEXT
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id        TEXT NOT NULL,
                sport           TEXT NOT NULL,
                minute          INTEGER,
                home_win_prob   REAL,
                draw_prob       REAL,
                away_win_prob   REAL,
                predicted_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_cache_key ON api_cache(cache_key);
            CREATE INDEX IF NOT EXISTS idx_match_history_sport ON match_history(sport);
            CREATE INDEX IF NOT EXISTS idx_predictions_match ON predictions(match_id);
        """)
    logger.info("Base de données initialisée.")


def cache_get(key: str) -> dict | None:
    """
    Récupère une valeur du cache si elle n'est pas expirée.
    Retourne None si absente ou expirée.
    """
    with managed_connection() as conn:
        row = conn.execute(
            """
            SELECT data, cached_at, ttl_seconds
            FROM api_cache
            WHERE cache_key = ?
            """,
            (key,)
        ).fetchone()

        if row is None:
            return None

        cached_at = datetime.fromisoformat(row["cached_at"])
        expiry = cached_at + timedelta(seconds=row["ttl_seconds"])

        if datetime.utcnow() > expiry:
            # Cache expiré, on le supprime
            conn.execute("DELETE FROM api_cache WHERE cache_key = ?", (key,))
            return None

        return json.loads(row["data"])


def cache_set(key: str, data: dict, ttl: int = 300):
    """
    Stocke une valeur dans le cache avec un TTL (en secondes).
    """
    with managed_connection() as conn:
        conn.execute(
            """
            INSERT INTO api_cache (cache_key, data, cached_at, ttl_seconds)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                data = excluded.data,
                cached_at = excluded.cached_at,
                ttl_seconds = excluded.ttl_seconds
            """,
            (key, json.dumps(data, ensure_ascii=False), datetime.utcnow().isoformat(), ttl)
        )


def save_match(match: dict, sport: str):
    """
    Persiste un match terminé dans l'historique pour le backtest.
    """
    with managed_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO match_history
                (id, sport, home_team, away_team, home_score, away_score, match_date, competition, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(match.get("id")),
                sport,
                match.get("home_team", ""),
                match.get("away_team", ""),
                match.get("home_score"),
                match.get("away_score"),
                match.get("match_date", ""),
                match.get("competition", ""),
                json.dumps(match, ensure_ascii=False),
            )
        )


def save_prediction(match_id: str, sport: str, minute: int | None, probs: dict):
    """Sauvegarde une prédiction pour analyse ultérieure."""
    with managed_connection() as conn:
        conn.execute(
            """
            INSERT INTO predictions (match_id, sport, minute, home_win_prob, draw_prob, away_win_prob)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(match_id),
                sport,
                minute,
                probs.get("home_win", 0),
                probs.get("draw", 0),
                probs.get("away_win", 0),
            )
        )


def get_history_from_db(sport: str, limit: int = 200) -> list[dict]:
    """Récupère l'historique des matchs depuis la DB pour le backtest."""
    with managed_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM match_history
            WHERE sport = ?
            ORDER BY match_date DESC
            LIMIT ?
            """,
            (sport, limit)
        ).fetchall()
        return [dict(row) for row in rows]


# Initialisation automatique au premier import
init_db()
