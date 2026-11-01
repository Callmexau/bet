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

            CREATE TABLE IF NOT EXISTS bets (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id        TEXT,
                match_name      TEXT NOT NULL,
                sport           TEXT DEFAULT 'football',
                competition     TEXT,
                market          TEXT NOT NULL,
                pick            TEXT NOT NULL,
                odds            REAL NOT NULL,
                stake_fcfa      REAL NOT NULL,
                status          TEXT DEFAULT 'PENDING',  -- PENDING, WON, LOST, VOID
                payout_fcfa     REAL DEFAULT 0.0,
                created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                settled_at      TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_cache_key ON api_cache(cache_key);
            CREATE INDEX IF NOT EXISTS idx_match_history_sport ON match_history(sport);
            CREATE INDEX IF NOT EXISTS idx_predictions_match ON predictions(match_id);
            CREATE INDEX IF NOT EXISTS idx_bets_status ON bets(status);
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


# ────────────────────── Gestion du Journal de Paris (Bankroll) ─────────────

def add_bet(
    match_name: str,
    competition: str,
    market: str,
    pick: str,
    odds: float,
    stake_fcfa: float,
    match_id: str | None = None,
    sport: str = "football",
) -> int:
    """Enregistre un nouveau pari dans le journal."""
    with managed_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO bets (match_id, match_name, sport, competition, market, pick, odds, stake_fcfa, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
            """,
            (match_id, match_name, sport, competition, market, pick, float(odds), float(stake_fcfa))
        )
        return cur.lastrowid


def get_bets(status: str | None = None) -> list[dict]:
    """Récupère tous les paris du journal, filtrés optionnellement par statut."""
    with managed_connection() as conn:
        if status and status != "ALL":
            rows = conn.execute(
                "SELECT * FROM bets WHERE status = ? ORDER BY id DESC",
                (status,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM bets ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]


def settle_bet(bet_id: int, status: str) -> bool:
    """
    Règle un pari (WON / LOST / VOID) et calcule le gain correspondant.
    """
    if status not in ("WON", "LOST", "VOID"):
        return False

    with managed_connection() as conn:
        row = conn.execute("SELECT odds, stake_fcfa FROM bets WHERE id = ?", (bet_id,)).fetchone()
        if not row:
            return False

        odds = row["odds"]
        stake = row["stake_fcfa"]

        if status == "WON":
            payout = round(stake * odds, 2)
        elif status == "VOID":
            payout = stake
        else:
            payout = 0.0

        conn.execute(
            """
            UPDATE bets
            SET status = ?, payout_fcfa = ?, settled_at = ?
            WHERE id = ?
            """,
            (status, payout, datetime.utcnow().isoformat(), bet_id)
        )
        return True


def delete_bet(bet_id: int) -> bool:
    """Supprime un pari du journal."""
    with managed_connection() as conn:
        conn.execute("DELETE FROM bets WHERE id = ?", (bet_id,))
        return True


def get_bankroll_stats() -> dict:
    """
    Calcule les métriques globales de rentabilité sur les paris terminés :
    Total misé, Total gains, Bénéfice net (FCFA), ROI (%), Taux de réussite (%).
    """
    with managed_connection() as conn:
        settled_rows = conn.execute(
            "SELECT status, stake_fcfa, payout_fcfa FROM bets WHERE status IN ('WON', 'LOST', 'VOID')"
        ).fetchall()
        pending_rows = conn.execute(
            "SELECT stake_fcfa FROM bets WHERE status = 'PENDING'"
        ).fetchall()

    total_settled_staked = sum(r["stake_fcfa"] for r in settled_rows if r["status"] != "VOID")
    total_payout = sum(r["payout_fcfa"] for r in settled_rows if r["status"] != "VOID")
    net_profit = round(total_payout - total_settled_staked, 2)
    roi = round((net_profit / total_settled_staked * 100), 1) if total_settled_staked > 0 else 0.0

    won_count = sum(1 for r in settled_rows if r["status"] == "WON")
    lost_count = sum(1 for r in settled_rows if r["status"] == "LOST")
    total_settled = won_count + lost_count
    win_rate = round(won_count / total_settled * 100, 1) if total_settled > 0 else 0.0

    pending_count = len(pending_rows)
    pending_staked = sum(r["stake_fcfa"] for r in pending_rows)

    return {
        "total_staked": total_settled_staked,
        "total_payout": total_payout,
        "net_profit": net_profit,
        "roi": roi,
        "won_count": won_count,
        "lost_count": lost_count,
        "win_rate": win_rate,
        "pending_count": pending_count,
        "pending_staked": pending_staked,
        "total_bets": len(settled_rows) + pending_count,
    }


# Initialisation automatique au premier import
init_db()
