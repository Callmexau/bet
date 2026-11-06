# -*- coding: utf-8 -*-
"""
Utilitaires partagés : dates, pondération temporelle, repos, forme, codes compétition.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, datetime

NEUTRAL_COMP_TOKENS = ("WC", "CAN", "EC", "WORLD CUP", "NATIONS", "EURO", "CLUB WORLD")

COMP_NAME_TO_CODE = {
    "PREMIER LEAGUE": "PL",
    "LA LIGA": "PD",
    "PRIMERA DIVISION": "PD",
    "LIGUE 1": "FL1",
    "BUNDESLIGA": "BL1",
    "SERIE A": "SA",
    "EREDIVISIE": "DED",
    "PRIMEIRA LIGA": "PPL",
    "CHAMPIONSHIP": "ELC",
    "CAMPEONATO BRASILEIRO SÉRIE A": "BSA",
    "CAMPEONATO BRASILEIRO SERIE A": "BSA",
    "UEFA CHAMPIONS LEAGUE": "CL",
    "CHAMPIONS LEAGUE": "CL",
    "COPA LIBERTADORES": "CLI",
    "FIFA WORLD CUP": "WC",
    "WORLD CUP": "WC",
    "EUROPEAN CHAMPIONSHIP": "EC",
    "EURO": "EC",
    "AFRICA CUP OF NATIONS": "CAN",
    "COUPE D'AFRIQUE DES NATIONS": "CAN",
    "COUPE D'AFRIQUE DES NATIONS (CAN)": "CAN",
}


def parse_date(value) -> date | None:
    """Parse une date de match (YYYY-MM-DD, ISO datetime, date/datetime)."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        if "T" in text or " " in text:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        return date.fromisoformat(text[:10])
    except (ValueError, TypeError):
        return None


def days_between(later, earlier) -> int | None:
    d1, d0 = parse_date(later), parse_date(earlier)
    if d1 is None or d0 is None:
        return None
    return (d1 - d0).days


def exp_decay_weight(days_ago: float, tau_days: float) -> float:
    """Poids exponentiel : 1 aujourd'hui, ~0.37 à tau jours."""
    tau = max(float(tau_days), 1.0)
    return math.exp(-max(days_ago, 0.0) / tau)


def competition_code(match: dict) -> str:
    code = (match.get("competition_code") or "").strip().upper()
    if code:
        return code
    name = (match.get("competition") or "").strip().upper()
    if name in COMP_NAME_TO_CODE:
        return COMP_NAME_TO_CODE[name]
    for key, val in COMP_NAME_TO_CODE.items():
        if key in name or name in key:
            return val
    return name[:8] if name else ""


def is_neutral_venue(match: dict) -> bool:
    blob = f"{competition_code(match)} {match.get('competition', '')}".upper()
    return any(tok in blob for tok in NEUTRAL_COMP_TOKENS)


def sort_matches(matches: list[dict]) -> list[dict]:
    def key(m: dict):
        d = parse_date(m.get("match_date"))
        return d or date.min

    return sorted(
        [
            m for m in matches
            if m.get("home_score") is not None and m.get("away_score") is not None
        ],
        key=key,
    )


def latest_date(matches: list[dict]) -> date | None:
    dates = [parse_date(m.get("match_date")) for m in matches]
    dates = [d for d in dates if d]
    return max(dates) if dates else None


def team_matches_before(history: list[dict], team: str, before) -> list[dict]:
    cutoff = parse_date(before)
    out = []
    for m in history:
        if m.get("home_team") != team and m.get("away_team") != team:
            continue
        if m.get("home_score") is None or m.get("away_score") is None:
            continue
        d = parse_date(m.get("match_date"))
        if cutoff and d and d >= cutoff:
            continue
        if cutoff and d is None:
            continue
        out.append(m)
    return sort_matches(out)


def rest_days(history: list[dict], team: str, as_of) -> float | None:
    """Jours de repos depuis le dernier match (None si inconnu)."""
    prev = team_matches_before(history, team, as_of)
    if not prev:
        return None
    return days_between(as_of, prev[-1].get("match_date"))


def recent_form(history: list[dict], team: str, as_of, n: int = 6) -> dict:
    """
    Forme courte : moyenne de la différence de buts et points / match
    sur les n derniers matchs avant as_of.
    """
    prev = team_matches_before(history, team, as_of)[-n:]
    if not prev:
        return {"n": 0, "gd_avg": 0.0, "ppg": 0.0}

    gd_sum = 0.0
    pts = 0.0
    for m in prev:
        hs, aws = int(m["home_score"]), int(m["away_score"])
        if m.get("home_team") == team:
            gd = hs - aws
            if hs > aws:
                pts += 3
            elif hs == aws:
                pts += 1
        else:
            gd = aws - hs
            if aws > hs:
                pts += 3
            elif aws == hs:
                pts += 1
        gd_sum += gd

    k = len(prev)
    return {"n": k, "gd_avg": gd_sum / k, "ppg": pts / k}


def rest_multiplier(days: float | None, *, sport: str = "football") -> float:
    """Ajuste légèrement λ / force selon la récupération."""
    if days is None:
        return 1.0
    if sport == "basketball":
        if days <= 1:
            return 0.97
        if days <= 2:
            return 0.99
        return 1.0
    if days <= 2:
        return 0.96
    if days <= 3:
        return 0.98
    if days >= 8:
        return 1.015
    return 1.0


def form_multiplier(form: dict, scale: float = 0.03) -> float:
    if not form or form.get("n", 0) < 3:
        return 1.0
    gd = max(-1.8, min(1.8, float(form["gd_avg"])))
    return 1.0 + scale * gd


def league_goal_averages(matches: list[dict], min_n: int = 8) -> dict[str, tuple[float, float]]:
    """Moyennes de buts domicile / extérieur par compétition."""
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    global_h, global_a = [], []
    for m in matches:
        hs, aws = m.get("home_score"), m.get("away_score")
        if hs is None or aws is None:
            continue
        code = competition_code(m) or "_global"
        buckets[code].append((float(hs), float(aws)))
        global_h.append(float(hs))
        global_a.append(float(aws))

    gh = sum(global_h) / len(global_h) if global_h else 1.45
    ga = sum(global_a) / len(global_a) if global_a else 1.15
    out: dict[str, tuple[float, float]] = {"_global": (gh, ga)}
    for code, rows in buckets.items():
        if len(rows) >= min_n:
            out[code] = (
                sum(r[0] for r in rows) / len(rows),
                sum(r[1] for r in rows) / len(rows),
            )
        else:
            out[code] = (gh, ga)
    return out


def nba_team_ratings(matches: list[dict], as_of=None, tau_days: float = 90.0) -> dict:
    """
    Ratings offensif / défensif NBA (points / match, pondérés par récence).
    """
    cutoff = parse_date(as_of)
    ref = cutoff or latest_date(matches) or date.today()
    stats: dict[str, dict] = defaultdict(lambda: {
        "w_pf": 0.0, "w_pa": 0.0, "w": 0.0, "games": 0, "last_date": None,
    })

    for m in sort_matches(matches):
        d = parse_date(m.get("match_date"))
        if cutoff and d and d >= cutoff:
            continue
        hs, aws = float(m["home_score"]), float(m["away_score"])
        days_ago = (ref - d).days if d else 60
        w = exp_decay_weight(days_ago, tau_days)
        for team, pf, pa in (
            (m["home_team"], hs, aws),
            (m["away_team"], aws, hs),
        ):
            s = stats[team]
            s["w_pf"] += w * pf
            s["w_pa"] += w * pa
            s["w"] += w
            s["games"] += 1
            if d and (s["last_date"] is None or d > s["last_date"]):
                s["last_date"] = d

    ratings = {}
    for team, s in stats.items():
        w = s["w"] or 1.0
        ratings[team] = {
            "off": s["w_pf"] / w,
            "def": s["w_pa"] / w,
            "games": s["games"],
            "last_date": s["last_date"],
        }
    return ratings
