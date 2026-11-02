# -*- coding: utf-8 -*-
"""Importe des résultats historiques Football-Data.co.uk dans SQLite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import math
from datetime import datetime
from pathlib import Path

from db import init_db, save_matches

logger = logging.getLogger(__name__)

DIVISIONS = {
    "E0": ("PL", "Premier League"),
    "E1": ("EN_CHAMPIONSHIP", "English Championship"),
    "E2": ("EN_LEAGUE_ONE", "English League One"),
    "E3": ("EN_LEAGUE_TWO", "English League Two"),
    "SC0": ("SC_PREMIERSHIP", "Scottish Premiership"),
    "SC1": ("SC_CHAMPIONSHIP", "Scottish Championship"),
    "D1": ("BL1", "Bundesliga"),
    "D2": ("DE_2_BUNDESLIGA", "2. Bundesliga"),
    "I1": ("SA", "Serie A"),
    "I2": ("IT_SERIE_B", "Serie B"),
    "SP1": ("PD", "La Liga"),
    "SP2": ("ES_SEGUNDA", "Segunda División"),
    "F1": ("FL1", "Ligue 1"),
    "F2": ("FR_LIGUE_2", "Ligue 2"),
    "N1": ("NL_EREDIVISIE", "Eredivisie"),
    "B1": ("BE_PRO_LEAGUE", "Belgian Pro League"),
    "P1": ("PT_PRIMEIRA", "Primeira Liga"),
    "T1": ("TR_SUPER_LIG", "Süper Lig"),
    "G1": ("GR_SUPER_LEAG", "Greek Super League"),
}

ODDS_COLUMNS = {
    "average_opening": ("AvgH", "AvgD", "AvgA"),
    "average_closing": ("AvgCH", "AvgCD", "AvgCA"),
}


def _parse_date(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None
    for date_format in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, date_format).date().isoformat()
        except ValueError:
            continue
    return None


def _parse_odds(row: dict[str, str], columns: tuple[str, str, str]) -> dict[str, float] | None:
    values = []
    for column in columns:
        try:
            value = float((row.get(column) or "").strip())
        except ValueError:
            return None
        if not math.isfinite(value) or value <= 1:
            return None
        values.append(value)
    return dict(zip(("home_win", "draw", "away_win"), values))


def _match_id(division: str, match_date: str, home: str, away: str) -> str:
    identity = "\x1f".join((division, match_date, home, away)).casefold()
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return f"fdc_{digest}"


def read_matches(path: Path) -> tuple[list[dict], int]:
    """Parse un CSV Football-Data.co.uk ; retourne les matchs terminés et ignorés."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            with path.open("r", encoding=encoding, newline="") as source:
                reader = csv.DictReader(source)
                if not reader.fieldnames:
                    raise ValueError(f"{path}: fichier CSV vide ou sans en-tête.")
                columns = {name.strip().lstrip("\ufeff") for name in reader.fieldnames if name}
                required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
                missing = required - columns
                if missing:
                    raise ValueError(
                        f"{path}: colonnes absentes {', '.join(sorted(missing))}; "
                        "fournissez un CSV Football-Data.co.uk."
                    )

                division = ""
                matches = []
                skipped = 0
                for line_number, row in enumerate(reader, start=2):
                    division = (row.get("Div") or "").strip().upper()
                    match_date = _parse_date(row.get("Date") or "")
                    home = (row.get("HomeTeam") or "").strip()
                    away = (row.get("AwayTeam") or "").strip()
                    home_goals = (row.get("FTHG") or "").strip()
                    away_goals = (row.get("FTAG") or "").strip()
                    if not match_date or not home or not away or not home_goals or not away_goals:
                        skipped += 1
                        continue

                    try:
                        home_score = int(home_goals)
                        away_score = int(away_goals)
                    except ValueError as error:
                        raise ValueError(
                            f"{path}, ligne {line_number}: scores invalides "
                            f"{home_goals!r}-{away_goals!r}."
                        ) from error
                    if home_score < 0 or away_score < 0:
                        raise ValueError(f"{path}, ligne {line_number}: score négatif.")

                    competition_code, competition = DIVISIONS.get(
                        division, (division or "UNKNOWN", division or "Football")
                    )
                    historical_odds = {
                        label: odds
                        for label, odds_columns in ODDS_COLUMNS.items()
                        if (odds := _parse_odds(row, odds_columns)) is not None
                    }
                    match = {
                        "id": _match_id(division, match_date, home, away),
                        "sport": "football",
                        "home_team": home,
                        "away_team": away,
                        "home_score": home_score,
                        "away_score": away_score,
                        "match_date": match_date,
                        "competition": competition,
                        "competition_code": competition_code,
                        "status": "FINISHED",
                        "historical_odds": historical_odds,
                        "source": "Football-Data.co.uk",
                    }
                    matches.append(match)
                return matches, skipped
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path}: impossible de décoder le fichier CSV.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Importe des CSV Football-Data.co.uk pour entraîner et évaluer le modèle."
    )
    parser.add_argument("files", nargs="+", type=Path, help="Un ou plusieurs fichiers CSV")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Vérifie les fichiers sans modifier la base de données.",
    )
    args = parser.parse_args()

    all_matches = []
    total_skipped = 0
    for path in args.files:
        if not path.is_file():
            parser.error(f"Fichier introuvable : {path}")
        matches, skipped = read_matches(path)
        all_matches.extend(matches)
        total_skipped += skipped
        print(f"{path}: {len(matches)} résultats terminés, {skipped} lignes ignorées")

    if args.dry_run:
        print(f"Vérification terminée : {len(all_matches)} résultats prêts à importer.")
        return 0

    init_db()
    save_matches(all_matches, "football")
    print(
        f"Import terminé : {len(all_matches)} résultats enregistrés "
        f"({total_skipped} lignes ignorées)."
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    raise SystemExit(main())
