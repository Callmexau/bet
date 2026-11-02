# -*- coding: utf-8 -*-
"""
backtest.py - Évaluation rétrospective des modèles de prédiction

Rejoue les 200 derniers matchs historiques et calcule :
  - Précision globale (accuracy) : le pronostique le plus probable était-il correct ?
  - Brier Score : mesure calibrée de la qualité des probabilités
    (0 = parfait, 1 = toujours faux, 0.25 = prédiction aléatoire)

Utilisation :
    python backtest.py                  # Backtest football uniquement
    python backtest.py --sport basket   # Backtest basketball
    python backtest.py --sport all      # Les deux sports
    python backtest.py --n 100          # 100 matchs au lieu de 200
"""

import argparse
import logging
import json
from datetime import datetime
import pandas as pd
import numpy as np

from db import get_history_from_db
from models.football import FootballModel
from models.basketball import BasketballModel
from models.calibration import TemperatureCalibrator, PlattCalibrator
from models.features import competition_code

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────── Fonctions de scoring ──────────────────────────────


def brier_score(y_true_probs: list[float], y_pred_probs: list[float]) -> float:
    """
    Calcule le Brier Score moyen.
    y_true_probs : probabilités réelles (0 ou 1 selon résultat)
    y_pred_probs : probabilités prédites
    Brier Score = mean((p_pred - p_real)^2)
    """
    if not y_true_probs or not y_pred_probs:
        return float("nan")
    arr_true = np.array(y_true_probs, dtype=float)
    arr_pred = np.array(y_pred_probs, dtype=float)
    return float(np.mean((arr_pred - arr_true) ** 2))


def accuracy(predictions: list[str], actuals: list[str]) -> float:
    """Calcule la précision simple (% de prédictions correctes)."""
    if not predictions:
        return 0.0
    correct = sum(p == a for p, a in zip(predictions, actuals))
    return correct / len(predictions)


def get_actual_result(match: dict) -> str | None:
    """Détermine le résultat réel d'un match (H/D/A)."""
    hs = match.get("home_score")
    as_ = match.get("away_score")
    if hs is None or as_ is None:
        return None
    if hs > as_:
        return "H"
    elif hs < as_:
        return "A"
    return "D"


def get_predicted_result(probs: dict) -> str:
    """Retourne le résultat le plus probable selon les probabilités."""
    hw = probs.get("home_win", 0)
    d  = probs.get("draw", 0)
    aw = probs.get("away_win", 0)
    if hw >= d and hw >= aw:
        return "H"
    if aw >= d and aw >= hw:
        return "A"
    return "D"


def simulate_opening_odds_bet(
    probs: dict,
    actual: str | None,
    raw_data: str | dict | None,
    min_ev: float = 0.03,
) -> dict | None:
    """Simule une mise fixe sur le meilleur EV des cotes moyennes d'ouverture."""
    if not actual or not raw_data:
        return None
    match_data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
    odds = match_data.get("historical_odds", {}).get("average_opening")
    if not odds:
        return None

    outcome_by_market = {
        "home_win": "H",
        "draw": "D",
        "away_win": "A",
    }
    candidates = []
    for market, outcome in outcome_by_market.items():
        try:
            odd = float(odds[market])
            probability = float(probs[market])
        except (KeyError, TypeError, ValueError):
            continue
        if odd > 1:
            candidates.append((probability * odd - 1, outcome, odd))

    if not candidates:
        return None
    ev, outcome, odd = max(candidates)
    if ev < min_ev:
        return None

    return {
        "outcome": outcome,
        "odds": odd,
        "ev": ev,
        "profit_units": odd - 1 if outcome == actual else -1.0,
    }


# ─────────────────────── Backtest Football ─────────────────────────────────


def backtest_football(n: int = 200, verbose: bool = True) -> dict:
    """
    Rejoue les n derniers matchs de football avec walk-forward validation.
    Le modèle est entraîné sur les matchs précédents, puis prédit le suivant.
    """
    logger.info("=== BACKTEST FOOTBALL (n=%d) ===", n)

    matches = get_history_from_db("football", limit=n + 50)
    if not matches:
        logger.error("Aucun match football en base. Lancez d'abord l'application pour peupler le cache.")
        return {}

    # Tri chronologique
    matches = sorted(
        [m for m in matches if get_actual_result(m) is not None],
        key=lambda x: x.get("match_date", "")
    )

    if len(matches) < 10:
        logger.warning("Pas assez de données (%d matchs). Besoin d'au moins 10.", len(matches))

    # Walk-forward : on entraîne sur les k premiers matchs et on prédit le (k+1)-ème
    WARMUP = min(50, len(matches) // 2)  # Matchs de warmup pour initialiser Elo
    test_matches = matches[WARMUP:WARMUP + n]

    predictions_result = []  # H/D/A prédit
    actuals_result = []       # H/D/A réel
    brier_home = []           # Pour Brier Score home
    brier_draw = []
    brier_away = []
    details = []
    calibrator = TemperatureCalibrator()
    cal_probs: list[list[float]] = []
    cal_y: list[str] = []
    value_bets = []

    for i, test_match in enumerate(test_matches):
        # Entraîner sur tous les matchs précédant ce test
        train_data = matches[:WARMUP + i]
        model = FootballModel()
        model.train(train_data, tune=False, calibrate=False)

        # Prédiction avant-match (on ne connaît pas le score)
        probs = model.predict_prematch(
            test_match.get("home_team", ""),
            test_match.get("away_team", ""),
            comp_code=competition_code(test_match),
            match_date=test_match.get("match_date"),
            match=test_match,
            calibrate=False,
        )
        raw_p = [probs.get("home_win", 0), probs.get("draw", 0), probs.get("away_win", 0)]
        if len(cal_probs) >= 20:
            calibrator.fit(np.array(cal_probs), cal_y)
            h, d, a = calibrator.transform(*raw_p)
            probs = {**probs, "home_win": h, "draw": d, "away_win": a}

        actual = get_actual_result(test_match)
        predicted = get_predicted_result(probs)
        simulated_bet = simulate_opening_odds_bet(
            probs, actual, test_match.get("raw_data"),
        )
        if simulated_bet:
            value_bets.append(simulated_bet)

        cal_probs.append(raw_p)
        cal_y.append(actual or "D")

        predictions_result.append(predicted)
        actuals_result.append(actual)

        # Vecteurs one-hot pour Brier Score
        brier_home.append((1.0 if actual == "H" else 0.0, probs.get("home_win", 0)))
        brier_draw.append((1.0 if actual == "D" else 0.0, probs.get("draw", 0)))
        brier_away.append((1.0 if actual == "A" else 0.0, probs.get("away_win", 0)))

        details.append({
            "date": test_match.get("match_date"),
            "home": test_match.get("home_team"),
            "away": test_match.get("away_team"),
            "score": f"{test_match.get('home_score')}-{test_match.get('away_score')}",
            "actual": actual,
            "predicted": predicted,
            "p_home": round(probs.get("home_win", 0), 3),
            "p_draw": round(probs.get("draw", 0), 3),
            "p_away": round(probs.get("away_win", 0), 3),
            "correct": actual == predicted,
        })

    if not details:
        logger.warning("Aucun match testé.")
        return {}

    # Calcul des métriques
    acc = accuracy(predictions_result, actuals_result)

    # Brier Score multiclasse = moyenne des Brier Scores par classe
    bs_home = brier_score([x[0] for x in brier_home], [x[1] for x in brier_home])
    bs_draw = brier_score([x[0] for x in brier_draw], [x[1] for x in brier_draw])
    bs_away = brier_score([x[0] for x in brier_away], [x[1] for x in brier_away])
    bs_avg  = (bs_home + bs_draw + bs_away) / 3
    roi_net_units = sum(bet["profit_units"] for bet in value_bets)
    roi_pct = roi_net_units / len(value_bets) * 100 if value_bets else None

    # Distribution des erreurs
    df = pd.DataFrame(details)
    correct_by_type = df.groupby("actual")["correct"].agg(["sum", "count"])

    results = {
        "sport": "football",
        "n_matches": len(details),
        "accuracy": round(acc, 4),
        "brier_score": round(bs_avg, 4),
        "brier_home": round(bs_home, 4),
        "brier_draw": round(bs_draw, 4),
        "brier_away": round(bs_away, 4),
        "value_bets": len(value_bets),
        "roi_net_units": round(roi_net_units, 2),
        "roi_pct": round(roi_pct, 2) if roi_pct is not None else None,
        "roi_min_ev": 0.03,
        "correct_by_outcome": correct_by_type.to_dict(),
    }

    if verbose:
        print("\n" + "=" * 60)
        print("  📊 RÉSULTATS BACKTEST FOOTBALL")
        print("=" * 60)
        print(f"  Matchs testés    : {len(details)}")
        print(f"  Précision        : {acc:.1%}")
        print(f"  Brier Score      : {bs_avg:.4f}  (plus bas = mieux; comparer à une baseline)")
        print(f"    ↳ Home         : {bs_home:.4f}")
        print(f"    ↳ Nul          : {bs_draw:.4f}")
        print(f"    ↳ Away         : {bs_away:.4f}")
        if value_bets:
            print(
                f"  ROI simulé      : {roi_pct:+.1f}% "
                f"({len(value_bets)} mises fixes, cotes moyennes d'ouverture, EV ≥ 3%)"
            )
        else:
            print("  ROI simulé      : indisponible (pas de cotes d'ouverture exploitables)")
        print()
        print("  Précision par type de résultat :")
        for outcome, row in correct_by_type.iterrows():
            label = {"H": "Victoire domicile", "D": "Nul", "A": "Victoire extérieur"}
            pct = row["sum"] / row["count"] if row["count"] > 0 else 0
            print(f"    {label.get(outcome, outcome):25s}: {pct:.1%} ({int(row['sum'])}/{int(row['count'])})")

        print()
        print("  Derniers matchs (5) :")
        for d in details[-5:]:
            ok = "✅" if d["correct"] else "❌"
            print(f"    {ok} {d['date']} | {d['home']} vs {d['away']} | "
                  f"{d['score']} ({d['actual']}) | prédit: {d['predicted']} | "
                  f"probs: {d['p_home']}/{d['p_draw']}/{d['p_away']}")
        print("=" * 60)

    return results


# ─────────────────────── Backtest Basketball ───────────────────────────────


def backtest_basketball(n: int = 200, verbose: bool = True) -> dict:
    """
    Rejoue les n derniers matchs NBA avec walk-forward validation.
    """
    logger.info("=== BACKTEST BASKETBALL (n=%d) ===", n)

    matches = get_history_from_db("basketball", limit=n + 50)
    if not matches:
        logger.error("Aucun match basketball en base. Lancez d'abord l'application pour peupler le cache.")
        return {}

    matches = sorted(
        [m for m in matches if get_actual_result(m) is not None],
        key=lambda x: x.get("match_date", "")
    )

    if len(matches) < 10:
        logger.warning("Pas assez de données (%d matchs).", len(matches))

    WARMUP = min(30, len(matches) // 2)
    test_matches = matches[WARMUP:WARMUP + n]

    predictions_result = []
    actuals_result = []
    brier_home = []
    brier_away = []
    details = []
    calibrator = PlattCalibrator()
    cal_p: list[float] = []
    cal_y: list[float] = []

    for i, test_match in enumerate(test_matches):
        train_data = matches[:WARMUP + i]
        model = BasketballModel()
        model.train(train_data, calibrate=False)

        probs = model.predict_prematch(
            test_match.get("home_team", ""),
            test_match.get("away_team", ""),
            match_date=test_match.get("match_date"),
            calibrate=False,
        )
        raw_ph = probs.get("home_win", 0.5)
        if len(cal_p) >= 20:
            calibrator.fit(np.array(cal_p), np.array(cal_y))
            ph = calibrator.transform(raw_ph)
            probs = {**probs, "home_win": ph, "away_win": 1.0 - ph}

        actual = get_actual_result(test_match)
        predicted = "H" if probs.get("home_win", 0.5) >= 0.5 else "A"
        cal_p.append(raw_ph)
        cal_y.append(1.0 if actual == "H" else 0.0)

        predictions_result.append(predicted)
        actuals_result.append(actual)

        brier_home.append((1.0 if actual == "H" else 0.0, probs.get("home_win", 0.5)))
        brier_away.append((1.0 if actual == "A" else 0.0, probs.get("away_win", 0.5)))

        details.append({
            "date": test_match.get("match_date"),
            "home": test_match.get("home_team"),
            "away": test_match.get("away_team"),
            "score": f"{test_match.get('home_score')}-{test_match.get('away_score')}",
            "actual": actual,
            "predicted": predicted,
            "p_home": round(probs.get("home_win", 0.5), 3),
            "p_away": round(probs.get("away_win", 0.5), 3),
            "correct": actual == predicted,
        })

    if not details:
        return {}

    acc = accuracy(predictions_result, actuals_result)
    bs_home = brier_score([x[0] for x in brier_home], [x[1] for x in brier_home])
    bs_away = brier_score([x[0] for x in brier_away], [x[1] for x in brier_away])
    bs_avg  = (bs_home + bs_away) / 2

    results = {
        "sport": "basketball",
        "n_matches": len(details),
        "accuracy": round(acc, 4),
        "brier_score": round(bs_avg, 4),
    }

    if verbose:
        df = pd.DataFrame(details)
        h_correct = df[df["actual"] == "H"]["correct"].mean()
        a_correct = df[df["actual"] == "A"]["correct"].mean()

        print("\n" + "=" * 60)
        print("  📊 RÉSULTATS BACKTEST BASKETBALL")
        print("=" * 60)
        print(f"  Matchs testés    : {len(details)}")
        print(f"  Précision        : {acc:.1%}")
        print(f"  Brier Score      : {bs_avg:.4f}")
        print(f"    ↳ Home win     : {bs_home:.4f}")
        print(f"    ↳ Away win     : {bs_away:.4f}")
        print()
        print(f"  Domicile gagné et prédit: {h_correct:.1%}")
        print(f"  Extérieur gagné et prédit: {a_correct:.1%}")
        print("=" * 60)

    return results


# ─────────────────────── Point d'entrée ────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Backtest des modèles de pronostics sportifs"
    )
    parser.add_argument(
        "--sport",
        choices=["football", "basket", "all"],
        default="football",
        help="Sport à backtester (défaut: football)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Nombre de matchs à tester (défaut: 200)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Désactive l'affichage détaillé",
    )
    args = parser.parse_args()

    print(f"\n🎯 SportsPredikt — Backtest lancé le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Sport: {args.sport} | Matchs: {args.n}")

    all_results = {}

    if args.sport in ("football", "all"):
        r = backtest_football(n=args.n, verbose=not args.quiet)
        all_results["football"] = r

    if args.sport in ("basket", "all"):
        r = backtest_basketball(n=args.n, verbose=not args.quiet)
        all_results["basketball"] = r

    # Sauvegarde des résultats
    output_file = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Résultats sauvegardés dans {output_file}")


if __name__ == "__main__":
    main()
