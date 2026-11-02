# -*- coding: utf-8 -*-
"""
app.py - Tableau de bord de pronostics sportifs
Interface Streamlit avec onglets Football / Basketball,
rafraîchissement automatique toutes les 30 secondes,
et graphiques d'évolution des probabilités en direct.

Lancement : streamlit run app.py
"""

import time
import logging
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Configuration du logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Imports locaux ─────────────────────────────────────────────────────────
from data_client import (
    FootballClient, BasketballClient, OddsClient,
)
from weather_client import WeatherClient
from models.football import FootballModel
from models.basketball import BasketballModel
from db import (
    save_prediction, get_history_from_db,
    add_bet, get_bets, settle_bet, delete_bet, get_bankroll_stats
)
from telegram_alert import send_value_bets_alert, is_telegram_configured

# ═══════════════════════════════════════════════════════════════════════════
# Configuration Streamlit
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="⚽🏀 SportsPredikt — Pronostics en Direct",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styles de l'interface ─────────────────────────────────────────────────
st.markdown("""
<style>
  /* ════ Variables de design ════ */
  :root {
    --bg-0:       #101418;
    --bg-1:       #13181d;
    --bg-2:       #1a2026;
    --bg-3:       #20272e;
    --bg-4:       #273039;

    --green:      #55bd8a;
    --green-dim:  rgba(85,189,138,0.10);
    --green-border: rgba(85,189,138,0.28);
    --blue:       #82b4db;
    --blue-dim:   rgba(130,180,219,0.10);
    --blue-border:rgba(130,180,219,0.26);
    --yellow:     #d9b66f;
    --yellow-dim: rgba(217,182,111,0.10);
    --yellow-border:rgba(217,182,111,0.26);
    --red:        #dc7379;
    --red-dim:    rgba(220,115,121,0.10);
    --orange:     #d89472;
    --purple:     #a99bd1;

    --text-0:     #f0f6fc;   /* blanc */
    --text-1:     #c9d1d9;   /* gris clair */
    --text-2:     #8b949e;   /* gris moyen */
    --text-3:     #484f58;   /* gris foncé */

    --border:     #2a333b;
    --radius-sm:  6px;
    --radius-md:  10px;
    --radius-lg:  12px;
    --shadow:     0 4px 16px rgba(0,0,0,0.18);
    --shadow-green: none;
  }

  /* ════ Reset ════ */
  html, body, [class*="css"] {
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background-color: var(--bg-1);
    color: var(--text-0);
  }
  .stApp { background-color: var(--bg-1); }
  .block-container { padding-top: 0.7rem !important; }

  /* ════ Header ════ */
  .main-header {
    text-align: left;
    padding: 0.4rem 0 0.65rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.55rem;
  }
  h1.main-title {
    font-size: 1.5rem !important;
    font-weight: 650 !important;
    line-height: 1.2 !important;
    color: var(--text-0);
    margin: 0;
    letter-spacing: -0.5px;
  }
  .main-subtitle {
    color: var(--text-2);
    font-size: 0.82rem;
    margin-top: 0.25rem;
    font-weight: 400;
  }
  .main-header p.main-subtitle { font-size: 0.82rem !important; }

  /* ════ Avertissement ════ */
  .warning-banner {
    color: var(--text-2);
    padding: 0.15rem 0;
    margin-bottom: 0.4rem;
    text-align: left;
    font-size: 0.72rem;
  }

  /* ════ Onglets ════ */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0.1rem;
    background: transparent;
    border-bottom: 1px solid var(--border);
    padding: 0;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 0;
    color: var(--text-2);
    font-weight: 500;
    font-size: 0.78rem;
    padding: 0.5rem 0.65rem;
  }
  .stTabs [aria-selected="true"] {
    color: var(--text-0) !important;
  }
  .stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--green) !important;
  }

  /* ════ CARTE MATCH ════ */
  .match-card {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 0.8rem 0.9rem 0.65rem;
    margin-bottom: 0.45rem;
    position: relative;
    overflow: hidden;
  }
  .match-card.live {
    border-color: rgba(220,115,121,0.42);
  }

  /* Ligne compétition + badge */
  .card-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }
  .card-competition {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-2);
    text-transform: uppercase;
    letter-spacing: 0.8px;
  }

  /* Badge LIVE */
  .live-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--red);
    color: white;
    font-size: 0.68rem;
    font-weight: 800;
    padding: 3px 9px;
    border-radius: 20px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
  }
  .live-dot {
    width: 6px; height: 6px;
    background: white;
    border-radius: 50%;
    animation: none;
  }

  /* Équipes + score */
  .teams-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.8rem;
    margin-bottom: 0.75rem;
  }
  .team-name {
    font-size: 0.98rem;
    font-weight: 650;
    color: var(--text-0);
    flex: 1;
    line-height: 1.2;
  }
  .team-name.home { text-align: left; }
  .team-name.away { text-align: right; }
  .score-box {
    background: var(--bg-4);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.5rem 1rem;
    text-align: center;
    min-width: 80px;
    flex-shrink: 0;
  }
  .score-text {
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--text-0);
    letter-spacing: 2px;
    line-height: 1;
  }
  .score-minute {
    font-size: 0.68rem;
    color: var(--red);
    font-weight: 700;
    margin-top: 2px;
    letter-spacing: 0.5px;
  }

  /* ════ BARRES DE PROBABILITÉS ════ */
  .prob-bar-container { margin: 0.1rem 0 0.6rem; }
  .prob-bar-wrapper {
    display: flex;
    height: 8px;
    border-radius: 4px;
    overflow: hidden;
    gap: 2px;
    background: var(--bg-4);
    padding: 2px;
    border-radius: 6px;
  }
  .prob-segment {
    height: 100%;
    border-radius: 3px;
    transition: none;
  }
  .prob-home { background: #659fc8; }
  .prob-draw { background: #707b86; }
  .prob-away { background: #d89472; }

  /* ════ BANNIÈRE PRONOSTIC (hero) ════ */
  .reco-banner {
    width: 100%;
    border-radius: var(--radius-md);
    padding: 0.5rem 0.65rem;
    margin: 0.1rem 0 0.4rem;
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .reco-title {
    font-size: 0.92rem;
    font-weight: 650;
    line-height: 1.2;
  }
  .reco-sub {
    font-size: 0.75rem;
    font-weight: 500;
    opacity: 0.85;
  }

  /* Niveaux de signal */
  .reco-strong {
    background: var(--green-dim);
    border: 1px solid var(--green-border);
  }
  .reco-strong .reco-title { color: var(--green); }
  .reco-strong .reco-sub   { color: var(--text-1); }

  .reco-medium {
    background: var(--blue-dim);
    border: 1.5px solid var(--blue-border);
  }
  .reco-medium .reco-title { color: var(--blue); }
  .reco-medium .reco-sub   { color: var(--text-1); }

  .reco-low {
    background: var(--yellow-dim);
    border: 1.5px solid var(--yellow-border);
  }
  .reco-low .reco-title { color: var(--yellow); }
  .reco-low .reco-sub   { color: var(--text-1); }

  .reco-skip {
    background: var(--bg-2);
    border: 1px solid var(--border);
  }
  .reco-skip .reco-title { color: var(--text-2); }
  .reco-skip .reco-sub   { color: var(--text-3); }

  /* ════ PILLS Scores probables ════ */
  .score-pills {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 0.3rem;
  }
  .score-pill {
    background: var(--bg-4);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-1);
  }
  .score-pill.top {
    background: var(--green-dim);
    border-color: var(--green-border);
    color: var(--green);
  }

  /* ════ Méta info ════ */
  .match-meta {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-top: 0.5rem;
  }
  .meta-tag {
    background: var(--bg-4);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.68rem;
    color: var(--text-2);
    font-weight: 500;
  }

  /* ════ Stat cards (métriques) ════ */
  .stat-card {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.1rem 1rem;
    text-align: center;
    transition: border-color 0.15s;
  }
  .stat-card:hover { border-color: var(--blue-border); }
  .stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--blue);
    line-height: 1;
  }
  .stat-label {
    font-size: 0.72rem;
    color: var(--text-2);
    margin-top: 0.3rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  /* ════ Empty state ════ */
  .empty-state {
    text-align: center;
    padding: 1.5rem 1rem;
    color: var(--text-2);
  }
  .empty-icon { font-size: 2rem; margin-bottom: 0.4rem; opacity: 0.55; }
  .empty-text { font-size: 0.9rem; line-height: 1.5; }

  /* ════ Value Bets rows ════ */
  .vb-row {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    flex-wrap: wrap;
    transition: all 0.15s ease;
    cursor: default;
  }
  .vb-row:hover { border-color: var(--green-border); background: var(--green-dim); }
  .vb-row.top   {
    border-color: var(--green);
    background: var(--green-dim);
    box-shadow: none;
  }
  .vb-ev    { font-size:1.5rem; font-weight:750; color:var(--green); min-width:64px; text-align:center; }
  .vb-teams { flex:1; min-width:160px; }
  .vb-match { font-size:0.92rem; font-weight:700; color:var(--text-0); margin-bottom:3px; }
  .vb-comp  { font-size:0.7rem; color:var(--text-2); margin-bottom:4px; }
  .vb-pick  { font-size:0.82rem; font-weight:700; color:var(--blue); margin-bottom:2px; }
  .vb-kelly { font-size:0.76rem; color:var(--yellow); }

  /* ════ Calendrier ════ */
  .cal-grid {
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    gap: 6px;
    margin: 0.8rem 0;
  }
  .cal-day-header {
    text-align: center;
    font-size: 0.68rem;
    font-weight: 700;
    color: var(--text-2);
    padding: 4px 0;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .cal-cell {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 7px 5px;
    min-height: 70px;
    font-size: 0.68rem;
    transition: all 0.15s ease;
  }
  .cal-cell.today {
    border-color: var(--purple);
    background: rgba(167,139,250,0.07);
  }
  .cal-cell.has-value {
    border-color: var(--green);
    background: var(--green-dim);
  }
  .cal-date { font-weight: 800; color: var(--text-0); font-size: 0.85rem; margin-bottom:4px; }
  .cal-badge {
    display: block;
    border-radius: 4px;
    padding: 2px 5px;
    margin-bottom: 2px;
    font-size: 0.62rem;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .cal-badge.value  { background: var(--green-dim);  color: var(--green); }
  .cal-badge.normal { background: var(--blue-dim);   color: var(--blue); }
  .cal-badge.low    { background: var(--bg-4);       color: var(--text-2); }

  /* ════ Sidebar ════ */
  section[data-testid="stSidebar"] {
    background: var(--bg-0) !important;
    border-right: 1px solid var(--border) !important;
  }

  @media (max-width: 700px) {
    html, body, .stApp { max-width: 100%; overflow-x: hidden; }
    .block-container {
      max-width: 100vw;
      padding: 2rem 0.65rem calc(1rem + env(safe-area-inset-bottom)) !important;
    }
    .main-header { padding: 0.3rem 0 0.45rem; }
    h1.main-title { font-size: 1.3rem !important; }
    .warning-banner { font-size: 0.68rem; }
    .stTabs [data-baseweb="tab-list"] {
      flex-wrap: nowrap;
      overflow-x: auto;
      scrollbar-width: none;
      scroll-snap-type: x proximity;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
    .stTabs [data-baseweb="tab"] {
      min-height: 44px;
      flex-shrink: 0;
      font-size: 0.75rem;
      padding: 0.5rem 0.6rem;
      scroll-snap-align: start;
    }
    div[data-testid="stHorizontalBlock"] {
      flex-wrap: wrap;
      gap: 0.5rem;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
      flex: 1 1 calc(50% - 0.5rem) !important;
      width: calc(50% - 0.5rem) !important;
      min-width: calc(50% - 0.5rem) !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.match-card) > div[data-testid="column"] {
      flex: 1 1 100% !important;
      width: 100% !important;
      min-width: 100% !important;
    }
    .match-card { padding: 0.8rem; }
    .teams-row { gap: 0.4rem; }
    .team-name {
      min-width: 0;
      font-size: 0.84rem;
      overflow-wrap: anywhere;
    }
    .score-box { min-width: 64px; padding: 0.4rem 0.5rem; }
    .score-text { font-size: 1.2rem; }
    button, [role="button"], input, select, textarea {
      min-height: 44px;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] { overflow-x: auto; }
    [data-testid="stPlotlyChart"] { max-width: 100%; overflow: hidden; }
    .empty-state { padding: 1rem 0.5rem; }
    .empty-icon { font-size: 1.6rem; margin-bottom: 0.25rem; }
  }

  @media (max-width: 380px) {
    .block-container {
      padding-right: 0.5rem !important;
      padding-left: 0.5rem !important;
    }
    .stTabs [data-baseweb="tab"] { padding-right: 0.45rem; padding-left: 0.45rem; }
    .score-box { min-width: 58px; }
  }

  /* ════ Inputs Streamlit ════ */
  .stNumberInput input, .stSlider { accent-color: var(--green) !important; }
  div[data-testid="stMetricValue"] { color: var(--text-0) !important; }

  /* ════ Masquer UI Streamlit ════ */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════════════
# Utilitaires d'affichage
# ═══════════════════════════════════════════════════════════════════════════

def get_recommendation(probs: dict, sport: str, match: dict, odds: dict | None = None) -> dict:
    """
    Génère une recommandation simple et lisible.

    Logique :
    - Si odds dispo   : recommande l'issue avec EV la plus élevée (si EV > 0)
    - Sans odds       : recommande l'issue avec la probabilité la plus haute
    - Seuils confiance : Forte >= 60%, Moyenne 45-60%, Faible 33-45%, Passer < 33%

    Retourne :
        {
          outcome : 'home_win' | 'draw' | 'away_win' | 'skip'
          label   : 'Arsenal'  | 'Nul'  | 'Real Madrid' | 'Passer'
          conf    : 0.68        (probabilité modèle)
          level   : 'strong' | 'medium' | 'low' | 'skip'
          ev      : 0.12 | None
          reason  : str   (phrase courte expl1iquant le choix)
        }
    """
    hw = probs.get("home_win", 0)
    d  = probs.get("draw",     0)
    aw = probs.get("away_win", 0)

    home = match.get("home_team", "Domicile")
    away = match.get("away_team", "Extérieur")

    outcome_labels = {
        "home_win": home,
        "draw":     "Nul",
        "away_win": away,
    }
    probs_map = {"home_win": hw, "draw": d, "away_win": aw}

    # Cas 1 : avec cotes — on cherche la meilleure EV positive
    if odds:
        from models.football import FootballModel
        evs = {}
        for k in ("home_win", "draw", "away_win"):
            p = probs_map[k]
            o = odds.get(k, 0)
            if o > 0:
                evs[k] = round(p * o - 1, 3)
        best_k = max(evs, key=evs.get) if evs else None
        best_ev = evs.get(best_k, 0) if best_k else 0

        if best_k and best_ev > 0:
            conf = probs_map[best_k]
            level = "strong" if best_ev > 0.08 else "medium" if best_ev > 0.03 else "low"
            return {
                "outcome": best_k,
                "label":   outcome_labels[best_k],
                "conf":    conf,
                "level":   level,
                "ev":      best_ev,
                "reason":  f"Cote {odds.get(best_k, 0):.2f} — EV +{round(best_ev*100,1)}%",
            }

    # Cas 2 : sans cotes — plus forte probabilité
    best_k = max(probs_map, key=probs_map.get)
    conf   = probs_map[best_k]

    # Pour le basket, on ignore le nul
    if sport == "basketball" and best_k == "draw":
        probs_map["draw"] = 0
        best_k = max(probs_map, key=probs_map.get)
        conf   = probs_map[best_k]

    if conf < 0.33:
        return {"outcome": "skip", "label": "➖ Passer ce match",
                "conf": conf, "level": "skip", "ev": None,
                "reason": "Trop incertain, éviter"}

    level  = "strong" if conf >= 0.60 else "medium" if conf >= 0.45 else "low"
    reason_map = {
        "strong": "Favori net, probabilité élevée",
        "medium":  "Légère domination modèle",
        "low":    "Marché incertain, mise modérée",
    }
    return {
        "outcome": best_k,
        "label":   outcome_labels[best_k],
        "conf":    conf,
        "level":   level,
        "ev":      None,
        "reason":  reason_map[level],
    }


def render_recommendation(reco: dict):
    """Affiche la bannière hero de recommandation sous la carte match."""
    level   = reco["level"]
    label   = reco["label"]
    conf    = round(reco["conf"] * 100, 1)
    reason  = reco["reason"]
    ev_part = f" · EV <strong>+{round(reco['ev']*100,1)}%</strong>" if reco.get("ev") else ""

    verb = "Passer" if level == "skip" else "Pronostic :"

    st.markdown(f"""
    <div class="reco-banner reco-{level}">
      <div class="reco-title">{verb} {label}</div>
      <div class="reco-sub">{conf}% estimés{ev_part} · {reason}</div>
    </div>
    """, unsafe_allow_html=True)


def render_probability_bar(probs: dict, show_draw: bool = True):
    """Barres de probabilités avec étiquettes colorées distinctes."""
    hw = probs.get("home_win", 0)
    d  = probs.get("draw", 0) if show_draw else 0
    aw = probs.get("away_win", 0)

    total = max(hw + d + aw, 0.001)
    hw_pct = round(hw * 100, 1)
    d_pct  = round(d  * 100, 1)
    aw_pct = round(aw * 100, 1)
    hw_w   = round(hw / total * 100, 1)
    d_w    = round(d  / total * 100, 1)
    aw_w   = round(aw / total * 100, 1)

    labels = [
        f"Dom. **{hw_pct:.1f}%**",
        *([f"Nul **{d_pct:.1f}%**"] if show_draw else []),
        f"Ext. **{aw_pct:.1f}%**",
    ]
    label_columns = st.columns(len(labels))
    for column, label in zip(label_columns, labels):
        column.markdown(label)

    segments = [
        f'<div class="prob-segment prob-home" style="width:{hw_w}%"></div>',
    ]
    if show_draw:
        segments.append(
            f'<div class="prob-segment prob-draw" style="width:{d_w}%"></div>'
        )
    segments.append(
        f'<div class="prob-segment prob-away" style="width:{aw_w}%"></div>'
    )
    bar_html = (
        '<div class="prob-bar-container"><div class="prob-bar-wrapper">'
        f'{"".join(segments)}'
        "</div></div>"
    )
    st.markdown(bar_html, unsafe_allow_html=True)


def render_football_market_forecasts(probs: dict) -> None:
    """Affiche les estimations de buts, BTTS et double chance."""
    markets = probs.get("markets")
    if not isinstance(markets, dict) or not markets:
        return

    st.markdown("**Buts & autres marchés**")
    history_matches = int(probs.get("history_matches", 0))
    if history_matches == 0:
        st.caption(
            "Modèle sans historique : estimations indicatives basées sur les "
            "moyennes par défaut, à prendre avec prudence."
        )
    else:
        st.caption(
            f"Probabilités estimées par le modèle sur {history_matches} matchs ; "
            "elles ne constituent pas une garantie."
        )

    lambda_home = probs.get("lambda_home")
    lambda_away = probs.get("lambda_away")
    if lambda_home is not None and lambda_away is not None:
        st.caption(
            f"Buts attendus : domicile {lambda_home:.2f} · extérieur {lambda_away:.2f}"
        )

    st.caption("Total de buts")
    goal_columns = st.columns(3)
    for column, line in zip(goal_columns, ("1_5", "2_5", "3_5")):
        over = float(markets.get(f"over_{line}", 0))
        under = float(markets.get(f"under_{line}", 0))
        selected_label, selected_probability = (
            (f"Plus de {line.replace('_', '.')}", over)
            if over >= under
            else (f"Moins de {line.replace('_', '.')}", under)
        )
        other_probability = min(over, under)
        with column:
            st.metric(
                selected_label,
                f"{selected_probability:.1%}",
                delta=f"Autre issue : {other_probability:.1%}",
                delta_color="off",
            )

    btts_yes = float(markets.get("btts_yes", 0))
    btts_no = float(markets.get("btts_no", 0))
    btts_label, btts_probability = (
        ("Oui, les deux marquent", btts_yes)
        if btts_yes >= btts_no
        else ("Non, les deux marquent", btts_no)
    )
    dc_columns = st.columns(4)
    with dc_columns[0]:
        st.metric(
            btts_label,
            f"{btts_probability:.1%}",
            delta=f"Autre issue : {min(btts_yes, btts_no):.1%}",
            delta_color="off",
        )
    for column, key, label in zip(
        dc_columns[1:],
        ("dc_1x", "dc_x2", "dc_12"),
        ("Double chance 1X", "Double chance X2", "Double chance 12"),
    ):
        with column:
            st.metric(label, f"{float(markets.get(key, 0)):.1%}")

    st.caption(
        "Ces marchés affichent des probabilités du modèle, pas des cotes ni une "
        "value bet. Les cotes de buts/BTTS ne sont pas intégrées."
    )


def render_match_card(match: dict, probs: dict, sport: str):
    """Affiche une carte de match complète."""
    is_live = match.get("status") in ("IN_PLAY", "PAUSED") or (
        match.get("sport") == "basketball" and match.get("period", 0) > 0
        and match.get("status") not in ("Final", "FINISHED")
    )

    home_score = match.get("home_score")
    away_score = match.get("away_score")

    # Score affichage
    if home_score is not None and away_score is not None:
        score_display = f"{home_score} – {away_score}"
    else:
        score_display = "vs"

    # Minute / période
    if sport == "football":
        minute_display = f"{match.get('minute', '?')}'" if is_live else "À venir"
    else:
        period = match.get("period", 0)
        time_str = match.get("time", "")
        if period > 0:
            minute_display = f"Q{period} {time_str}" if time_str else f"Q{period}"
        else:
            minute_display = "À venir"

    live_badge = '<span class="live-badge"><span class="live-dot"></span>LIVE</span>' if is_live else ""
    card_class = "match-card live" if is_live else "match-card"

    # Cartons rouges
    red_home = "🟥" * match.get("red_cards_home", 0)
    red_away = "🟥" * match.get("red_cards_away", 0)

    # Méta-info
    competition = match.get("competition", "")
    match_date  = match.get("match_date", "")
    model_type  = probs.get("model", "?")
    elo_info    = ""
    if "home_elo" in probs:
        elo_info = f"Elo: {int(probs['home_elo'])} / {int(probs['away_elo'])}"

    st.markdown(f"""
    <div class="{card_class}">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.8rem;">
          <div style="font-size:0.78rem;color:var(--text-secondary);">
            {competition}{f'<br><span style="color:var(--text-3)">{match_date}</span>' if match_date else ''}
        </div>
        {live_badge}
      </div>

      <div class="teams-row">
        <span class="team-name home">{match.get('home_team','?')} {red_home}</span>
        <div class="score-box">
          <div class="score-text">{score_display}</div>
          <div class="score-minute">{minute_display}</div>
        </div>
        <span class="team-name away">{red_away} {match.get('away_team','?')}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if sport == "football":
        weather = WeatherClient().get_match_weather(match)
        if weather:
            weather_parts = [weather["condition"]]
            if weather["temperature"] is not None:
                weather_parts.append(f"{weather['temperature']} °C")
            if weather["precipitation_probability"] is not None:
                weather_parts.append(f"pluie {weather['precipitation_probability']} %")
            if weather["wind_kmh"] is not None:
                weather_parts.append(f"vent {weather['wind_kmh']} km/h")
            st.caption(
                f"Météo prévue · {' · '.join(weather_parts)} · Open-Meteo. "
                "Information indicative, non intégrée aux probabilités du modèle."
            )

    # Barres de probabilités (hors du HTML pour Streamlit)
    show_draw = (sport == "football")
    render_probability_bar(probs, show_draw=show_draw)

    # ── Recommandation pronostic ──
    reco = get_recommendation(probs, sport, match)
    render_recommendation(reco)

    # Les informations secondaires restent disponibles sans alourdir la carte.
    top_scores = probs.get("top_scores", [])
    markets = probs.get("markets")
    if top_scores or markets or model_type or elo_info:
        expander_label = (
            "Buts & détails du modèle"
            if sport == "football"
            else "Détails du match"
        )
        with st.expander(expander_label, expanded=False):
            details = [model_type]
            if elo_info:
                details.append(elo_info)
            if match_date:
                details.append(match_date)
            st.caption(" · ".join(details))

            if sport == "football":
                render_football_market_forecasts(probs)

            if top_scores:
                scores_html = " ".join(
                    f'<span class="meta-tag" style="font-weight:600;">{s["score"]} ({s["prob"]}%)</span>'
                    for s in top_scores[:3]
                )
                st.markdown(
                    f'<div class="match-meta">Scores les plus probables : {scores_html}</div>',
                    unsafe_allow_html=True
                )

    st.markdown("---")


def build_probability_chart(history: list[dict], home_team: str, away_team: str, sport: str):
    """
    Crée un graphique Plotly de l'évolution des probabilités dans le temps.
    history : liste de dicts {minute/time, home_win, draw, away_win}
    """
    if not history:
        return None

    df = pd.DataFrame(history)
    x_col = "minute" if sport == "football" else "time_elapsed"

    if x_col not in df.columns:
        return None

    fig = go.Figure()

    # Probabilité home win
    fig.add_trace(go.Scatter(
        x=df[x_col], y=(df["home_win"] * 100).round(1),
        mode="lines+markers",
        name=f"🏠 {home_team}",
        line=dict(color="#58a6ff", width=2.5),
        marker=dict(size=4),
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.08)",
        hovertemplate=f"{home_team}: %{{y:.1f}}%<extra></extra>",
    ))

    # Probabilité draw (football uniquement)
    if sport == "football" and "draw" in df.columns:
        fig.add_trace(go.Scatter(
            x=df[x_col], y=(df["draw"] * 100).round(1),
            mode="lines",
            name="🤝 Nul",
            line=dict(color="#e3b341", width=2, dash="dot"),
            hovertemplate="Nul: %{y:.1f}%<extra></extra>",
        ))

    # Probabilité away win
    fig.add_trace(go.Scatter(
        x=df[x_col], y=(df["away_win"] * 100).round(1),
        mode="lines+markers",
        name=f"✈️ {away_team}",
        line=dict(color="#f78166", width=2.5),
        marker=dict(size=4),
        fill="tozeroy",
        fillcolor="rgba(247,129,102,0.08)",
        hovertemplate=f"{away_team}: %{{y:.1f}}%<extra></extra>",
    ))

    # Ligne 50%
    fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.15)", line_width=1)

    # Style du graphique
    xlabel = "Minute" if sport == "football" else "Temps écoulé (min)"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,27,34,0.8)",
        font=dict(family="Inter", color="#8b949e", size=12),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f0f6fc"),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        height=280,
        xaxis=dict(
            title=xlabel, gridcolor="rgba(48,54,61,0.5)",
            showgrid=True, zeroline=False,
            title_font=dict(color="#8b949e"),
        ),
        yaxis=dict(
            title="Probabilité (%)", gridcolor="rgba(48,54,61,0.5)",
            showgrid=True, zeroline=False, range=[0, 100],
            title_font=dict(color="#8b949e"),
        ),
        hoverlabel=dict(bgcolor="var(--bg-card)", font_color="#f0f6fc"),
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Logique principale
# ═══════════════════════════════════════════════════════════════════════════

# Les clients API ne sont PAS mis en cache (pour éviter de figer les méthodes).
# Les modèles Elo SONT mis en cache car leur entraînement est coûteux.
TRAINING_HISTORY_LIMIT = 5000
FOOTBALL_PREMATCH_STATUSES = {"SCHEDULED", "TIMED"}
MIN_VALUE_BET_HISTORY = 200
MIN_RECOMMENDED_EV = 0.08
MIN_VALUE_PROBABILITY = 0.65
MAX_VALUE_ODDS = 1.80
MAX_SINGLE_STAKE_FRACTION = 0.005
MAX_PORTFOLIO_STAKE_FRACTION = 0.01

def get_football_client() -> FootballClient:
    return FootballClient()


def get_basketball_client() -> BasketballClient:
    return BasketballClient()


def get_odds_client() -> OddsClient:
    return OddsClient()


def football_history_count(model: FootballModel) -> int:
    history = getattr(model, "history", None)
    if not isinstance(history, list):
        logger.warning(
            "Historique du modèle football absent ou invalide ; compté comme vide."
        )
        return 0
    return len(history)


def conservative_match_ev(model: FootballModel, probs: dict, odds_event: dict) -> dict | None:
    if football_history_count(model) < MIN_VALUE_BET_HISTORY:
        return None
    conservative_probs = model.conservative_probabilities(
        probs, odds_event.get("avg_odds", {})
    )
    if conservative_probs is None:
        return None
    return {
        **model.compute_ev(conservative_probs, odds_event.get("best_odds", {})),
        "probabilities": conservative_probs,
    }


def is_value_bet_eligible(probability: float, odds: float, ev: float, min_ev: float) -> bool:
    return (
        probability >= MIN_VALUE_PROBABILITY
        and 1 < odds <= MAX_VALUE_ODDS
        and ev >= min_ev
    )


def _training_history_json(sport: str) -> str:
    history = get_history_from_db(sport, limit=TRAINING_HISTORY_LIMIT)
    training_rows = [
        {
            "home_team": match["home_team"],
            "away_team": match["away_team"],
            "home_score": match["home_score"],
            "away_score": match["away_score"],
            "match_date": match["match_date"],
            "competition": match.get("competition"),
        }
        for match in history
    ]
    return json.dumps(training_rows, ensure_ascii=False, sort_keys=True)


@st.cache_resource(show_spinner=False, max_entries=2)
def _train_football_model(history_json: str) -> FootballModel:
    """Instancie et entraîne le modèle football sur l'historique en DB."""
    model = FootballModel()
    history = json.loads(history_json)
    if history:
        model.train(history, tune=True, calibrate=True)
    return model


def _sync_football_history() -> None:
    client = get_football_client()
    sync_history = getattr(client, "sync_competition_history", None)
    if not callable(sync_history):
        logger.warning(
            "Synchronisation de l'historique indisponible dans cette version "
            "de FootballClient."
        )
        return
    sync_history()


def get_football_model() -> FootballModel:
    history_json = _training_history_json("football")
    if len(json.loads(history_json)) < MIN_VALUE_BET_HISTORY:
        _sync_football_history()
        history_json = _training_history_json("football")
    return _train_football_model(history_json)


@st.cache_resource(show_spinner=False, max_entries=2)
def _train_basketball_model(history_json: str) -> BasketballModel:
    """Instancie et entraîne le modèle basketball sur l'historique en DB."""
    model = BasketballModel()
    history = json.loads(history_json)
    if history:
        model.train(history, calibrate=True)
    return model


def get_basketball_model() -> BasketballModel:
    return _train_basketball_model(_training_history_json("basketball"))


def render_sidebar():
    """Panneau latéral : statut des APIs et contrôles."""
    with st.sidebar:
        st.markdown("## ⚙️ Paramètres")

        # Sélection de la devise
        st.markdown("### 💱 Devise & Bookmaker")
        selected_currency = st.selectbox(
            "Devise pour les mises",
            options=["FCFA", "EUR (€)", "USD ($)"],
            index=0,  # FCFA par défaut pour Congobet
            key="app_currency",
            help="Adapte la bankroll, les paliers et les calculs de mises Kelly (ex: FCFA pour Congobet)."
        )
        if selected_currency == "FCFA":
            st.caption("🇨🇬 Adapté pour **Congobet** (mises en FCFA entiers)")

        st.markdown("---")

        # Compétitions suivies
        st.markdown("### 🏆 Compétitions Football")
        st.markdown(
            "- 🏴󠁧󠁢󠁥󠁮󠁧󠁿 **Premier League** (PL)\n"
            "- 🇪🇸 **La Liga** (PD)\n"
            "- 🇫🇷 **Ligue 1** (FL1)\n"
            "- 🇩🇪 **Bundesliga** (BL1)\n"
            "- 🇮🇹 **Serie A** (SA)\n"
            "- ⭐ **Champions League** (CL)\n"
            "- 🏆 **Coupe du Monde** (WC)\n"
            "- 🌍 **CAN** (CAF)"
        )

        st.markdown("---")

        # Statut des clés API
        import os
        football_key = bool(os.getenv("FOOTBALL_DATA_API_KEY"))
        odds_key     = bool(os.getenv("ODDS_API_KEY"))
        bball_key    = os.getenv("BALLDONTLIE_API_KEY", "")

        st.markdown("### 🔑 Statut des APIs")
        st.markdown(
            f"{'✅' if football_key else '❌'} **football-data.org** "
            f"{'(clé configurée)' if football_key else '(clé manquante)'}"
        )
        st.markdown(
            f"{'✅' if odds_key else '⚠️'} **The Odds API** "
            f"{'(cotes actives)' if odds_key else '(optionnelle — EV désactivé)'}"
        )
        st.markdown(
            f"{'✅' if bball_key else '⚠️'} **balldontlie** "
            f"{'(clé configurée)' if bball_key else '(clé optionnelle)'}"
        )

        # Statut Telegram Bot
        tg_ready = is_telegram_configured()
        st.markdown(
            f"{'✅' if tg_ready else '⚠️'} **Telegram Bot** "
            f"{'(alertes actives)' if tg_ready else '(optionnel - alertes désactivées)'}"
        )

        st.markdown("---")

        # Bouton vider le cache modèles
        if st.button("🗑️ Vider le cache modèles", use_container_width=True):
            st.cache_resource.clear()
            st.success("Cache vidé — modèles rechargés au prochain rafraîchissement.")

        st.markdown("---")
        st.markdown(
            "<small style='color:#8b949e'>Modèles : Elo + Dixon-Coles (foot) "
            "· Elo + Loi normale (basket)<br>"
            "Sources : football-data.org · balldontlie · The Odds API</small>",
            unsafe_allow_html=True,
        )


def render_football_tab():
    """Affiche le contenu de l'onglet Football."""
    client = get_football_client()
    model  = get_football_model()

    with st.spinner("⚽ Chargement des matchs de football..."):
        matches = client.get_today_matches()

    st.caption(
        "Les blessures, suspensions et compositions ne sont pas couvertes par les "
        "sources actuellement configurées. La météo, lorsqu'elle est disponible, "
        "est informative et n'ajuste pas encore les probabilités."
    )

    if not matches:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-text">Aucun match de football aujourd'hui<br>
          <small style="color:#8b949e">Vérifiez votre clé API football-data.org dans le fichier .env</small>
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # L'API peut classer un match à venir comme SCHEDULED ou TIMED.
    status_order = {"IN_PLAY": 0, "PAUSED": 1, "TIMED": 2, "SCHEDULED": 2, "FINISHED": 3}
    matches = sorted(matches, key=lambda m: status_order.get(m.get("status", ""), 99))

    # Filtres
    col_filter1, col_filter2 = st.columns([1, 2])
    with col_filter1:
        show_finished = st.checkbox("Afficher les matchs terminés", value=False, key="foot_finished")
    with col_filter2:
        available_comps = sorted(list({m.get("competition", "") for m in matches if m.get("competition")}))
        selected_comp = st.selectbox(
            "Compétition",
            options=["Toutes les compétitions"] + available_comps,
            index=0,
            key="foot_comp_filter"
        )

    if selected_comp != "Toutes les compétitions":
        matches = [m for m in matches if m.get("competition") == selected_comp]

    live_matches = [m for m in matches if m.get("status") in ("IN_PLAY", "PAUSED")]
    scheduled    = [m for m in matches if m.get("status") in FOOTBALL_PREMATCH_STATUSES]
    finished     = [m for m in matches if m.get("status") == "FINISHED"]

    # Métriques rapides
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("En direct", len(live_matches))
    c2.metric("Programmés", len(scheduled))
    c3.metric("Terminés", len(finished))
    c4.metric("Total", len(matches))

    st.markdown("<br>", unsafe_allow_html=True)

    # Affichage des matchs
    display_matches = live_matches + scheduled
    if show_finished:
        display_matches += finished

    if not display_matches:
        st.info("Aucun match à afficher avec les filtres actuels.")
        return

    # Deux colonnes pour les cartes
    cols = st.columns(2, gap="medium")

    for i, match in enumerate(display_matches):
        with cols[i % 2]:
            probs = model.predict(match)

            # Sauvegarde des prédictions pour analyse
            if match.get("status") in FOOTBALL_PREMATCH_STATUSES:
                save_prediction(str(match["id"]), "football", None, probs)

            render_match_card(match, probs, sport="football")

            # Graphique d'évolution (uniquement pour les matchs en cours)
            if match.get("status") in ("IN_PLAY", "PAUSED"):
                history = st.session_state.get(f"history_foot_{match['id']}", [])
                history.append({
                    "minute": match.get("minute", 0),
                    "home_win": probs.get("home_win", 0),
                    "draw": probs.get("draw", 0),
                    "away_win": probs.get("away_win", 0),
                })
                st.session_state[f"history_foot_{match['id']}"] = history[-30:]

                if len(history) > 1:
                    fig = build_probability_chart(
                        history,
                        match.get("home_team", ""),
                        match.get("away_team", ""),
                        sport="football",
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_foot_{match['id']}")


def render_basketball_tab():
    """Affiche le contenu de l'onglet Basketball."""
    client = get_basketball_client()
    model  = get_basketball_model()

    with st.spinner("🏀 Chargement des matchs NBA..."):
        matches = client.get_today_matches()

    if not matches:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-text">Aucun match NBA aujourd'hui<br>
          <small style="color:#8b949e">L'API balldontlie ne nécessite pas de clé API (plan gratuit).<br>
          Vérifiez votre connexion internet.</small>
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Trier les matchs
    def bask_order(m):
        s = m.get("status", "")
        if s not in ("Final", ""):
            return 0   # En cours
        if s == "":
            return 1   # À venir
        return 2       # Terminé

    matches = sorted(matches, key=bask_order)

    live_matches = [m for m in matches if m.get("status") not in ("Final", "", None) and m.get("period", 0) > 0]
    scheduled    = [m for m in matches if not m.get("period") and m.get("status") not in ("Final",)]
    finished     = [m for m in matches if m.get("status") == "Final"]

    # Filtre
    show_finished = st.checkbox("Afficher les matchs terminés", value=False, key="bask_finished")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("En direct", len(live_matches))
    c2.metric("Programmés", len(scheduled))
    c3.metric("Terminés", len(finished))
    c4.metric("Total", len(matches))

    st.markdown("<br>", unsafe_allow_html=True)

    display_matches = live_matches + scheduled
    if show_finished:
        display_matches += finished

    if not display_matches:
        st.info("Aucun match à afficher.")
        return

    cols = st.columns(2, gap="medium")

    for i, match in enumerate(display_matches):
        with cols[i % 2]:
            probs = model.predict(match)

            if not match.get("period") and match.get("status") != "Final":
                save_prediction(str(match["id"]), "basketball", None, probs)

            render_match_card(match, probs, sport="basketball")

            # Graphique d'évolution pour les matchs en cours
            if match.get("period", 0) > 0 and match.get("status") != "Final":
                time_elapsed = TOTAL_GAME_MINUTES - match.get("time_remaining_secs", 0) / 60
                history = st.session_state.get(f"history_bask_{match['id']}", [])
                history.append({
                    "time_elapsed": round(time_elapsed, 1),
                    "home_win": probs.get("home_win", 0),
                    "draw": 0,
                    "away_win": probs.get("away_win", 0),
                })
                st.session_state[f"history_bask_{match['id']}"] = history[-30:]

                if len(history) > 1:
                    fig = build_probability_chart(
                        history,
                        match.get("home_team", ""),
                        match.get("away_team", ""),
                        sport="basketball",
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_bask_{match['id']}")


TOTAL_GAME_MINUTES = 48  # Durée NBA


def format_currency(amount: float | int, currency: str = "FCFA") -> str:
    """Formate un montant selon la devise choisie (FCFA, EUR, USD)."""
    if currency == "FCFA":
        val = int(round(amount))
        return f"{val:,} FCFA".replace(",", " ")
    elif currency == "USD ($)":
        return f"${amount:,.2f}"
    else:  # EUR (€)
        return f"{amount:,.2f} €".replace(",", " ")


def kelly_fraction(prob: float, odds: float, fraction: float = 0.25) -> float:
    """
    Calcule la fraction de bankroll à miser selon le critère de Kelly.
    Utilise le quart-Kelly et plafonne la mise à 0,5 % de la bankroll.

    Formule : f = (p × o - 1) / (o - 1)
    Quart-Kelly : f_q = f × fraction

    Args:
        prob   : probabilité modèle (0-1)
        odds   : cote décimale bookmaker
        fraction: réducteur de risque (0.25 = quart-Kelly recommandé)

    Returns:
        Fraction de bankroll en % (0-0,5), ou 0 si mise déconseillée.
    """
    if odds <= 1 or prob <= 0:
        return 0.0
    ev = prob * odds - 1
    if ev <= 0:
        return 0.0
    kelly = ev / (odds - 1)
    return round(min(kelly * fraction * 100, MAX_SINGLE_STAKE_FRACTION * 100), 2)


def cap_portfolio_stakes(value_bets: list[dict], bankroll: float, currency: str) -> None:
    portfolio_cap = bankroll * MAX_PORTFOLIO_STAKE_FRACTION
    proposed_total = sum(v["stake"] for v in value_bets)
    if proposed_total <= portfolio_cap or proposed_total <= 0:
        return
    scale = portfolio_cap / proposed_total
    for value_bet in value_bets:
        scaled_stake = value_bet["stake"] * scale
        value_bet["stake"] = (
            int(scaled_stake)
            if currency == "FCFA"
            else int(scaled_stake * 100) / 100
        )
        value_bet["kelly_pct"] = round(value_bet["stake"] / bankroll * 100, 2)


def render_value_bets_tab():
    """
    Onglet Value Bets : liste uniquement les matchs à EV positif,
    triés par valeur décroissante, avec mise Kelly conseillée.
    """
    client  = get_football_client()
    model   = get_football_model()
    odds_cl = get_odds_client()
    history_count = football_history_count(model)
    if history_count < MIN_VALUE_BET_HISTORY:
        st.info(
            f"Les EV prudentes sont masquées : historique disponible "
            f"{history_count}/{MIN_VALUE_BET_HISTORY} matchs."
        )

    currency = st.session_state.get("app_currency", "FCFA")

    if not odds_cl._is_available():
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">💰</div>
          <div class="empty-text">Clé API Odds manquante<br>
          <small style="color:#8b949e">Ajoutez <code>ODDS_API_KEY</code> dans .env<br>
          Inscription gratuite sur <strong>the-odds-api.com</strong> (500 req/mois)</small>
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Contrôles de bankroll & EV minimale
    c_ctrl1, c_ctrl2 = st.columns([1, 1])
    with c_ctrl1:
        if currency == "FCFA":
            bankroll = st.number_input(
                "💳 Bankroll disponible (FCFA)",
                min_value=500, max_value=50000000, value=50000, step=5000,
                help="Montant total disponible sur votre compte Congobet en FCFA"
            )
        else:
            bankroll = st.number_input(
                f"💳 Bankroll disponible ({currency})",
                min_value=10, max_value=200000, value=500, step=50,
                help="Montant total disponible pour calculer les mises Kelly"
            )

    with c_ctrl2:
        min_ev = st.slider(
            "🎯 EV prudente minimale (%)", min_value=5, max_value=20,
            value=int(MIN_RECOMMENDED_EV * 100),
            help="Seuil appliqué après rapprochement des probabilités du modèle et du marché"
        ) / 100

    # ── Calculateur rapide de cote Congobet ────────────────────────────────
    with st.expander("🧮 Calculateur de Value Bet Congobet (manuel)", expanded=False):
        st.markdown(
            "<small style='color:#8b949e'>Estimation théorique à partir de la probabilité que vous saisissez. "
            "Cette probabilité n'est pas vérifiée par le modèle.</small>",
            unsafe_allow_html=True
        )
        cc1, cc2, cc3 = st.columns([1, 1, 1.2])
        with cc1:
            p_manual = st.number_input(
                "Probabilité estimée (%)", min_value=1.0, max_value=99.0, value=55.0, step=1.0,
                key="calc_manual_prob"
            ) / 100
        with cc2:
            cote_manual = st.number_input(
                "Cote sur Congobet", min_value=1.01, max_value=100.0, value=2.10, step=0.05,
                key="calc_manual_odd"
            )
        with cc3:
            ev_manual = (p_manual * cote_manual) - 1
            k_pct_manual = kelly_fraction(p_manual, cote_manual)
            raw_k = bankroll * k_pct_manual / 100
            if currency == "FCFA":
                m_stake = int(raw_k // 50 * 50)
            else:
                m_stake = round(raw_k, 2)
            stake_display = format_currency(m_stake, currency)

            if ev_manual > 0:
                color_ev = "#39d353"
                advice = f"Probabilité saisie : EV positive. Mise théorique plafonnée à <strong style='color:#e3b341;'>{stake_display}</strong> ({k_pct_manual}% bankroll)."
            else:
                color_ev = "#ff4757"
                advice = "❌ <strong>Pas de value</strong> (EV négative). Éviter de miser."

            st.markdown(f"""
            <div style="background:#161b22;padding:0.7rem 0.9rem;border-radius:8px;border:1px solid #30363d;margin-top:0.3rem;">
              <div style="font-size:0.8rem;color:#8b949e">Valeur espérée (EV) : <strong style="color:{color_ev};">{'+' if ev_manual > 0 else ''}{ev_manual*100:.1f}%</strong></div>
              <div style="font-size:0.82rem;color:#c9d1d9;margin-top:0.3rem;">{advice}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if history_count < MIN_VALUE_BET_HISTORY:
        st.warning(
            f"Mode prudent : les value bets automatiques sont désactivés tant que le modèle "
            f"n'a pas au moins {MIN_VALUE_BET_HISTORY} matchs historiques. "
            f"Historique disponible : {history_count}. Importez des résultats puis "
            "évaluez le modèle en backtest avant de vous fier à ses estimations."
        )
        return

    with st.spinner("📊 Analyse des cotes et calcul des EV..."):
        week_matches = client.get_week_matches(days=7)
        odds_by_comp = odds_cl.get_football_odds()
        all_odds_flat = [ev for evs in odds_by_comp.values() for ev in evs]

    value_bets = []
    for match in week_matches:
        if match.get("status") == "FINISHED":
            continue
        probs = model.predict(match)
        matched_odds = odds_cl.match_odds_to_match(match, all_odds_flat)
        if not matched_odds:
            continue
        best_odds = matched_odds.get("best_odds", {})
        ev_dict = conservative_match_ev(model, probs, matched_odds)
        if ev_dict is None:
            continue

        match_candidates = []
        for outcome in ("home_win", "draw", "away_win"):
            ev_val = ev_dict.get(outcome)
            odd_val = best_odds.get(outcome, 0)
            prob_val = ev_dict["probabilities"].get(outcome, 0)
            if ev_val is not None and is_value_bet_eligible(
                prob_val, odd_val, ev_val, min_ev
            ):
                kelly_pct = kelly_fraction(prob_val, odd_val)
                raw_stake = bankroll * kelly_pct / 100
                if currency == "FCFA":
                    stake_val = int(raw_stake // 50 * 50)
                else:
                    stake_val = int(raw_stake * 100) / 100

                outcome_label = {
                    "home_win": f"🏠 {match.get('home_team','?')}",
                    "draw":     "🤝 Nul",
                    "away_win": f"✈️ {match.get('away_team','?')}",
                }[outcome]
                match_candidates.append({
                    "match":        f"{match.get('home_team','?')} vs {match.get('away_team','?')}",
                    "competition":  match.get("competition", ""),
                    "date":         match.get("match_date", ""),
                    "outcome":      outcome,
                    "pick":         outcome_label,
                    "ev":           ev_val,
                    "odd":          odd_val,
                    "prob":         prob_val,
                    "model_prob":   probs.get(outcome, 0),
                    "kelly_pct":    kelly_pct,
                    "stake":        stake_val,
                })
        if match_candidates:
            value_bets.append(max(match_candidates, key=lambda item: item["ev"]))

    # Tri par EV décroissante
    value_bets.sort(key=lambda x: x["ev"], reverse=True)
    cap_portfolio_stakes(value_bets, bankroll, currency)
    value_bets = [value_bet for value_bet in value_bets if value_bet["stake"] > 0]

    # KPIs
    n_vb = len(value_bets)
    avg_ev = round(sum(v["ev"] for v in value_bets) / n_vb * 100, 1) if value_bets else 0
    total_kelly = sum(v["stake"] for v in value_bets)
    total_kelly_str = format_currency(total_kelly, currency)
    bankroll_str = format_currency(bankroll, currency)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""
    <div class="stat-card">
      <div class="stat-value" style="color:#39d353">{n_vb}</div>
      <div class="stat-label">Value bets détectés</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""
    <div class="stat-card">
      <div class="stat-value" style="color:#58a6ff">+{avg_ev}%</div>
      <div class="stat-label">EV moyenne</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""
    <div class="stat-card">
      <div class="stat-value" style="color:#e3b341">{total_kelly_str}</div>
      <div class="stat-label">Total mises Kelly ({currency})</div>
    </div>""", unsafe_allow_html=True)

    # ── Bouton d'alerte Telegram ──────────────────────────────────────────
    c_tg1, c_tg2 = st.columns([1, 1])
    with c_tg1:
        if st.button("📲 Envoyer le Top 5 Value Bets sur Telegram", use_container_width=True, key="btn_send_tg_top5"):
            if not is_telegram_configured():
                st.warning("⚠️ Pour recevoir les alertes sur smartphone, ajoutez `TELEGRAM_BOT_TOKEN` et `TELEGRAM_CHAT_ID` dans votre fichier `.env`.")
            else:
                ok, msg = send_value_bets_alert(value_bets, bankroll=bankroll, currency=currency)
                if ok:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        "Probabilité prudente = 25 % estimation du modèle + 75 % consensus des cotes "
        "moyennes sans marge. Filtre strict : probabilité ≥ 65 %, cote ≤ 1,80, "
        "EV minimale par défaut : 8 %. "
        "Cotes The Odds API à vérifier chez votre bookmaker ; aucun pari n'est garanti."
    )

    if not value_bets:
        st.info(
            f"Aucun signal ne passe tous les critères : probabilité prudente ≥ "
            f"{MIN_VALUE_PROBABILITY:.0%}, cote ≤ {MAX_VALUE_ODDS:.2f} et EV ≥ {min_ev:.0%}."
        )
        return

    st.markdown(f"### 🔥 {n_vb} Value Bet{'s' if n_vb > 1 else ''} détecté{'' if n_vb == 1 else 's'}")

    for i, vb in enumerate(value_bets):
        is_top = i == 0
        ev_pct  = round(vb["ev"] * 100, 1)
        prob_pct = round(vb["prob"] * 100, 1)
        stake_str = format_currency(vb["stake"], currency)
        kelly_str = (
            f"Mise plafonnée : <strong style='color:#e3b341'>"
            f"{vb['kelly_pct']}% → {stake_str}</strong> "
            f"(quart-Kelly, max. 0,5 % par pari ; 1 % au total)"
            if vb["kelly_pct"] > 0 else "Mise non conseillée"
        )
        col_vb_card, col_vb_btn = st.columns([5, 1])
        with col_vb_card:
            st.markdown(f"""
            <div class="vb-row {'top' if is_top else ''}">
              <div class="vb-ev">+{ev_pct}%</div>
              <div class="vb-teams">
                <div class="vb-match">{vb['match']}</div>
                <div class="vb-comp">{vb['competition']} · {vb['date']}</div>
                <div class="vb-pick">🎯 Miser sur : {vb['pick']} @ {vb['odd']:.2f}</div>
                <div class="vb-kelly">{kelly_str}</div>
              </div>
              <div style="text-align:right;min-width:80px;">
                <div style="color:#8b949e;font-size:0.72rem;">Prob. prudente</div>
                <div style="font-size:1.1rem;font-weight:700;color:#f0f6fc;">{prob_pct}%</div>
                <div style="color:#8b949e;font-size:0.68rem;">Modèle : {vb['model_prob']*100:.1f}%</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        with col_vb_btn:
            if st.button("💾 Suivre", key=f"add_vb_btn_{i}", help="Ajouter ce pari à votre journal de bankroll"):
                add_bet(
                    match_name=vb["match"],
                    competition=vb.get("competition", "Football"),
                    market="1X2",
                    pick=f"{vb['pick']} @ {vb['odd']:.2f}",
                    odds=vb["odd"],
                    stake_fcfa=vb["stake"],
                )
                st.toast(f"Pari {vb['match']} ajouté à votre journal !", icon="✅")

    st.markdown("---")
    st.markdown(
        "<small style='color:#8b949e'>⚠️ Le critère de Kelly est théorique. "
        "Utilisez le quart-Kelly pour réduire le risque de ruine. "
        "Ces calculs sont indicatifs et ne constituent pas un conseil en paris.</small>",
        unsafe_allow_html=True,
    )


def render_bets_journal_tab():
    """
    Onglet '💼 Mes Paris' (Bankroll Management) :
    - Statistiques de rentabilité en FCFA (Bénéfice net, ROI %, Taux de réussite)
    - Enregistrement de paris (Simple / Combiné / Congobet)
    - Suivi des tickets en cours (Gagné / Perdu / Annuler)
    - Historique des paris archivés
    """
    currency = st.session_state.get("app_currency", "FCFA")
    stats = get_bankroll_stats()

    st.markdown("### 💼 Journal de Paris & Gestion de Bankroll")
    st.markdown(
        "<small style='color:#8b949e'>Suivez vos vrais paris placés sur <strong>Congobet</strong>, "
        "mesurez votre ROI réel et sécurisez votre capital en FCFA.</small>",
        unsafe_allow_html=True
    )

    # ── KPIs de Bankroll ──
    profit = stats["net_profit"]
    profit_color = "#39d353" if profit > 0 else "#ff4757" if profit < 0 else "#8b949e"
    profit_sign = "+" if profit > 0 else ""
    profit_str = f"{profit_sign}{format_currency(profit, currency)}"
    pending_str = format_currency(stats["pending_staked"], currency)

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"""
    <div class="stat-card">
      <div class="stat-value" style="color:{profit_color}">{profit_str}</div>
      <div class="stat-label">Bénéfice Net ({currency})</div>
    </div>""", unsafe_allow_html=True)

    roi_color = "#39d353" if stats["roi"] > 0 else "#ff4757" if stats["roi"] < 0 else "#8b949e"
    k2.markdown(f"""
    <div class="stat-card">
      <div class="stat-value" style="color:{roi_color}">{stats['roi']:+.1f}%</div>
      <div class="stat-label">ROI Réel</div>
    </div>""", unsafe_allow_html=True)

    total_settled = stats["won_count"] + stats["lost_count"]
    k3.markdown(f"""
    <div class="stat-card">
      <div class="stat-value" style="color:#58a6ff">{stats['win_rate']:.1f}%</div>
      <div class="stat-label">Taux Réussite ({stats['won_count']}/{total_settled})</div>
    </div>""", unsafe_allow_html=True)

    k4.markdown(f"""
    <div class="stat-card">
      <div class="stat-value" style="color:#e3b341">{stats['pending_count']}</div>
      <div class="stat-label">En cours ({pending_str})</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Formulaire d'enregistrement ──
    with st.expander("➕ Enregistrer un nouveau pari (placé sur Congobet)", expanded=False):
        with st.form("form_add_bet", clear_on_submit=True):
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                b_match = st.text_input("Match", placeholder="ex: Real Madrid vs Barcelone")
                b_comp = st.selectbox(
                    "Compétition",
                    ["Premier League", "La Liga", "Ligue 1", "Bundesliga", "Serie A", "Champions League", "Coupe du Monde", "CAN", "NBA", "Autre"]
                )
                b_market = st.selectbox(
                    "Marché",
                    ["1X2 (Résultat)", "Over/Under 2.5 buts", "Les deux marquent (BTTS)", "Double Chance", "Score exact", "Combiné", "Autre"]
                )
            with f_col2:
                b_pick = st.text_input("Sélection / Pronostic", placeholder="ex: Victoire Real Madrid @ 2.10")
                b_odd = st.number_input("Cote décimale", min_value=1.01, max_value=200.0, value=1.90, step=0.05)
                default_stake = 5000 if currency == "FCFA" else 20.0
                b_stake = st.number_input(f"Mise ({currency})", min_value=100.0 if currency == "FCFA" else 1.0, value=float(default_stake), step=500.0 if currency == "FCFA" else 5.0)

            submitted = st.form_submit_button("💾 Enregistrer le pari", use_container_width=True)
            if submitted:
                if not b_match or not b_pick:
                    st.error("Veuillez renseigner le nom du match et votre sélection.")
                else:
                    new_id = add_bet(
                        match_name=b_match.strip(),
                        competition=b_comp,
                        market=b_market,
                        pick=b_pick.strip(),
                        odds=b_odd,
                        stake_fcfa=b_stake,
                    )
                    st.success(f"✅ Pari #{new_id} enregistré avec succès !")
                    st.rerun()

    # ── Paris en cours ──
    st.markdown("#### ⏳ Paris en cours (Tickets en attente)")
    pending_bets = get_bets(status="PENDING")

    if not pending_bets:
        st.info("Aucun pari en cours pour le moment. Enregistrez vos paris ci-dessus pour suivre vos résultats.")
    else:
        for b in pending_bets:
            pot_payout = round(b["stake_fcfa"] * b["odds"], 2)
            p_stk = format_currency(b["stake_fcfa"], currency)
            p_pot = format_currency(pot_payout, currency)

            c_info, c_btn1, c_btn2, c_btn3 = st.columns([3, 1, 1, 0.8])
            with c_info:
                st.markdown(f"""
                <div style="background:#161b22;padding:0.6rem 0.9rem;border-radius:8px;border-left:3px solid #e3b341;margin-bottom:0.4rem;">
                  <strong>{b['match_name']}</strong> <small style="color:#8b949e">({b['competition']} · {b['market']})</small><br>
                  <span style="color:#58a6ff;">🎯 {b['pick']} @ {b['odds']:.2f}</span> &nbsp;|&nbsp;
                  Mise: <strong>{p_stk}</strong> &nbsp;→&nbsp; Gain potentiel: <strong style="color:#39d353">{p_pot}</strong>
                </div>
                """, unsafe_allow_html=True)
            with c_btn1:
                if st.button("✅ Gagné", key=f"won_{b['id']}", use_container_width=True):
                    settle_bet(b["id"], "WON")
                    st.rerun()
            with c_btn2:
                if st.button("❌ Perdu", key=f"lost_{b['id']}", use_container_width=True):
                    settle_bet(b["id"], "LOST")
                    st.rerun()
            with c_btn3:
                if st.button("🗑️", key=f"del_{b['id']}", use_container_width=True, help="Supprimer ce pari"):
                    delete_bet(b["id"])
                    st.rerun()

    st.markdown("---")

    # ── Historique des paris réglés ──
    st.markdown("#### 📜 Historique des paris terminés")
    all_bets = get_bets()
    settled = [b for b in all_bets if b["status"] != "PENDING"]

    if not settled:
        st.caption("Aucun pari archivé pour l'instant.")
        return

    hist_data = []
    for b in settled:
        net = round(b["payout_fcfa"] - b["stake_fcfa"], 2)
        badge = "✅ GAGNÉ" if b["status"] == "WON" else "❌ PERDU" if b["status"] == "LOST" else "⚪ NUL"
        hist_data.append({
            "ID": f"#{b['id']}",
            "Date": b["created_at"][:10] if b.get("created_at") else "",
            "Match": b["match_name"],
            "Marché": b["market"],
            "Sélection": b["pick"],
            "Cote": f"{b['odds']:.2f}",
            "Mise": format_currency(b["stake_fcfa"], currency),
            "Résultat": badge,
            "Gain": format_currency(b["payout_fcfa"], currency),
            "Bénéfice": f"{net:+,.0f} {currency}".replace(",", " ") if currency == "FCFA" else f"{net:+.2f} €",
        })

    st.dataframe(pd.DataFrame(hist_data), use_container_width=True, hide_index=True)


def render_calendar_tab():
    """
    Onglet Calendrier : vue hebdomadaire avec cases cliquables.
    Code couleur : vert = value bet, bleu = match normal, gris = peu de matchs.
    La case sélectionnée affiche les matchs du jour en dessous.
    """
    from datetime import date, timedelta
    from collections import defaultdict

    client  = get_football_client()
    model   = get_football_model()
    odds_cl = get_odds_client()
    odds_available = odds_cl._is_available()
    history_count = football_history_count(model)
    if history_count < MIN_VALUE_BET_HISTORY:
        st.info(
            f"Les EV du calendrier sont masquées : historique disponible "
            f"{history_count}/{MIN_VALUE_BET_HISTORY} matchs."
        )

    # Navigation semaine
    today = date.today()
    if "cal_week_offset" not in st.session_state:
        st.session_state["cal_week_offset"] = 0
    offset = st.session_state["cal_week_offset"]
    week_start = today - timedelta(days=today.weekday()) + timedelta(weeks=offset)
    week_end   = week_start + timedelta(days=6)

    nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
    with nav_col1:
        if st.button("← Semaine préc.", use_container_width=True):
            st.session_state["cal_week_offset"] -= 1
            st.rerun()
    with nav_col2:
        st.markdown(
            f"<h4 style='text-align:center;margin:0;'>"
            f"{week_start.strftime('%d %b')} — {week_end.strftime('%d %b %Y')}"
            f"</h4>",
            unsafe_allow_html=True,
        )
    with nav_col3:
        if st.button("Semaine suiv. →", use_container_width=True):
            st.session_state["cal_week_offset"] += 1
            st.rerun()

    # Charger les matchs de la période
    with st.spinner("📅 Chargement du calendrier..."):
        all_matches = client.get_week_matches(days=14)  # 2 semaines de tampon

    # Récupérer les cotes si dispo
    all_odds_flat = []
    if odds_available:
        odds_by_comp = odds_cl.get_football_odds()
        all_odds_flat = [ev for evs in odds_by_comp.values() for ev in evs]

    # Indexer les matchs par date
    matches_by_date = defaultdict(list)
    for m in all_matches:
        d_str = m.get("match_date", "")[:10]
        try:
            d = date.fromisoformat(d_str)
            if week_start <= d <= week_end:
                matches_by_date[d_str].append(m)
        except ValueError:
            pass

    # Pré-calcul EV par date pour le code couleur
    ev_by_date = {}
    for d_str, day_matches in matches_by_date.items():
        best_ev = 0
        for m in day_matches:
            if m.get("status") == "FINISHED":
                continue
            probs = model.predict(m)
            if all_odds_flat:
                mo = odds_cl.match_odds_to_match(m, all_odds_flat)
                if mo:
                    ev_d = conservative_match_ev(model, probs, mo)
                    if ev_d:
                        best_ev = max(best_ev, ev_d.get("best_ev") or 0)
        ev_by_date[d_str] = best_ev

    # ── Grille calendrier ─────────────────────────────────────────────────
    days_fr = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    header_html = "".join(f'<div class="cal-day-header">{d}</div>' for d in days_fr)

    cells_html = ""
    selected_date = st.session_state.get("cal_selected_date")

    for i in range(7):
        d = week_start + timedelta(days=i)
        d_str = d.isoformat()
        day_matches_list = matches_by_date.get(d_str, [])
        n_matches = len(day_matches_list)
        best_ev = ev_by_date.get(d_str, 0)

        is_today = (d == today)
        has_value = best_ev >= MIN_RECOMMENDED_EV
        cell_class = "cal-cell today" if is_today else ("cal-cell has-value" if has_value else "cal-cell")

        # Badges matchs
        badges = ""
        for m in day_matches_list[:3]:
            home = m.get("home_team", "")[:8]
            away = m.get("away_team", "")[:8]
            badge_class = "cal-badge value" if has_value else "cal-badge normal"
            badges += f'<div class="{badge_class}">{home} v {away}</div>'
        if n_matches > 3:
            badges += f'<div class="cal-badge low">+{n_matches-3} matchs</div>'

        ev_indicator = ""
        if best_ev >= MIN_RECOMMENDED_EV:
            ev_indicator = f'<div style="color:#39d353;font-size:0.65rem;font-weight:700;">EV prudente +{round(best_ev*100,1)}%</div>'

        cell_style = "cursor:pointer;"
        cells_html += (
            f'<div class="{cell_class}" style="{cell_style}">'
            f'<div class="cal-date">{d.day}</div>'
            f'{ev_indicator}'
            f'{badges}'
            f'</div>'
        )


    st.markdown(
        f'<div class="cal-grid">{header_html}{cells_html}</div>',
        unsafe_allow_html=True,
    )

    # Sélection du jour via selectbox (fallback interactif)
    st.markdown("<br>", unsafe_allow_html=True)
    available_dates = sorted(matches_by_date.keys())
    if not available_dates:
        st.info("📅 Aucun match trouvé pour cette semaine.")
        return

    selected = st.selectbox(
        "📅 Voir les matchs du :",
        options=available_dates,
        format_func=lambda d: f"{d} ({len(matches_by_date[d])} matchs)",
        key="cal_day_select",
    )

    if selected:
        st.markdown(f"### 📅 Matchs du {selected}")
        day_m = matches_by_date[selected]
        cols = st.columns(2, gap="medium")
        for i, match in enumerate(day_m):
            with cols[i % 2]:
                probs = model.predict(match)
                render_match_card(match, probs, sport="football")
                # Cotes + EV si dispo
                if all_odds_flat:
                    mo = odds_cl.match_odds_to_match(match, all_odds_flat)
                    if mo:
                        odds = mo.get("best_odds", {})
                        ev = conservative_match_ev(model, probs, mo)
                        if ev is None:
                            st.caption("EV prudente indisponible : historique insuffisant.")
                            continue
                        ev_badges = " &nbsp;|&nbsp; ".join(filter(None, [
                            render_ev_badge(ev.get("home_win"), "1"),
                            render_ev_badge(ev.get("draw"), "X"),
                            render_ev_badge(ev.get("away_win"), "2"),
                        ]))
                        best_ev_val = ev.get("best_ev", 0) or 0
                        best_k = ev.get("best", "")
                        kelly_pct = kelly_fraction(
                            ev["probabilities"].get(best_k, 0),
                            odds.get(best_k, 0)
                        ) if best_k else 0
                        kelly_html = ""
                        if kelly_pct > 0:
                            kelly_html = (
                                f'<span style="color:#e3b341;font-size:0.78rem;">'
                                f"💰 Kelly: {kelly_pct}% de bankroll</span>"
                            )
                        st.markdown(f"""
                        <div style="margin:-0.5rem 0 0.5rem;font-size:0.82rem;">
                          EV prudente: {ev_badges}<br>{kelly_html}
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("---")



def render_ev_badge(ev_val: float | None, label: str) -> str:
    """Retourne un badge HTML coloré selon la valeur EV."""
    if ev_val is None:
        return ""
    if ev_val >= MIN_RECOMMENDED_EV:
        color = "#39d353"  # Vert vif = Value !
        icon = "🔥"
    elif ev_val > 0:
        color = "#58a6ff"  # Bleu = EV légèrement positive
        icon = "📈"
    elif ev_val > -0.05:
        color = "#e3b341"  # Jaune = neutre
        icon = "➖"
    else:
        color = "#f78166"  # Rouge = mauvaise value
        icon = "📉"
    pct = round(ev_val * 100, 1)
    sign = "+" if pct >= 0 else ""
    return f'<span style="color:{color};font-weight:700;font-size:0.82rem;">{icon} {label}: {sign}{pct}%</span>'


def render_week_tab():
    """
    Onglet 'Semaine + Cotes' : matchs des 7 prochains jours
    avec probabilités Dixon-Coles, cotes bookmakers et calcul EV.
    """
    client  = get_football_client()
    model   = get_football_model()
    odds_cl = get_odds_client()

    with st.spinner("📅 Chargement des matchs de la semaine..."):
        week_matches = client.get_week_matches(days=7)

    # Récupérer les cotes si l'API Odds est configurée
    odds_available = odds_cl._is_available()
    all_odds_flat = []
    if odds_available:
        with st.spinner("📊 Récupération des cotes bookmakers..."):
            odds_by_comp = odds_cl.get_football_odds()
            for events in odds_by_comp.values():
                all_odds_flat.extend(events)

    if not week_matches:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">📅</div>
          <div class="empty-text">Aucun match de football cette semaine<br>
          <small style="color:#8b949e">Vérifiez votre clé API dans .env</small>
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Grouper par date
    from collections import defaultdict
    matches_by_date = defaultdict(list)
    for m in week_matches:
        d = m.get("match_date", "?")[:10]
        matches_by_date[d].append(m)

    # Info banner
    st.markdown(f"""
    <div style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1rem;">
      <div class="stat-card" style="flex:1;min-width:120px;">
        <div class="stat-value">{len(week_matches)}</div>
        <div class="stat-label">Matchs cette semaine</div>
      </div>
      <div class="stat-card" style="flex:1;min-width:120px;">
        <div class="stat-value">{len(matches_by_date)}</div>
        <div class="stat-label">Jours de match</div>
      </div>
      <div class="stat-card" style="flex:1;min-width:120px;">
        <div class="stat-value">{len(all_odds_flat)}</div>
        <div class="stat-label">Cotes disponibles</div>
      </div>
      <div class="stat-card" style="flex:1;min-width:120px;">
        <div class="stat-value" style="color:{'#39d353' if odds_available else '#f78166'}">
          {'✅' if odds_available else '❌'}
        </div>
        <div class="stat-label">API Cotes</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not odds_available:
        st.info(
            "💡 Pour voir les cotes et la valeur espérée (EV), ajoutez une clé "
            "ODDS_API_KEY dans .env → Inscrivez-vous gratuitement sur the-odds-api.com"
        )

    # Filtre de compétition pour la semaine
    week_comps = sorted(list({m.get("competition", "") for m in week_matches if m.get("competition")}))
    sel_week_comp = st.selectbox(
        "🏆 Filtrer par compétition",
        options=["Toutes les compétitions"] + week_comps,
        index=0,
        key="week_comp_filter"
    )
    if sel_week_comp != "Toutes les compétitions":
        filtered_week = [m for m in week_matches if m.get("competition") == sel_week_comp]
        matches_by_date = defaultdict(list)
        for m in filtered_week:
            d = m.get("match_date", "?")[:10]
            matches_by_date[d].append(m)

    # Affichage par date
    for match_date in sorted(matches_by_date.keys()):
        day_matches = matches_by_date[match_date]

        # Header de la journée
        st.markdown(f"""
        <div style="margin:1.5rem 0 0.8rem;padding:0.5rem 1rem;background:linear-gradient(90deg,rgba(88,166,255,0.15),transparent);
                    border-left:3px solid #58a6ff;border-radius:0 8px 8px 0;">
          <span style="font-size:1.1rem;font-weight:700;color:#f0f6fc;">📅 {match_date}</span>
          <span style="margin-left:0.8rem;color:#8b949e;font-size:0.85rem;">{len(day_matches)} matchs</span>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(2, gap="medium")

        for i, match in enumerate(day_matches):
            with cols[i % 2]:
                probs = model.predict(match)
                render_match_card(match, probs, sport="football")

                # Chercher les cotes correspondantes et calculer l'EV
                if all_odds_flat:
                    matched_odds = odds_cl.match_odds_to_match(match, all_odds_flat)
                    if matched_odds:
                        odds = matched_odds.get("best_odds", {})
                        ev = conservative_match_ev(model, probs, matched_odds)
                        if ev is None:
                            st.caption("EV prudente indisponible : historique insuffisant.")
                            continue

                        # Affichage cotes + EV
                        ev_badges = " &nbsp;|&nbsp; ".join(filter(None, [
                            render_ev_badge(ev.get("home_win"), "1"),
                            render_ev_badge(ev.get("draw"), "X"),
                            render_ev_badge(ev.get("away_win"), "2"),
                        ]))

                        best_ev = ev.get("best_ev", 0)
                        value_alert = ""
                        if best_ev and best_ev >= MIN_RECOMMENDED_EV:
                            outcome_label = {"home_win": "Domicile", "draw": "Nul", "away_win": "Extérieur"}
                            value_alert = f"""
                            <div style="background:rgba(57,211,83,0.1);border:1px solid #39d353;
                                        border-radius:8px;padding:0.5rem 0.8rem;margin-top:0.4rem;text-align:center;">
                              🔥 <strong style="color:#39d353;">Signal de valeur prudent</strong>
                              <span style="color:#f0f6fc;font-size:0.85rem;">
                                {outcome_label.get(ev.get('best'), '?')} @ {odds.get(ev.get('best', ''), 0):.2f}
                                (EV: +{round(best_ev * 100, 1)}%)
                              </span>
                            </div>
                            """

                        st.markdown(f"""
                        <div style="margin:-0.5rem 0 0.4rem;">
                          <div style="font-size:0.78rem;color:#8b949e;margin-bottom:0.3rem;">&#x1F4B0; Meilleures cotes:
                            <span style="color:#f0f6fc;">1: {odds.get('home_win', 0):.2f}</span> |
                            <span style="color:#f0f6fc;">X: {odds.get('draw', 0):.2f}</span> |
                            <span style="color:#f0f6fc;">2: {odds.get('away_win', 0):.2f}</span>
                            <small style="color:#8b949e;">({matched_odds.get('n_bookmakers', '?')} bookmakers)</small>
                          </div>
                          <div style="font-size:0.82rem;">EV prudente: {ev_badges}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Value alert dans son propre st.markdown (evite conflit triple-quotes)
                        best_ev = ev.get("best_ev", 0)
                        if best_ev and best_ev >= MIN_RECOMMENDED_EV:
                            outcome_label = {"home_win": "Domicile", "draw": "Nul", "away_win": "Extérieur"}
                            pick      = outcome_label.get(ev.get("best", ""), "?")
                            odd_val   = odds.get(ev.get("best", ""), 0)
                            ev_pct    = round(best_ev * 100, 1)
                            st.markdown(
                                f'<div style="background:rgba(45,202,86,0.12);border:1.5px solid #2dca56;'
                                f'border-radius:12px;padding:0.7rem 1rem;margin-bottom:0.4rem;text-align:center;">'
                                f'<div style="font-size:1rem;font-weight:700;color:var(--green);">'
                                f'\U0001f525 Signal prudent &mdash; {pick} @ {odd_val:.2f}</div>'
                                f'<div style="font-size:0.78rem;color:rgba(45,202,86,0.8);">'
                                f'EV +{ev_pct}% · {matched_odds.get("n_bookmakers","?")}</div></div>',
                                unsafe_allow_html=True,
                            )

                        st.markdown("---")



def render_performance_tab():
    """
    Onglet Performance : évalue la qualité des prédictions passées.
    - Accuracy globale et par sport
    - Brier Score (qualité de calibration, plus bas = mieux)
    - Graphique de calibration : prob prédite vs fréquence observée
    - Tableau des 20 dernières prédictions avec résultat
    """
    from db import get_connection as get_db_connection
    import numpy as np

    st.markdown("### 📊 Performance du modèle")
    st.markdown(
        "<small style='color:#8b949e'>Basé sur la première prédiction pré-match sauvegardée "
        "et comparée au score final.</small>",
        unsafe_allow_html=True,
    )

    # ── Récupérer les prédictions loggées ─────────────────────────────────
    try:
        conn = get_db_connection()
        rows = conn.execute("""
            SELECT p.match_id, p.sport,
                   p.home_win_prob AS predicted_home_win,
                   p.draw_prob AS predicted_draw,
                   p.away_win_prob AS predicted_away_win,
                   p.minute,
                   m.home_score, m.away_score
            FROM predictions p
            JOIN match_history m ON p.match_id = m.id AND p.sport = m.sport
            WHERE p.minute IS NULL
              AND m.home_score IS NOT NULL
              AND m.away_score IS NOT NULL
              AND p.id = (
                  SELECT MIN(first_p.id)
                  FROM predictions first_p
                  WHERE first_p.match_id = p.match_id
                    AND first_p.sport = p.sport
                    AND first_p.minute IS NULL
              )
            ORDER BY p.id DESC
            LIMIT 200
        """).fetchall()
        conn.close()
    except Exception as e:
        st.warning(f"⚠️ Base de données : {e}")
        rows = []

    if not rows:
        st.info(
            "📭 Pas encore de données de performance.\n\n"
            "Les pronostics pré-match sont enregistrés pour les matchs à venir, "
            "puis évalués une fois leur résultat final disponible."
        )
        # Afficher quand même le guide de lecture
        st.markdown("---")
        st.markdown("#### 📖 Comment lire les métriques ?")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Accuracy**
            Proportion de matchs où le pronostic le plus probable
            correspond au résultat final.
            - > 55% → Excellent
            - 50-55% → Bien
            - < 50% → À revoir
            """)
        with col2:
            st.markdown("""
            **Brier Score**
            Mesure l'erreur quadratique des probabilités (0 = parfait, 1 = pire).
            - < 0.20 → Excellent
            - 0.20-0.25 → Bien
            - > 0.25 → À améliorer

            *Formule : BS = mean((p_outcome - 1_outcome)²)*
            """)
        return

    # ── Calcul des métriques ───────────────────────────────────────────────
    records = []
    for r in rows:
        hw = r["predicted_home_win"] or 0
        d  = r["predicted_draw"]     or 0
        aw = r["predicted_away_win"] or 0
        hs = r["home_score"] or 0
        as_ = r["away_score"] or 0

        if hs > as_:
            actual = "home_win"
        elif hs == as_:
            actual = "draw"
        else:
            actual = "away_win"

        pred_outcome = max({"home_win": hw, "draw": d, "away_win": aw},
                           key=lambda k: {"home_win": hw, "draw": d, "away_win": aw}[k])
        correct = (pred_outcome == actual)

        # Brier Score pour cet événement (3 issues)
        brier = (
            (hw - (1 if actual == "home_win" else 0)) ** 2 +
            (d  - (1 if actual == "draw"     else 0)) ** 2 +
            (aw - (1 if actual == "away_win" else 0)) ** 2
        ) / 3

        records.append({
            "sport":        r["sport"],
            "match_id":     r["match_id"],
            "hw": hw, "d": d, "aw": aw,
            "actual":       actual,
            "predicted":    pred_outcome,
            "correct":      correct,
            "brier":        brier,
            "score":        f"{hs}-{as_}",
        })

    df = pd.DataFrame(records)
    n  = len(df)
    accuracy    = round(df["correct"].mean() * 100, 1)
    brier_score = round(df["brier"].mean(), 3)

    # ── KPIs ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    acc_color   = "#39d353" if accuracy > 55 else "#e3b341" if accuracy > 48 else "#f78166"
    brier_color = "#39d353" if brier_score < 0.20 else "#e3b341" if brier_score < 0.25 else "#f78166"
    c1.markdown(f"""
    <div class="stat-card">
      <div class="stat-value" style="color:{acc_color}">{accuracy}%</div>
      <div class="stat-label">Accuracy</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""
    <div class="stat-card">
      <div class="stat-value" style="color:{brier_color}">{brier_score}</div>
      <div class="stat-label">Brier Score ↓</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""
    <div class="stat-card">
      <div class="stat-value">{n}</div>
      <div class="stat-label">Matchs évalués</div>
    </div>""", unsafe_allow_html=True)
    correct_n = int(df["correct"].sum())
    c4.markdown(f"""
    <div class="stat-card">
      <div class="stat-value" style="color:#39d353">{correct_n}</div>
      <div class="stat-label">Pronostics corrects</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Accuracy par sport ─────────────────────────────────────────────────
    by_sport = df.groupby("sport")["correct"].agg(["mean", "count"]).reset_index()
    by_sport.columns = ["Sport", "Accuracy", "Matchs"]
    by_sport["Accuracy"] = (by_sport["Accuracy"] * 100).round(1)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("#### 🏅 Par sport")
        for _, row in by_sport.iterrows():
            emoji = "⚽" if row["Sport"] == "football" else "🏀"
            color = "#39d353" if row["Accuracy"] > 55 else "#e3b341" if row["Accuracy"] > 48 else "#f78166"
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid var(--border-color);
                        border-radius:10px;padding:0.8rem 1rem;margin-bottom:0.6rem;">
              <div style="font-size:1rem;font-weight:700;">{emoji} {row['Sport'].title()}</div>
              <div style="color:{color};font-size:1.5rem;font-weight:800;">{row['Accuracy']}%</div>
              <div style="color:#8b949e;font-size:0.75rem;">{int(row['Matchs'])} matchs</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Graphique de calibration ───────────────────────────────────────────
    with col_right:
        st.markdown("#### 📈 Courbe de calibration")
        # Bin les probabilités prédites par tranche de 10%
        all_probs   = []
        all_actuals = []
        for _, row in df.iterrows():
            for outcome, prob_col in [("home_win","hw"),("draw","d"),("away_win","aw")]:
                all_probs.append(row[prob_col])
                all_actuals.append(1 if row["actual"] == outcome else 0)

        bins = np.linspace(0, 1, 11)
        bin_centers, obs_freq = [], []
        for i in range(len(bins) - 1):
            mask = [(bins[i] <= p < bins[i+1]) for p in all_probs]
            if sum(mask) >= 3:
                bin_centers.append((bins[i] + bins[i+1]) / 2)
                obs_freq.append(np.mean([all_actuals[j] for j, m in enumerate(mask) if m]))

        fig_cal = go.Figure()
        # Ligne parfaite
        fig_cal.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines", name="Calibration parfaite",
            line=dict(color="#30363d", dash="dash", width=1.5),
        ))
        # Points observés
        if bin_centers:
            fig_cal.add_trace(go.Scatter(
                x=bin_centers, y=obs_freq,
                mode="lines+markers", name="Modèle",
                line=dict(color="#58a6ff", width=2.5),
                marker=dict(size=8, color="#58a6ff"),
                hovertemplate="Prob prédite: %{x:.0%}<br>Fréq observée: %{y:.0%}<extra></extra>",
            ))
        fig_cal.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(22,27,34,0.8)",
            font=dict(family="Inter", color="#8b949e"),
            xaxis=dict(title="Probabilité prédite", tickformat=".0%",
                       gridcolor="rgba(48,54,61,0.5)", range=[0,1]),
            yaxis=dict(title="Fréquence observée", tickformat=".0%",
                       gridcolor="rgba(48,54,61,0.5)", range=[0,1]),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#f0f6fc")),
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
        )
        st.plotly_chart(fig_cal, use_container_width=True, key="calibration_chart")

    # ── Tableau des 20 dernières prédictions ──────────────────────────────
    st.markdown("#### 🗒️ Dernières prédictions")
    display_df = df.head(20)[["match_id","sport","predicted","actual","correct","score","brier"]].copy()
    display_df.columns = ["Match ID","Sport","Pronostic","Résultat","✓","Score","Brier"]
    display_df["✓"] = display_df["✓"].map({True: "✅", False: "❌"})
    display_df["Pronostic"] = display_df["Pronostic"].map({
        "home_win": "🏠 Domicile", "draw": "🤝 Nul", "away_win": "✈️ Extérieur"
    })
    display_df["Résultat"] = display_df["Résultat"].map({
        "home_win": "🏠 Domicile", "draw": "🤝 Nul", "away_win": "✈️ Extérieur"
    })
    display_df["Brier"] = display_df["Brier"].round(3)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Interface principale
# ═══════════════════════════════════════════════════════════════════════════

def main():

    # ── Sidebar ───────────────────────────────────────────────────────────
    render_sidebar()

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
      <h1 class="main-title">SportsPredikt</h1>
    </div>
    """, unsafe_allow_html=True)

    # ── Avertissement légal ────────────────────────────────────────────────
    st.markdown("""
    <div class="warning-banner">
      Estimations indicatives · Aucun résultat garanti · Pas un conseil de pari.
    </div>
    """, unsafe_allow_html=True)

    # ── Rafraîchissement automatique ───────────────────────────────────────
    refresh_col1, refresh_col2 = st.columns([1, 1])
    with refresh_col1:
        auto_refresh = st.toggle("Actualisation auto", value=True, key="auto_refresh")
    with refresh_col2:
        last_update = st.empty()
        last_update.markdown(
            f"<small style='color:#8b949e'>Actualisé à {datetime.now().strftime('%H:%M')}</small>",
            unsafe_allow_html=True
        )

    # ── Onglets principaux ─────────────────────────────────────────────────
    tab_foot, tab_bask, tab_week, tab_vb, tab_bets, tab_cal, tab_perf = st.tabs(
        ["Football", "Basket", "Semaine", "Value bets", "Mes paris", "Calendrier", "Bilan"]
    )

    with tab_foot:
        render_football_tab()

    with tab_bask:
        render_basketball_tab()

    with tab_week:
        render_week_tab()

    with tab_vb:
        render_value_bets_tab()

    with tab_bets:
        render_bets_journal_tab()

    with tab_cal:
        render_calendar_tab()

    with tab_perf:
        render_performance_tab()

    # ── Pied de page ───────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem;color:#8b949e;font-size:0.72rem;border-top:1px solid var(--border);margin-top:1.5rem;">
      SportsPredikt v2 · Probabilités indicatives — aucun résultat garanti
    </div>
    """, unsafe_allow_html=True)

    # ── Auto-refresh toutes les 30 secondes ────────────────────────────────
    if auto_refresh:
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()
