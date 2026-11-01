# -*- coding: utf-8 -*-
"""
telegram_alert.py - Envoi d'alertes instantanées de Value Bets sur Telegram
Permet de recevoir les meilleures opportunités détectées directement sur smartphone.
"""

import os
import logging
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def is_telegram_configured(bot_token: str | None = None, chat_id: str | None = None) -> bool:
    """Vérifie si les identifiants Telegram sont présents."""
    token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat  = chat_id or os.getenv("TELEGRAM_CHAT_ID", "").strip()
    return bool(token and chat)


def send_telegram_message(
    text: str,
    bot_token: str | None = None,
    chat_id: str | None = None,
    parse_mode: str = "HTML",
) -> tuple[bool, str]:
    """
    Envoie un message via l'API Telegram Bot.

    Args:
        text: Corps du message (formaté en HTML)
        bot_token: Token du bot (par défaut depuis .env)
        chat_id: Chat ID du destinataire (par défaut depuis .env)
        parse_mode: 'HTML' ou 'Markdown'

    Returns:
        (succès: bool, message_retour: str)
    """
    token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    cid   = chat_id or os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token or not cid:
        return False, "Token de bot ou Chat ID Telegram non configuré."

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": cid,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": False,
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            return True, "Message envoyé avec succès sur Telegram !"
        else:
            err_msg = resp.json().get("description", resp.text)
            logger.error("Erreur Telegram API (%d): %s", resp.status_code, err_msg)
            return False, f"Erreur Telegram: {err_msg}"
    except Exception as e:
        logger.error("Exception lors de l'envoi Telegram: %s", e)
        return False, f"Erreur réseau: {e}"


def send_value_bets_alert(
    value_bets: list[dict],
    bankroll: float = 50000,
    currency: str = "FCFA",
    bot_token: str | None = None,
    chat_id: str | None = None,
) -> tuple[bool, str]:
    """
    Formate et envoie les meilleurs Value Bets détectés vers Telegram.
    """
    if not value_bets:
        return False, "Aucun Value Bet à envoyer."

    top_bets = value_bets[:5]  # Limiter aux 5 meilleures opportunités

    lines = [
        "🎯 <b>SPORTSPREDIKT — ALERTES VALUE BETS</b>",
        f"📅 Date: <b>Aujourd'hui</b> | Devise: <b>{currency}</b>",
        f"💳 Bankroll de référence: <b>{int(bankroll):,} {currency}</b>".replace(",", " "),
        "─────────────────────────",
    ]

    for i, vb in enumerate(top_bets, 1):
        ev_pct = round(vb.get("ev", 0) * 100, 1)
        prob_pct = round(vb.get("prob", 0) * 100, 1)
        stake = vb.get("stake") or vb.get("stake_amount") or 0
        stake_str = f"{int(stake):,} {currency}".replace(",", " ") if currency == "FCFA" else f"{stake:.2f} €"

        lines.append(
            f"🔥 <b>#{i} {vb.get('match', '?')}</b>\n"
            f"🏆 {vb.get('competition', '')}\n"
            f"🎯 <b>Miser sur :</b> {vb.get('pick', '?')} @ <b>{vb.get('odd', 0):.2f}</b>\n"
            f"📈 EV: <b>+{ev_pct}%</b> | Prob modèle: <b>{prob_pct}%</b>\n"
            f"💰 Mise Kelly conseillée: <b>{stake_str}</b> ({vb.get('kelly_pct', 0)}%)\n"
        )

    lines.append("─────────────────────────")
    lines.append('🔗 <a href="https://www.congobet.net/sports">Parier sur Congobet</a>')
    lines.append("⚠️ <i>Probabilités statistiques indicatives. Jouez de manière responsable.</i>")

    full_message = "\n".join(lines)
    return send_telegram_message(full_message, bot_token=bot_token, chat_id=chat_id)
