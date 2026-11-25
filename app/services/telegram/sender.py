"""Telegram delivery. Plain text only; dry-run by default (SIGNALS_DRY_RUN)."""

import logging

import httpx
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.time import utcnow
from app.db.models import Match, Signal
from app.services.signals.messages import format_signal

log = logging.getLogger(__name__)

DEMO_LEAGUE = "Demo League"


def send_telegram_message(text: str, client: httpx.Client | None = None) -> str:
    """Send one message and return its message id. Raises on any failure."""
    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHANNEL_ID:
        raise RuntimeError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHANNEL_ID not set")
    own = client is None
    client = client or httpx.Client(timeout=15)
    try:
        resp = client.post(
            f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": settings.TELEGRAM_CHANNEL_ID, "text": text,
                  "disable_web_page_preview": True},
        )
        resp.raise_for_status()
        return str(resp.json().get("result", {}).get("message_id"))
    finally:
        if own:
            client.close()


def dispatch_signals(db: Session, client: httpx.Client | None = None) -> dict:
    """Deliver pending signals. Dry-run records the text without sending anything.

    Demo-league signals are never sent, whatever the settings.
    """
    counts = {"dry_run": 0, "sent": 0, "failed": 0}
    pending = db.query(Signal).filter(Signal.status == "pending").order_by(Signal.id).all()
    for sig in pending:
        match = db.get(Match, sig.match_id)
        text = format_signal(sig, match)
        live = not settings.SIGNALS_DRY_RUN and match.league != DEMO_LEAGUE
        if not live:
            log.info("[dry-run] signal %s\n%s", sig.id, text)
            sig.status = "dry_run"
            sig.dispatched_at = utcnow()
            counts["dry_run"] += 1
            continue
        try:
            sig.telegram_message_id = send_telegram_message(text, client)
            sig.channel_name = settings.TELEGRAM_CHANNEL_ID
            sig.status = "sent"
            sig.dispatched_at = utcnow()
            counts["sent"] += 1
        except Exception as exc:  # network / API error: keep going, retry next run
            log.error("telegram send failed for signal %s: %s", sig.id, type(exc).__name__)
            sig.status = "failed"
            counts["failed"] += 1
    db.commit()
    return counts
