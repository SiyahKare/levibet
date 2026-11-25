"""Background pipeline: ingest odds -> predict -> signals -> dispatch -> settle."""

import logging
import time

from app.core.config import settings
from app.db.session import SessionLocal

log = logging.getLogger(__name__)


def run_pipeline_once(ingest: bool = True) -> dict:
    from app.services.predictions.generate_predictions import generate_predictions
    from app.services.signals.engine import generate_signals
    from app.services.signals.settlement import settle_signals
    from app.services.telegram.sender import dispatch_signals

    summary: dict = {}
    db = SessionLocal()
    try:
        if ingest and settings.ODDS_API_KEY:
            from app.services.odds.ingest import ingest_the_odds_api

            summary["ingest"] = ingest_the_odds_api(db)
        summary["predictions"] = len(generate_predictions(db))
        summary["signals"] = len(generate_signals(db))
        summary["dispatch"] = dispatch_signals(db)
        summary["settled"] = settle_signals(db)
    finally:
        db.close()
    return summary


def run_prediction_loop() -> None:
    log.info("worker started (dry_run=%s)", settings.SIGNALS_DRY_RUN)
    while True:
        try:
            log.info("pipeline: %s", run_pipeline_once())
        except Exception:
            log.exception("pipeline run failed")
        time.sleep(settings.WORKER_INTERVAL_SECONDS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_prediction_loop()
