import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from sqlalchemy.orm import Session

import app.db.models as models
from app.core.config import settings
from app.core.time import utcnow
from app.db.session import Base, engine, get_db

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # MVP only; use Alembic migrations in production.
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="LeviBet Sports Analytics", lifespan=lifespan)


def _row(obj, fields):
    return {f: getattr(obj, f) for f in fields}


MATCH_FIELDS = ["id", "league", "home_team", "away_team", "kickoff_at", "status"]
PRED_FIELDS = ["id", "match_id", "model_version", "as_of", "home_win_prob", "draw_prob",
               "away_win_prob", "expected_home_goals", "expected_away_goals",
               "confidence_score", "edge_score", "extra"]
SIGNAL_FIELDS = ["id", "match_id", "selection", "bookmaker", "odds", "model_prob", "prob_std",
                 "market_prob", "edge", "conservative_edge", "status", "outcome", "pnl", "clv",
                 "created_at"]


@app.get("/health")
def health_check():
    return {"status": "ok", "dry_run": settings.SIGNALS_DRY_RUN}


@app.get("/matches/upcoming")
def get_upcoming_matches(db: Session = Depends(get_db)):
    rows = (
        db.query(models.Match)
        .filter(models.Match.status == "scheduled", models.Match.kickoff_at >= utcnow())
        .order_by(models.Match.kickoff_at)
        .limit(15)
        .all()
    )
    return {"matches": [_row(m, MATCH_FIELDS) for m in rows]}


@app.get("/predictions/latest")
def get_latest_predictions(db: Session = Depends(get_db)):
    rows = db.query(models.Prediction).order_by(models.Prediction.as_of.desc()).limit(15).all()
    return {"predictions": [_row(p, PRED_FIELDS) for p in rows]}


@app.get("/signals/latest")
def get_latest_signals(db: Session = Depends(get_db)):
    rows = db.query(models.Signal).order_by(models.Signal.id.desc()).limit(15).all()
    return {"signals": [_row(s, SIGNAL_FIELDS) for s in rows]}


@app.get("/performance")
def get_performance(db: Session = Depends(get_db)):
    from app.services.signals.performance import performance

    return performance(db)


def _train_job():
    from app.services.bayesian.train_team_strength import train_all

    try:
        log.info("training finished: %s", train_all())
    except Exception:
        log.exception("training failed")


@app.post("/admin/train-bayes/run")
def run_bayes_training(background: BackgroundTasks, x_admin_token: str = Header(default="")):
    if not settings.ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="admin endpoints disabled (ADMIN_TOKEN not set)")
    if not hmac.compare_digest(x_admin_token, settings.ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="invalid admin token")
    background.add_task(_train_job)
    return {"message": "Bayesian training started in the background."}
