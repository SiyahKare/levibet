"""Compare model probabilities with devigged market odds and emit value signals.

A signal is *information*: "the model rates this outcome higher than the market
does". It is not betting advice and the engine never places or sizes a bet.

edge              = p_model * odds - 1
conservative_edge = (p_model - k * p_std) * odds - 1     (k = UNCERTAINTY_K)
A signal needs conservative_edge >= MIN_EDGE. Edges above MAX_EDGE are treated
as a probable data or model error and dropped.
"""

import logging
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.time import as_utc, utcnow
from app.db.models import Match, OddsSnapshot, Prediction, Signal
from app.services.odds.devig import devig

log = logging.getLogger(__name__)

SELECTIONS = ("H", "D", "A")
ODDS_ATTR = {"H": "home_odds", "D": "draw_odds", "A": "away_odds"}


@dataclass
class Candidate:
    selection: str
    bookmaker: str
    odds: float
    odds_captured_at: object
    model_prob: float
    prob_std: float
    market_prob: float
    edge: float
    conservative_edge: float


def latest_snapshots(db: Session, match_id: int, now, max_age_minutes: int) -> list[OddsSnapshot]:
    """Newest non-closing snapshot per bookmaker that is not older than the age limit."""
    cutoff = now - timedelta(minutes=max_age_minutes)
    rows = (
        db.query(OddsSnapshot)
        .filter(OddsSnapshot.match_id == match_id, OddsSnapshot.is_closing.is_(False))
        .order_by(OddsSnapshot.captured_at.desc())
        .all()
    )
    out: dict[str, OddsSnapshot] = {}
    for r in rows:
        if as_utc(r.captured_at) < cutoff or as_utc(r.captured_at) > now:
            continue
        out.setdefault(r.bookmaker, r)
    return list(out.values())


def consensus_market_probs(snaps: list[OddsSnapshot]) -> list[float]:
    """Average of the per-bookmaker devigged probabilities."""
    per_book = [devig([s.home_odds, s.draw_odds, s.away_odds]) for s in snaps]
    n = len(per_book)
    avg = [sum(p[i] for p in per_book) / n for i in range(3)]
    total = sum(avg)
    return [x / total for x in avg]


def evaluate(pred: Prediction, snaps: list[OddsSnapshot]) -> list[Candidate]:
    """All selections that pass the filters, best conservative edge first."""
    if not snaps:
        return []
    market = consensus_market_probs(snaps)
    model = [pred.home_win_prob, pred.draw_prob, pred.away_win_prob]
    stds = (pred.extra or {}).get("std", {})
    out = []
    for i, sel in enumerate(SELECTIONS):
        best = max(snaps, key=lambda s: getattr(s, ODDS_ATTR[sel]))
        odds = getattr(best, ODDS_ATTR[sel])
        if odds < settings.MIN_ODDS or odds > settings.MAX_ODDS:
            continue
        p, sd = model[i], float(stds.get(sel, 0.0))
        edge = p * odds - 1.0
        cons = (p - settings.UNCERTAINTY_K * sd) * odds - 1.0
        if cons < settings.MIN_EDGE or edge > settings.MAX_EDGE:
            continue
        out.append(
            Candidate(sel, best.bookmaker, odds, best.captured_at, p, sd, market[i], edge, cons)
        )
    out.sort(key=lambda c: c.conservative_edge, reverse=True)
    return out


def generate_signals(db: Session | None = None, now=None) -> list[Signal]:
    """Create new signals for upcoming matches. One signal per match, deduplicated."""
    from app.db.session import SessionLocal

    own = db is None
    db = db or SessionLocal()
    now = now or utcnow()
    try:
        earliest = now + timedelta(minutes=settings.MIN_MINUTES_TO_KICKOFF)
        horizon = now + timedelta(hours=settings.PREDICTION_HORIZON_HOURS)
        matches = (
            db.query(Match)
            .filter(Match.status == "scheduled", Match.kickoff_at >= earliest, Match.kickoff_at <= horizon)
            .all()
        )
        found: list[tuple[Prediction, Candidate]] = []
        for m in matches:
            pred = (
                db.query(Prediction)
                .filter(Prediction.match_id == m.id)
                .order_by(Prediction.as_of.desc(), Prediction.id.desc())
                .first()
            )
            if pred is None or pred.draw_prob is None:
                continue
            snaps = latest_snapshots(db, m.id, now, settings.MAX_ODDS_AGE_MINUTES)
            cands = evaluate(pred, snaps)
            pred.edge_score = cands[0].conservative_edge if cands else None
            if not cands:
                continue
            exists = db.query(Signal.id).filter(Signal.match_id == m.id, Signal.market == "1X2").first()
            if exists:
                continue
            found.append((pred, cands[0]))

        found.sort(key=lambda t: t[1].conservative_edge, reverse=True)
        created = []
        for pred, c in found[: settings.MAX_SIGNALS_PER_RUN]:
            sig = Signal(
                prediction_id=pred.id,
                match_id=pred.match_id,
                model_version=pred.model_version,
                market="1X2",
                selection=c.selection,
                bookmaker=c.bookmaker,
                odds=c.odds,
                odds_captured_at=c.odds_captured_at,
                model_prob=c.model_prob,
                prob_std=c.prob_std,
                market_prob=c.market_prob,
                edge=c.edge,
                conservative_edge=c.conservative_edge,
                status="pending",
            )
            db.add(sig)
            created.append(sig)
        db.commit()
        return created
    finally:
        if own:
            db.close()
