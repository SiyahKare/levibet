"""Settle signals once matches are finished: outcome, flat 1-unit P&L and CLV."""

from sqlalchemy.orm import Session

from app.core.time import utcnow
from app.db.models import Match, OddsSnapshot, Signal
from app.services.odds.devig import devig

ODDS_ATTR = {"H": 0, "D": 1, "A": 2}


def result_of(match: Match) -> str | None:
    if match.home_goals is None or match.away_goals is None:
        return None
    if match.home_goals > match.away_goals:
        return "H"
    if match.home_goals < match.away_goals:
        return "A"
    return "D"


def closing_snapshot(db: Session, sig: Signal) -> OddsSnapshot | None:
    """Prefer the signal's own bookmaker's closing line, else any closing line."""
    q = db.query(OddsSnapshot).filter(
        OddsSnapshot.match_id == sig.match_id, OddsSnapshot.is_closing.is_(True)
    )
    own = q.filter(OddsSnapshot.bookmaker == sig.bookmaker).order_by(OddsSnapshot.captured_at.desc()).first()
    return own or q.order_by(OddsSnapshot.captured_at.desc()).first()


def settle_signal(db: Session, sig: Signal, match: Match) -> bool:
    if match.status == "void":
        sig.outcome, sig.pnl = "void", 0.0
    else:
        res = result_of(match)
        if match.status != "finished" or res is None:
            return False
        if res == sig.selection:
            sig.outcome, sig.pnl = "win", sig.odds - 1.0
        else:
            sig.outcome, sig.pnl = "loss", -1.0

    snap = closing_snapshot(db, sig)
    if snap is not None:
        close = [snap.home_odds, snap.draw_odds, snap.away_odds]
        sig.closing_odds = close[ODDS_ATTR[sig.selection]]
        fair_close = devig(close)[ODDS_ATTR[sig.selection]]
        # CLV: how much better than the fair closing price the odds taken were.
        sig.clv = sig.odds * fair_close - 1.0
    sig.settled_at = utcnow()
    return True


def settle_signals(db: Session) -> int:
    open_signals = db.query(Signal).filter(Signal.outcome.is_(None)).all()
    n = 0
    for sig in open_signals:
        match = db.get(Match, sig.match_id)
        if settle_signal(db, sig, match):
            n += 1
    db.commit()
    return n
