"""Track record of settled signals. Flat 1-unit stakes, purely for evaluation."""

import math

from sqlalchemy.orm import Session

from app.db.models import Signal


def performance(db: Session, include_demo: bool = True) -> dict:
    from app.db.models import Match

    q = db.query(Signal, Match).join(Match, Match.id == Signal.match_id).filter(Signal.outcome.isnot(None))
    if not include_demo:
        q = q.filter(Match.league != "Demo League")
    rows = [s for s, _ in q.all()]
    graded = [s for s in rows if s.outcome in ("win", "loss")]
    n = len(graded)
    out: dict = {"settled": len(rows), "graded": n, "voids": len(rows) - n}
    if n == 0:
        return out

    pnl = [s.pnl for s in graded]
    mean = sum(pnl) / n
    var = sum((x - mean) ** 2 for x in pnl) / (n - 1) if n > 1 else 0.0
    wins = sum(1 for s in graded if s.outcome == "win")
    clvs = [s.clv for s in graded if s.clv is not None]

    # Brier / log loss of the picked outcome's probability (win = 1).
    brier = sum((s.model_prob - (1 if s.outcome == "win" else 0)) ** 2 for s in graded) / n
    eps = 1e-9
    logloss = -sum(
        math.log(max(eps, s.model_prob if s.outcome == "win" else 1 - s.model_prob)) for s in graded
    ) / n

    bins = []
    for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
        part = [s for s in graded if lo <= s.model_prob < lo + 0.2 or (lo == 0.8 and s.model_prob == 1.0)]
        if part:
            bins.append(
                {
                    "range": f"{lo:.1f}-{lo + 0.2:.1f}",
                    "n": len(part),
                    "avg_pred": sum(s.model_prob for s in part) / len(part),
                    "hit_rate": sum(1 for s in part if s.outcome == "win") / len(part),
                }
            )

    out.update(
        {
            "hit_rate": wins / n,
            "roi": mean,
            "roi_se": math.sqrt(var / n) if n > 1 else None,
            "total_pnl": sum(pnl),
            "avg_clv": sum(clvs) / len(clvs) if clvs else None,
            "brier": brier,
            "log_loss": logloss,
            "calibration": bins,
            "note": "Small samples are noisy: judge ROI together with its standard error and CLV.",
        }
    )
    return out
