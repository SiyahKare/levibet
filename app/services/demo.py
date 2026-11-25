"""Synthetic "Demo League" so the whole pipeline can run without any real data.

Teams have hidden true strengths. A double round-robin of results is simulated
and priced with a bookmaker margin. Upcoming fixtures are priced with some
deliberate mispricing, so signals can appear. Demo signals are never sent to
Telegram and carry no real-world meaning.
"""

from datetime import timedelta

import numpy as np
from sqlalchemy.orm import Session

from app.core.time import utcnow
from app.db.models import Match
from app.services.bayesian import poisson
from app.services.odds.ingest import _add_snapshot, record_result

LEAGUE = "Demo League"
TEAMS = ["Alpha FC", "Bravo United", "Charlie City", "Delta Town",
         "Echo Rovers", "Foxtrot Athletic", "Golf Wanderers", "Hotel Albion"]
INTERCEPT, HOME_ADV, MARGIN = 0.25, 0.25, 0.05


def _true_probs(att, dfn, h, a):
    lam_h = np.exp(INTERCEPT + HOME_ADV + att[h] - dfn[a])
    lam_a = np.exp(INTERCEPT + att[a] - dfn[h])
    return poisson.outcome_probs(poisson.score_matrix([lam_h], [lam_a]))[0], lam_h, lam_a


def _price(probs, margin=MARGIN):
    return tuple(round(1.0 / (p * (1 + margin)), 2) for p in probs)


def seed_demo(db: Session, seed: int = 7, now=None) -> dict:
    if db.query(Match.id).filter(Match.league == LEAGUE).first():
        return {"skipped": "demo data already present"}
    rng = np.random.default_rng(seed)
    now = now or utcnow()
    n = len(TEAMS)
    att = rng.normal(0, 0.3, n)
    att -= att.mean()
    dfn = rng.normal(0, 0.3, n)
    dfn -= dfn.mean()

    counts = {"finished": 0, "upcoming": 0}
    pairs = [(h, a) for h in range(n) for a in range(n) if h != a]
    # three rounds of results -> 168 matches over the past ~170 days
    for r in range(3):
        for k, (h, a) in enumerate(pairs):
            kick = (now - timedelta(days=170) + timedelta(days=(r * len(pairs) + k) * 170 / (3 * len(pairs)))
                    ).replace(minute=0, second=0, microsecond=0)
            probs, lh, la = _true_probs(att, dfn, h, a)
            m = Match(external_match_id=f"demo:{r}:{h}:{a}", sport="football", league=LEAGUE,
                      home_team=TEAMS[h], away_team=TEAMS[a], kickoff_at=kick, status="scheduled")
            db.add(m)
            db.flush()
            record_result(db, m, rng.poisson(lh), rng.poisson(la))
            closing = _price(probs)
            _add_snapshot(db, m, "DemoBook", closing, kick - timedelta(hours=24))
            _add_snapshot(db, m, "DemoBook", closing, kick - timedelta(minutes=5), True)
            counts["finished"] += 1

    # upcoming fixtures; bookmaker misprices one outcome by a random factor
    picks = rng.permutation(pairs)[:8]
    for i, (h, a) in enumerate(picks):
        probs, _, _ = _true_probs(att, dfn, h, a)
        skew = np.ones(3)
        skew[rng.integers(0, 3)] = rng.uniform(0.80, 0.92)  # book underrates one outcome
        mispriced = probs * skew
        mispriced /= mispriced.sum()
        kick = (now + timedelta(hours=6 + 5 * i)).replace(minute=0, second=0, microsecond=0)
        m = Match(external_match_id=f"demo:up:{i}", sport="football", league=LEAGUE,
                  home_team=TEAMS[int(h)], away_team=TEAMS[int(a)], kickoff_at=kick, status="scheduled")
        db.add(m)
        db.flush()
        for book, extra in (("DemoBook", 1.0), ("DemoBook2", 1.0)):
            _add_snapshot(db, m, book, _price(mispriced, MARGIN), now - timedelta(minutes=10))
        counts["upcoming"] += 1
    db.commit()
    return counts
