"""Turn team-strength posteriors into 1X2 (and goals-market) probabilities.

Parameter uncertainty is propagated by drawing from the stored posterior means
and standard deviations (independence between parameters is assumed, which is an
approximation). The spread of the resulting probabilities is what the signal
engine later uses to stay conservative.
"""

import logging
from datetime import timedelta

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.time import as_utc, utcnow
from app.db.models import BayesianTeamStrength, Match, Prediction
from app.db.session import SessionLocal
from app.services.bayesian import poisson

log = logging.getLogger(__name__)

# Mean std of the outcome probabilities at which confidence_score reaches 0.
CONFIDENCE_STD_SCALE = 0.10


def latest_strengths(db: Session, league: str) -> dict[str, BayesianTeamStrength]:
    rows = (
        db.query(BayesianTeamStrength)
        .filter(BayesianTeamStrength.league == league)
        .order_by(BayesianTeamStrength.as_of.desc(), BayesianTeamStrength.id.desc())
        .all()
    )
    out: dict[str, BayesianTeamStrength] = {}
    for r in rows:
        out.setdefault(r.team_name, r)
    return out


def predict_from_strengths(
    home: BayesianTeamStrength,
    away: BayesianTeamStrength,
    n_draws: int = 1000,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)

    def draw(mean, std):
        return rng.normal(mean or 0.0, std or 0.0, n_draws)

    intercept = draw(home.intercept_mean, home.intercept_std)
    home_adv = draw(home.home_adv_mean, home.home_adv_std)
    att_h = draw(home.attack_mean, home.attack_std)
    def_h = draw(home.defense_mean, home.defense_std)
    att_a = draw(away.attack_mean, away.attack_std)
    def_a = draw(away.defense_mean, away.defense_std)

    lam_h = np.exp(intercept + home_adv + att_h - def_a)
    lam_a = np.exp(intercept + att_a - def_h)

    mat = poisson.score_matrix(lam_h, lam_a)
    probs = poisson.outcome_probs(mat)  # (n, 3)
    mean = probs.mean(axis=0)
    mean = mean / mean.sum()
    std = probs.std(axis=0)

    return {
        "probs": mean,  # H, D, A
        "std": std,
        "exp_home_goals": float(lam_h.mean()),
        "exp_away_goals": float(lam_a.mean()),
        "over_2_5": float(poisson.over_prob(mat).mean()),
        "btts": float(poisson.btts_prob(mat).mean()),
    }


def generate_predictions(db: Session | None = None, now=None) -> list[Prediction]:
    own = db is None
    db = db or SessionLocal()
    now = now or utcnow()
    try:
        horizon = now + timedelta(hours=settings.PREDICTION_HORIZON_HOURS)
        matches = (
            db.query(Match)
            .filter(Match.status == "scheduled", Match.kickoff_at > now, Match.kickoff_at <= horizon)
            .order_by(Match.kickoff_at)
            .all()
        )
        strengths_cache: dict[str, dict[str, BayesianTeamStrength]] = {}
        created: list[Prediction] = []

        for m in matches:
            strengths = strengths_cache.setdefault(m.league, latest_strengths(db, m.league))
            h, a = strengths.get(m.home_team), strengths.get(m.away_team)
            if h is None or a is None:
                log.info("no strengths for %s vs %s (%s), skipped", m.home_team, m.away_team, m.league)
                continue

            model_as_of = max(as_utc(h.as_of), as_utc(a.as_of))
            last = (
                db.query(Prediction)
                .filter(Prediction.match_id == m.id)
                .order_by(Prediction.as_of.desc())
                .first()
            )
            if last is not None and as_utc(last.as_of) >= model_as_of:
                continue  # model unchanged since the last prediction

            res = predict_from_strengths(h, a, seed=int(m.id))
            p, s = res["probs"], res["std"]
            confidence = 1.0 - float(np.clip(s.mean() / CONFIDENCE_STD_SCALE, 0.0, 1.0))
            pred = Prediction(
                match_id=m.id,
                model_version=h.model_version,
                as_of=now,
                home_win_prob=float(p[0]),
                draw_prob=float(p[1]),
                away_win_prob=float(p[2]),
                expected_home_goals=res["exp_home_goals"],
                expected_away_goals=res["exp_away_goals"],
                confidence_score=confidence,
                extra={
                    "std": {"H": float(s[0]), "D": float(s[1]), "A": float(s[2])},
                    "over_2_5": res["over_2_5"],
                    "btts": res["btts"],
                },
            )
            db.add(pred)
            created.append(pred)
        db.commit()
        return created
    finally:
        if own:
            db.close()


if __name__ == "__main__":
    print(f"{len(generate_predictions())} predictions generated")
