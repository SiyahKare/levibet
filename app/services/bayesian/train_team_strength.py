"""Hierarchical Bayesian Poisson model of team attack/defence strength.

goals_home ~ Poisson(exp(intercept + home_adv + attack[home] - defence[away]))
goals_away ~ Poisson(exp(intercept + attack[away] - defence[home]))

Attack and defence are zero-sum constrained per league so the intercept stays
identifiable. PyMC is imported lazily: the API and worker start without it.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.time import utcnow
from app.db.models import BayesianTeamStrength, Match
from app.db.session import SessionLocal

log = logging.getLogger(__name__)

MODEL_VERSION = "pymc_poisson_v2"


def load_historical_data(db: Session, league: str | None = None) -> pd.DataFrame:
    """Finished matches with a recorded score, oldest first."""
    q = db.query(Match).filter(
        Match.status == "finished",
        Match.home_goals.isnot(None),
        Match.away_goals.isnot(None),
    )
    if league:
        q = q.filter(Match.league == league)
    rows = q.order_by(Match.kickoff_at).all()
    return pd.DataFrame(
        [
            {
                "league": m.league,
                "home_team": m.home_team,
                "away_team": m.away_team,
                "home_goals": m.home_goals,
                "away_goals": m.away_goals,
                "kickoff_at": m.kickoff_at,
            }
            for m in rows
        ],
        columns=["league", "home_team", "away_team", "home_goals", "away_goals", "kickoff_at"],
    )


def train_poisson_model(
    df: pd.DataFrame,
    draws: int | None = None,
    tune: int | None = None,
    seed: int = 42,
):
    import pymc as pm

    draws = draws or settings.TRAIN_DRAWS
    tune = tune or settings.TRAIN_TUNE

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    index = {t: i for i, t in enumerate(teams)}
    home_idx = df["home_team"].map(index).to_numpy()
    away_idx = df["away_team"].map(index).to_numpy()

    with pm.Model():
        intercept = pm.Normal("intercept", mu=0.2, sigma=0.5)
        home_adv = pm.Normal("home_adv", mu=0.2, sigma=0.3)
        sd_att = pm.HalfNormal("sd_att", sigma=0.5)
        sd_def = pm.HalfNormal("sd_def", sigma=0.5)
        attack = pm.ZeroSumNormal("attack", sigma=sd_att, shape=len(teams))
        defense = pm.ZeroSumNormal("defense", sigma=sd_def, shape=len(teams))

        home_theta = pm.math.exp(intercept + home_adv + attack[home_idx] - defense[away_idx])
        away_theta = pm.math.exp(intercept + attack[away_idx] - defense[home_idx])

        pm.Poisson("home_goals", mu=home_theta, observed=df["home_goals"].to_numpy())
        pm.Poisson("away_goals", mu=away_theta, observed=df["away_goals"].to_numpy())

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=2,
            cores=1,
            random_seed=seed,
            progressbar=False,
        )
    return idata, teams


def convergence_report(idata) -> dict:
    import arviz as az

    summary = az.summary(idata, var_names=["intercept", "home_adv", "attack", "defense"])
    diverging = 0
    if "diverging" in idata.sample_stats:
        diverging = int(idata.sample_stats["diverging"].values.sum())
    return {
        "max_r_hat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "divergences": diverging,
    }


def export_posteriors(
    idata,
    teams: list[str],
    league: str,
    db: Session,
    model_version: str = MODEL_VERSION,
    as_of: datetime | None = None,
) -> int:
    n = len(teams)
    att = idata.posterior["attack"].values.reshape(-1, n)
    dfn = idata.posterior["defense"].values.reshape(-1, n)
    ha = idata.posterior["home_adv"].values.flatten()
    ic = idata.posterior["intercept"].values.flatten()
    as_of = as_of or utcnow()

    for i, team in enumerate(teams):
        db.add(
            BayesianTeamStrength(
                team_name=team,
                league=league,
                as_of=as_of,
                model_version=model_version,
                attack_mean=float(att[:, i].mean()),
                attack_std=float(att[:, i].std()),
                defense_mean=float(dfn[:, i].mean()),
                defense_std=float(dfn[:, i].std()),
                intercept_mean=float(ic.mean()),
                intercept_std=float(ic.std()),
                home_adv_mean=float(ha.mean()),
                home_adv_std=float(ha.std()),
                sample_size=int(att.shape[0]),
            )
        )
    db.commit()
    return n


def train_all(
    db: Session | None = None,
    league: str | None = None,
    min_matches: int | None = None,
    draws: int | None = None,
    tune: int | None = None,
) -> list[dict]:
    """Train one model per league that has enough finished matches."""
    own = db is None
    db = db or SessionLocal()
    min_matches = min_matches or settings.MIN_TRAINING_MATCHES
    try:
        data = load_historical_data(db, league)
        results = []
        for lg, part in data.groupby("league"):
            if len(part) < min_matches:
                log.warning("skip %s: %d matches, need %d", lg, len(part), min_matches)
                results.append({"league": lg, "matches": len(part), "trained": False})
                continue
            idata, teams = train_poisson_model(part.reset_index(drop=True), draws, tune)
            export_posteriors(idata, teams, lg, db)
            report = convergence_report(idata)
            if report["max_r_hat"] > 1.05 or report["divergences"] > 0:
                log.warning("%s: weak convergence %s", lg, report)
            results.append(
                {"league": lg, "matches": len(part), "teams": len(teams), "trained": True, **report}
            )
        return results
    finally:
        if own:
            db.close()
