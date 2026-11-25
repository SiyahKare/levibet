from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.sql import func

from app.db.session import Base

# BigInteger primary keys do not autoincrement on SQLite (used in tests).
PK = BigInteger().with_variant(Integer, "sqlite")


class Match(Base):
    __tablename__ = "matches"

    id = Column(PK, primary_key=True, index=True)
    external_match_id = Column(String, unique=True, nullable=False)
    sport = Column(String, nullable=False, default="football")
    league = Column(String, nullable=False, index=True)
    season = Column(String)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    kickoff_at = Column(DateTime(timezone=True), nullable=False, index=True)
    # scheduled | finished | void
    status = Column(String, nullable=False, default="scheduled", index=True)
    home_goals = Column(Integer)
    away_goals = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class OddsSnapshot(Base):
    """Decimal 1X2 odds from one bookmaker at one point in time."""

    __tablename__ = "odds_snapshots"

    id = Column(PK, primary_key=True, index=True)
    match_id = Column(BigInteger, ForeignKey("matches.id"), nullable=False, index=True)
    bookmaker = Column(String, nullable=False)
    market = Column(String, nullable=False, default="1X2")
    home_odds = Column(Float, nullable=False)
    draw_odds = Column(Float, nullable=False)
    away_odds = Column(Float, nullable=False)
    captured_at = Column(DateTime(timezone=True), nullable=False, index=True)
    is_closing = Column(Boolean, nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint(
            "match_id", "bookmaker", "captured_at", "is_closing", name="uq_odds_snapshot"
        ),
    )


class BayesianTeamStrength(Base):
    __tablename__ = "bayesian_team_strength"

    id = Column(PK, primary_key=True, index=True)
    team_name = Column(String, nullable=False, index=True)
    league = Column(String, nullable=False, index=True)
    as_of = Column(DateTime(timezone=True), nullable=False, index=True)
    model_version = Column(String, nullable=False)
    attack_mean = Column(Float, nullable=False)
    attack_std = Column(Float, nullable=False)
    defense_mean = Column(Float, nullable=False)
    defense_std = Column(Float, nullable=False)
    # League-wide parameters, repeated on every row of the same training run.
    intercept_mean = Column(Float)
    intercept_std = Column(Float)
    home_adv_mean = Column(Float)
    home_adv_std = Column(Float)
    sample_size = Column(BigInteger)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class FeatureSnapshot(Base):
    """Kept for compatibility. The current pipeline does not need it."""

    __tablename__ = "feature_snapshots"

    id = Column(PK, primary_key=True, index=True)
    match_id = Column(BigInteger, nullable=False, index=True)
    feature_set_version = Column(String, nullable=False)
    as_of = Column(DateTime(timezone=True), nullable=False, index=True)
    features = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(PK, primary_key=True, index=True)
    match_id = Column(BigInteger, ForeignKey("matches.id"), nullable=False, index=True)
    model_version = Column(String, nullable=False)
    as_of = Column(DateTime(timezone=True), nullable=False, index=True)
    home_win_prob = Column(Float, nullable=False)
    draw_prob = Column(Float, nullable=True)
    away_win_prob = Column(Float, nullable=False)
    expected_home_goals = Column(Float)
    expected_away_goals = Column(Float)
    # 1 = parameters well pinned down, 0 = very uncertain (see generate_predictions.py)
    confidence_score = Column(Float)
    # Best conservative edge found against market odds at the last signal run.
    edge_score = Column(Float)
    # {"std": {"H":..,"D":..,"A":..}, "over_2_5":.., "btts":..}
    extra = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Signal(Base):
    __tablename__ = "signals"

    id = Column(PK, primary_key=True, index=True)
    prediction_id = Column(BigInteger, ForeignKey("predictions.id"), nullable=False)
    match_id = Column(BigInteger, ForeignKey("matches.id"), nullable=False, index=True)
    model_version = Column(String, nullable=False)
    market = Column(String, nullable=False, default="1X2")
    selection = Column(String, nullable=False)  # H | D | A
    bookmaker = Column(String, nullable=False)
    odds = Column(Float, nullable=False)
    odds_captured_at = Column(DateTime(timezone=True))
    model_prob = Column(Float, nullable=False)
    prob_std = Column(Float, nullable=False, default=0.0)
    market_prob = Column(Float, nullable=False)
    edge = Column(Float, nullable=False)
    conservative_edge = Column(Float, nullable=False)

    channel_name = Column(String)
    telegram_message_id = Column(String)
    dispatched_at = Column(DateTime(timezone=True))
    # pending | dry_run | sent | failed
    status = Column(String, nullable=False, default="pending", index=True)

    # Filled in by settlement
    outcome = Column(String)  # win | loss | void
    pnl = Column(Float)  # flat 1-unit stake
    closing_odds = Column(Float)
    clv = Column(Float)
    settled_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("match_id", "market", name="uq_signal_match_market"),
        Index("ix_signals_outcome", "outcome"),
    )
