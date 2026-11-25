from datetime import timedelta

import pytest

from app.core.config import settings
from app.core.time import utcnow
from app.db.models import BayesianTeamStrength, Match, OddsSnapshot, Prediction, Signal
from app.services.odds.ingest import _add_snapshot, record_result
from app.services.predictions.generate_predictions import generate_predictions
from app.services.signals.engine import generate_signals
from app.services.signals.messages import format_signal
from app.services.signals.performance import performance
from app.services.signals.settlement import settle_signals
from app.services.telegram.sender import dispatch_signals


def _match(db, hours=10, league="Test League"):
    m = Match(external_match_id=f"t{db.query(Match).count()}", league=league, home_team="A",
              away_team="B", kickoff_at=utcnow() + timedelta(hours=hours), status="scheduled")
    db.add(m)
    db.flush()
    return m


def _pred(db, m, probs=(0.55, 0.25, 0.20), std=0.01):
    p = Prediction(match_id=m.id, model_version="t", as_of=utcnow(), home_win_prob=probs[0],
                   draw_prob=probs[1], away_win_prob=probs[2],
                   extra={"std": {"H": std, "D": std, "A": std}})
    db.add(p)
    db.flush()
    return p


def test_signal_created_for_value(db):
    m = _match(db)
    _pred(db, m)
    _add_snapshot(db, m, "b1", (2.20, 3.6, 4.2), utcnow() - timedelta(minutes=5))
    db.commit()
    sigs = generate_signals(db)
    assert len(sigs) == 1 and sigs[0].selection == "H"
    assert sigs[0].edge == pytest.approx(0.55 * 2.20 - 1)
    assert sigs[0].conservative_edge < sigs[0].edge
    assert generate_signals(db) == []  # deduplicated


def test_no_signal_without_value_or_with_stale_odds(db):
    m = _match(db)
    _pred(db, m)
    _add_snapshot(db, m, "b1", (1.60, 3.6, 5.0), utcnow() - timedelta(minutes=5))
    m2 = _match(db)
    _pred(db, m2)
    _add_snapshot(db, m2, "b1", (2.20, 3.6, 4.2), utcnow() - timedelta(hours=10))
    db.commit()
    assert generate_signals(db) == []


def test_absurd_edge_is_dropped_and_uncertainty_matters(db):
    m = _match(db)
    _pred(db, m, probs=(0.80, 0.12, 0.08))
    _add_snapshot(db, m, "b1", (2.50, 4.0, 6.0), utcnow() - timedelta(minutes=5))  # edge 100%
    m2 = _match(db)
    _pred(db, m2, std=0.12)  # edge 21% but very uncertain -> conservative edge < MIN_EDGE
    _add_snapshot(db, m2, "b1", (2.20, 3.6, 4.2), utcnow() - timedelta(minutes=5))
    db.commit()
    assert generate_signals(db) == []


def test_too_close_to_kickoff(db):
    m = _match(db, hours=0.2)
    _pred(db, m)
    _add_snapshot(db, m, "b1", (2.20, 3.6, 4.2), utcnow() - timedelta(minutes=5))
    db.commit()
    assert generate_signals(db) == []


def test_max_signals_per_run(db, monkeypatch):
    monkeypatch.setattr(settings, "MAX_SIGNALS_PER_RUN", 2)
    for _ in range(4):
        m = _match(db)
        _pred(db, m)
        _add_snapshot(db, m, "b1", (2.20, 3.6, 4.2), utcnow() - timedelta(minutes=5))
    db.commit()
    assert len(generate_signals(db)) == 2


def test_dry_run_dispatch_and_demo_never_sent(db, monkeypatch):
    m = _match(db, league="Demo League")
    _pred(db, m)
    _add_snapshot(db, m, "b1", (2.20, 3.6, 4.2), utcnow() - timedelta(minutes=5))
    db.commit()
    generate_signals(db)
    monkeypatch.setattr(settings, "SIGNALS_DRY_RUN", False)
    monkeypatch.setattr("app.services.telegram.sender.send_telegram_message",
                        lambda *a, **k: pytest.fail("demo signal must not be sent"))
    assert dispatch_signals(db) == {"dry_run": 1, "sent": 0, "failed": 0}


def test_live_dispatch_and_failure(db, monkeypatch):
    m = _match(db)
    _pred(db, m)
    _add_snapshot(db, m, "b1", (2.20, 3.6, 4.2), utcnow() - timedelta(minutes=5))
    db.commit()
    generate_signals(db)
    monkeypatch.setattr(settings, "SIGNALS_DRY_RUN", False)

    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr("app.services.telegram.sender.send_telegram_message", boom)
    assert dispatch_signals(db)["failed"] == 1
    sig = db.query(Signal).one()
    sig.status = "pending"
    db.commit()
    monkeypatch.setattr("app.services.telegram.sender.send_telegram_message", lambda *a, **k: "42")
    assert dispatch_signals(db)["sent"] == 1
    assert db.query(Signal).one().telegram_message_id == "42"


def test_message_has_disclaimer_and_no_promises(db):
    m = _match(db)
    p = _pred(db, m)
    _add_snapshot(db, m, "b1", (2.20, 3.6, 4.2), utcnow() - timedelta(minutes=5))
    db.commit()
    sig = generate_signals(db)[0]
    for lang in ("tr", "en"):
        text = format_signal(sig, m, lang)
        assert "18" in text and "A - B" in text
    assert "garanti" not in format_signal(sig, m, "tr").split("Geçmiş")[0].lower()


def test_settlement_pnl_clv_performance(db):
    m = _match(db)
    p = _pred(db, m)
    _add_snapshot(db, m, "b1", (2.20, 3.6, 4.2), utcnow() - timedelta(minutes=5))
    _add_snapshot(db, m, "b1", (2.00, 3.6, 4.8), utcnow() - timedelta(minutes=1), True)
    db.commit()
    generate_signals(db)
    assert settle_signals(db) == 0  # not finished yet
    record_result(db, m, 2, 1)
    db.commit()
    assert settle_signals(db) == 1
    s = db.query(Signal).one()
    assert s.outcome == "win" and s.pnl == pytest.approx(1.2)
    assert s.closing_odds == 2.00 and s.clv is not None and s.clv > 0
    perf = performance(db)
    assert perf["graded"] == 1 and perf["hit_rate"] == 1.0 and perf["roi"] == pytest.approx(1.2)


def test_loss_and_void(db):
    for goals, status, outcome, pnl in ((0, "finished", "loss", -1.0), (0, "void", "void", 0.0)):
        m = _match(db)
        _pred(db, m)
        _add_snapshot(db, m, "b1", (2.20, 3.6, 4.2), utcnow() - timedelta(minutes=5))
        db.commit()
        generate_signals(db)
        record_result(db, m, goals, 1)
        m.status = status
        db.commit()
    settle_signals(db)
    got = sorted((s.outcome, s.pnl) for s in db.query(Signal).all())
    assert got == [("loss", -1.0), ("void", 0.0)]


def test_predictions_from_strengths(db):
    now = utcnow()
    for team, att, dfn in (("A", 0.4, 0.2), ("B", -0.4, -0.2)):
        db.add(BayesianTeamStrength(team_name=team, league="Test League", as_of=now, model_version="t",
                                    attack_mean=att, attack_std=0.1, defense_mean=dfn, defense_std=0.1,
                                    intercept_mean=0.25, intercept_std=0.05,
                                    home_adv_mean=0.25, home_adv_std=0.05))
    _match(db)
    db.commit()
    preds = generate_predictions(db)
    assert len(preds) == 1
    p = preds[0]
    assert p.home_win_prob + p.draw_prob + p.away_win_prob == pytest.approx(1.0)
    assert p.home_win_prob > p.away_win_prob and 0 <= p.confidence_score <= 1
    assert generate_predictions(db) == []  # idempotent until model changes
