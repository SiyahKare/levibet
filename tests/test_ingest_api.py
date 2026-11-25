import io

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.db.models import Match, OddsSnapshot
from app.services.odds.ingest import (
    import_football_data_text, ingest_odds_events, normalize_team, parse_sport_setting,
)

CSV = """Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A,PSCH,PSCD,PSCA
16/08/2024,20:00,Man United,Fulham,1,0,1.60,4.2,5.5,1.55,4.3,6.0
17/08/2024,12:30,Spurs,Everton,,,1.8,3.8,4.5,,,
bad-date,,X,Y,1,1,,,,,,
"""


def test_csv_import_idempotent(db):
    c1 = import_football_data_text(db, CSV, "EPL")
    assert c1["matches_new"] == 2 and c1["rows_skipped"] == 1 and c1["snapshots_new"] == 3
    c2 = import_football_data_text(db, CSV, "EPL")
    assert c2["matches_new"] == 0 and c2["snapshots_new"] == 0
    m = db.query(Match).filter_by(home_team="Manchester United").one()
    assert m.status == "finished" and (m.home_goals, m.away_goals) == (1, 0)
    assert db.query(Match).filter_by(home_team="Tottenham").one().status == "scheduled"
    assert db.query(OddsSnapshot).filter_by(is_closing=True).count() == 1


def test_csv_missing_columns(db):
    with pytest.raises(ValueError):
        import_football_data_text(db, "a,b\n1,2\n", "EPL")


def test_names_and_sport_setting():
    assert normalize_team("  Man   City ") == "Manchester City"
    assert parse_sport_setting("soccer_epl=Premier League, soccer_spain_la_liga") == [
        ("soccer_epl", "Premier League"), ("soccer_spain_la_liga", None)]


def test_odds_api_events(db):
    ev = [{"id": "e1", "sport_key": "soccer_epl", "home_team": "Arsenal", "away_team": "Chelsea",
           "commence_time": "2030-01-01T15:00:00Z",
           "bookmakers": [{"key": "pinnacle", "last_update": "2029-12-31T10:00:00Z",
                           "markets": [{"key": "h2h", "outcomes": [
                               {"name": "Arsenal", "price": 2.1}, {"name": "Draw", "price": 3.5},
                               {"name": "Chelsea", "price": 3.6}]}]}]},
          {"bad": "event"}]
    c = ingest_odds_events(db, ev, "Premier League")
    assert c == {"matches_new": 1, "snapshots_new": 1, "events_skipped": 1}
    assert ingest_odds_events(db, ev, "Premier League")["snapshots_new"] == 0


def test_api(db):
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        for path in ("/matches/upcoming", "/predictions/latest", "/signals/latest", "/performance"):
            assert client.get(path).status_code == 200
        assert client.post("/admin/train-bayes/run").status_code == 401
        assert client.post("/admin/train-bayes/run", headers={"x-admin-token": "wrong"}).status_code == 401
