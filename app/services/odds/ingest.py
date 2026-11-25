"""Bring matches, results and odds into the database.

Two sources are supported:
  * football-data.co.uk style CSV files (history, results and odds), and
  * The Odds API (live upcoming odds).

Team names differ between providers. `normalize_team` maps the usual suspects to
one canonical name; extend TEAM_ALIASES for the leagues you cover.
"""

import io
import logging
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.time import as_utc, utcnow
from app.db.models import Match, OddsSnapshot

log = logging.getLogger(__name__)

TEAM_ALIASES = {
    "man united": "Manchester United",
    "man utd": "Manchester United",
    "man city": "Manchester City",
    "spurs": "Tottenham",
    "tottenham hotspur": "Tottenham",
    "newcastle utd": "Newcastle",
    "newcastle united": "Newcastle",
    "wolves": "Wolverhampton",
    "wolverhampton wanderers": "Wolverhampton",
    "nott'm forest": "Nottingham Forest",
    "nottm forest": "Nottingham Forest",
    "west ham united": "West Ham",
    "brighton and hove albion": "Brighton",
    "brighton & hove albion": "Brighton",
    "leeds united": "Leeds",
    "leicester city": "Leicester",
    "sheffield utd": "Sheffield United",
    "luton town": "Luton",
    "norwich city": "Norwich",
}

# (bookmaker, home col, draw col, away col)
OPENING_SETS = [
    ("B365", "B365H", "B365D", "B365A"),
    ("PS", "PSH", "PSD", "PSA"),
    ("AVG", "AvgH", "AvgD", "AvgA"),
]
CLOSING_SETS = [
    ("PS", "PSCH", "PSCD", "PSCA"),
    ("B365", "B365CH", "B365CD", "B365CA"),
    ("AVG", "AvgCH", "AvgCD", "AvgCA"),
]


def normalize_team(name: str) -> str:
    clean = " ".join(str(name).strip().split())
    return TEAM_ALIASES.get(clean.lower(), clean)


def find_match(db: Session, home: str, away: str, kickoff: datetime) -> Match | None:
    """Same teams within 36 hours of `kickoff`, whatever provider created the row."""
    kickoff = as_utc(kickoff)
    return (
        db.query(Match)
        .filter(
            Match.home_team == home,
            Match.away_team == away,
            Match.kickoff_at >= kickoff - timedelta(hours=36),
            Match.kickoff_at <= kickoff + timedelta(hours=36),
        )
        .first()
    )


def _add_snapshot(
    db: Session,
    match: Match,
    bookmaker: str,
    odds: tuple[float, float, float],
    captured_at: datetime,
    is_closing: bool = False,
) -> bool:
    if any(o is None or pd.isna(o) or o <= 1.0 for o in odds):
        return False
    captured_at = as_utc(captured_at)
    exists = (
        db.query(OddsSnapshot.id)
        .filter(
            OddsSnapshot.match_id == match.id,
            OddsSnapshot.bookmaker == bookmaker,
            OddsSnapshot.captured_at == captured_at,
            OddsSnapshot.is_closing == is_closing,
        )
        .first()
    )
    if exists:
        return False
    db.add(
        OddsSnapshot(
            match_id=match.id,
            bookmaker=bookmaker,
            home_odds=float(odds[0]),
            draw_odds=float(odds[1]),
            away_odds=float(odds[2]),
            captured_at=captured_at,
            is_closing=is_closing,
        )
    )
    return True


def record_result(db: Session, match: Match, home_goals: int, away_goals: int) -> None:
    match.home_goals = int(home_goals)
    match.away_goals = int(away_goals)
    match.status = "finished"


# --------------------------------------------------------------------------- CSV


def import_football_data_csv(
    db: Session, source, league: str, season: str | None = None
) -> dict:
    """Import a football-data.co.uk style CSV (path, file object or text buffer).

    Needs Date, HomeTeam, AwayTeam. FTHG/FTAG make the match finished, B365H/D/A
    (or PS/Avg columns) become the opening odds and PSC*/B365C* the closing odds.
    The import is idempotent: running it twice does not duplicate anything.
    """
    df = pd.read_csv(source)
    missing = {"Date", "HomeTeam", "AwayTeam"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    counts = {"matches_new": 0, "matches_updated": 0, "snapshots_new": 0, "rows_skipped": 0}
    for _, row in df.iterrows():
        try:
            day = pd.to_datetime(row["Date"], dayfirst=True)
            clock = str(row["Time"]) if "Time" in df.columns and pd.notna(row.get("Time")) else "15:00"
            hh, mm = clock.split(":")[:2]
            kickoff = datetime(day.year, day.month, day.day, int(hh), int(mm), tzinfo=timezone.utc)
        except Exception:
            counts["rows_skipped"] += 1
            continue

        home = normalize_team(row["HomeTeam"])
        away = normalize_team(row["AwayTeam"])
        has_result = (
            "FTHG" in df.columns
            and pd.notna(row.get("FTHG"))
            and pd.notna(row.get("FTAG"))
        )

        match = find_match(db, home, away, kickoff)
        if match is None:
            match = Match(
                external_match_id=f"fd:{league}:{kickoff:%Y%m%d}:{home}:{away}",
                sport="football",
                league=league,
                season=season,
                home_team=home,
                away_team=away,
                kickoff_at=kickoff,
                status="scheduled",
            )
            db.add(match)
            db.flush()
            counts["matches_new"] += 1
        elif has_result and match.status != "finished":
            counts["matches_updated"] += 1
        if has_result:
            record_result(db, match, row["FTHG"], row["FTAG"])

        for book, h, d, a in OPENING_SETS:
            if {h, d, a} <= set(df.columns) and pd.notna(row[h]):
                if _add_snapshot(db, match, book, (row[h], row[d], row[a]), kickoff - timedelta(hours=24)):
                    counts["snapshots_new"] += 1
                break
        for book, h, d, a in CLOSING_SETS:
            if {h, d, a} <= set(df.columns) and pd.notna(row[h]):
                if _add_snapshot(
                    db, match, book, (row[h], row[d], row[a]), kickoff - timedelta(minutes=5), True
                ):
                    counts["snapshots_new"] += 1
                break
    db.commit()
    return counts


def import_football_data_text(db: Session, text: str, league: str, season: str | None = None) -> dict:
    return import_football_data_csv(db, io.StringIO(text), league, season)


# ---------------------------------------------------------------- The Odds API

API_BASE = "https://api.the-odds-api.com/v4"


def _parse_iso(value: str) -> datetime:
    return as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def parse_sport_setting(value: str) -> list[tuple[str, str | None]]:
    """'soccer_epl=Premier League,soccer_spain_la_liga' -> [(key, league or None), ...]"""
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        key, _, league = part.partition("=")
        out.append((key.strip(), league.strip() or None))
    return out


def fetch_odds(
    sport_key: str,
    api_key: str,
    regions: str = "eu",
    client: httpx.Client | None = None,
) -> tuple[list[dict], dict]:
    own = client is None
    client = client or httpx.Client(timeout=20)
    try:
        resp = client.get(
            f"{API_BASE}/sports/{sport_key}/odds/",
            params={
                "apiKey": api_key,
                "regions": regions,
                "markets": "h2h",
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
        )
        resp.raise_for_status()
        usage = {
            "remaining": resp.headers.get("x-requests-remaining"),
            "used": resp.headers.get("x-requests-used"),
        }
        return resp.json(), usage
    finally:
        if own:
            client.close()


def ingest_odds_events(db: Session, events: list[dict], league: str | None = None) -> dict:
    counts = {"matches_new": 0, "snapshots_new": 0, "events_skipped": 0}
    for ev in events:
        try:
            home = normalize_team(ev["home_team"])
            away = normalize_team(ev["away_team"])
            kickoff = _parse_iso(ev["commence_time"])
        except Exception:
            counts["events_skipped"] += 1
            continue

        match = find_match(db, home, away, kickoff)
        if match is None:
            match = Match(
                external_match_id=f"oddsapi:{ev.get('id')}",
                sport="football",
                league=league or ev.get("sport_title") or ev.get("sport_key", "unknown"),
                home_team=home,
                away_team=away,
                kickoff_at=kickoff,
                status="scheduled",
            )
            db.add(match)
            db.flush()
            counts["matches_new"] += 1

        for bm in ev.get("bookmakers", []):
            market = next((m for m in bm.get("markets", []) if m.get("key") == "h2h"), None)
            if not market:
                continue
            prices = {o["name"]: o["price"] for o in market.get("outcomes", [])}
            h = prices.get(ev["home_team"])
            d = prices.get("Draw")
            a = prices.get(ev["away_team"])
            if h is None or d is None or a is None:
                continue
            captured = _parse_iso(bm["last_update"]) if bm.get("last_update") else utcnow()
            if _add_snapshot(db, match, bm.get("key", "unknown"), (h, d, a), captured):
                counts["snapshots_new"] += 1
    db.commit()
    return counts


def ingest_the_odds_api(db: Session, client: httpx.Client | None = None) -> dict:
    if not settings.ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY is not set")
    total = {"matches_new": 0, "snapshots_new": 0, "events_skipped": 0}
    for sport_key, league in parse_sport_setting(settings.ODDS_API_SPORTS):
        events, usage = fetch_odds(sport_key, settings.ODDS_API_KEY, settings.ODDS_API_REGIONS, client)
        counts = ingest_odds_events(db, events, league)
        for k, v in counts.items():
            total[k] += v
        log.info("odds api %s: %s (requests remaining: %s)", sport_key, counts, usage["remaining"])
    return total
