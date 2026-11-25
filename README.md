# LeviBet – Sports Analytics & Signal Engine

A football analytics pipeline: Bayesian team-strength model → match probabilities → comparison with
de-margined bookmaker odds → **informational value signals** → (optional) Telegram channel, with
settlement and a transparent track record (ROI ± standard error, closing line value, Brier, log loss).

> **LeviBet does not place bets, does not take stakes, has no affiliate links and gives no stake sizing.**
> Signals are statistical model output, not betting advice, and promise no profit.

## Pipeline

```
football-data CSV / The Odds API ─► matches, results, odds_snapshots
        └─► train-bayes   (PyMC Poisson attack/defence, zero-sum + hierarchical sd, per league)
        └─► predict       (posterior draws → 1X2, over 2.5, BTTS, probability std, confidence)
        └─► signals       (devig power method, consensus market prob, edge vs conservative edge)
        └─► dispatch      (dry-run by default; Telegram plain text with disclaimer)
        └─► settle        (win/loss/void, flat 1-unit P&L, closing odds, CLV)
        └─► performance   (hit rate, ROI ± SE, avg CLV, Brier, log loss, calibration)
```

Signal rules: `edge = p·odds − 1`, `conservative edge = (p − k·σ)·odds − 1`. A signal needs
conservative edge ≥ `MIN_EDGE`; edges above `MAX_EDGE` are discarded as probable errors; odds must be
within `MIN_ODDS..MAX_ODDS`, fresh (`MAX_ODDS_AGE_MINUTES`) and the match at least
`MIN_MINUTES_TO_KICKOFF` away. One signal per match, at most `MAX_SIGNALS_PER_RUN` per run.

## Quick start (no data, no keys)

```bash
pip install -r requirements.txt -r requirements-dev.txt
export DATABASE_URL=sqlite:///./levibet.db
python cli.py seed-demo       # synthetic "Demo League" (never sent to Telegram)
python cli.py train-bayes
python cli.py predict
python cli.py signals         # dry-run: prints the message instead of sending
python cli.py performance
python cli.py run-api         # http://localhost:8000/docs
pytest
```

## Real data

```bash
python cli.py import-csv E0.csv "Premier League" --season 2024/25   # football-data.co.uk format
python cli.py train-bayes
# live odds: set ODDS_API_KEY (and ODDS_API_SPORTS="soccer_epl=Premier League")
python cli.py ingest-odds && python cli.py predict && python cli.py signals
```

Team names differ between providers; extend `TEAM_ALIASES` in `app/services/odds/ingest.py` for your leagues.

## Docker

```bash
cp .env.example .env
docker compose up --build        # api on :8000, worker loop, postgres
docker compose run --rm api python cli.py seed-demo
```

## API

`GET /health`, `/matches/upcoming`, `/predictions/latest`, `/signals/latest`, `/performance`;
`POST /admin/train-bayes/run` (header `X-Admin-Token`; disabled while `ADMIN_TOKEN` is empty).

## Going live with Telegram

Signals stay in **dry-run** until `SIGNALS_DRY_RUN=false` *and* `TELEGRAM_BOT_TOKEN` /
`TELEGRAM_CHANNEL_ID` are set. Demo-league signals are never sent.

## Limits you should know

- Independent-Poisson goals ignore correlation and low-score inflation; the model is a baseline, not an oracle.
- Beating closing odds consistently is hard. Judge the system by CLV and calibration over hundreds of
  signals, not by a handful of results. Backtest on historical data before trusting any signal.
- Posterior parameters are treated as independent when drawing predictions (approximation).
- Alembic migrations are not set up; `create_all` is MVP-only.

## Compliance (not legal advice)

Betting-related content is heavily regulated. In Türkiye, advertising or promoting betting/gambling
(including via social media and Telegram channels) is restricted under Law 7258 and related rules, and
illegal betting promotion carries penalties. App stores (Apple, Google Play) have their own gambling and
"real-money" policies, and tipster services are regulated in many countries. Before publishing signals
publicly or monetising them, get advice from a lawyer in the jurisdiction(s) you target. Keep the
disclaimer in every message, restrict to 18+, make no profit guarantees and add no affiliate links
unless counsel confirms it is lawful.
