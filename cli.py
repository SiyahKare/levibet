import json
import logging

import typer

from app.db.session import Base, SessionLocal, engine

app = typer.Typer(help="LeviBet sports analytics CLI (analytics only: no bets are placed)")


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")


def _echo(obj):
    typer.echo(json.dumps(obj, indent=2, default=str))


@app.command("init-db")
def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
    typer.echo("Tables created.")


@app.command("seed-demo")
def seed_demo_cmd():
    """Load the synthetic Demo League (results, odds, upcoming fixtures)."""
    from app.services.demo import seed_demo

    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        _echo(seed_demo(db))


@app.command("import-csv")
def import_csv(path: str, league: str, season: str = typer.Option(None)):
    """Import a football-data.co.uk style CSV (results + odds)."""
    from app.services.odds.ingest import import_football_data_csv

    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        _echo(import_football_data_csv(db, path, league, season))


@app.command("ingest-odds")
def ingest_odds():
    """Fetch upcoming odds from The Odds API (needs ODDS_API_KEY)."""
    from app.services.odds.ingest import ingest_the_odds_api

    with SessionLocal() as db:
        _echo(ingest_the_odds_api(db))


@app.command("train-bayes")
def train_bayes(league: str = typer.Option(None)):
    """Train the Bayesian model per league and store the posteriors."""
    from app.services.bayesian.train_team_strength import train_all

    _echo(train_all(league=league))


@app.command("predict")
def predict():
    """Generate predictions for upcoming matches."""
    from app.services.predictions.generate_predictions import generate_predictions

    typer.echo(f"{len(generate_predictions())} predictions generated.")


@app.command("signals")
def signals():
    """Find value signals and dispatch them (dry-run unless SIGNALS_DRY_RUN=false)."""
    from app.services.signals.engine import generate_signals
    from app.services.telegram.sender import dispatch_signals

    with SessionLocal() as db:
        created = generate_signals(db)
        typer.echo(f"{len(created)} new signals.")
        _echo(dispatch_signals(db))


@app.command("settle")
def settle():
    """Settle signals of finished matches."""
    from app.services.signals.settlement import settle_signals

    with SessionLocal() as db:
        typer.echo(f"{settle_signals(db)} signals settled.")


@app.command("performance")
def performance_cmd():
    """Print the track record of settled signals."""
    from app.services.signals.performance import performance

    with SessionLocal() as db:
        _echo(performance(db))


@app.command("run-api")
def run_api(port: int = 8000, host: str = "0.0.0.0"):
    import uvicorn

    uvicorn.run("app.api.main:app", host=host, port=port)


@app.command("run-worker")
def run_worker():
    from app.workers.predict_worker import run_prediction_loop

    try:
        run_prediction_loop()
    except KeyboardInterrupt:
        typer.echo("Worker stopped.")


if __name__ == "__main__":
    app()
