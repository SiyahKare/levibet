from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = "postgresql://sports:sports@localhost:5432/sports"

    # Telegram delivery. Nothing is ever sent unless SIGNALS_DRY_RUN=false AND both are set.
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHANNEL_ID: str = ""

    # Odds provider (https://the-odds-api.com). Optional.
    ODDS_API_KEY: str = ""
    ODDS_API_SPORTS: str = "soccer_epl=Premier League"  # comma separated sport keys
    ODDS_API_REGIONS: str = "eu"

    # Protects the /admin endpoints. Leave empty to disable them.
    ADMIN_TOKEN: str = ""

    # Signal policy
    SIGNALS_DRY_RUN: bool = True
    SIGNAL_LANG: str = "tr"  # "tr" or "en"
    MIN_EDGE: float = 0.03  # minimum conservative edge (see engine.py)
    UNCERTAINTY_K: float = 1.0  # how many posterior std-devs to subtract from p
    MAX_EDGE: float = 0.25  # edges above this are treated as data/model errors
    MIN_ODDS: float = 1.30
    MAX_ODDS: float = 6.00
    MAX_ODDS_AGE_MINUTES: int = 120
    MIN_MINUTES_TO_KICKOFF: int = 30
    PREDICTION_HORIZON_HOURS: int = 72
    MAX_SIGNALS_PER_RUN: int = 5

    # Training / worker
    MIN_TRAINING_MATCHES: int = 40
    TRAIN_DRAWS: int = 500
    TRAIN_TUNE: int = 500
    WORKER_INTERVAL_SECONDS: int = 300


settings = Settings()
