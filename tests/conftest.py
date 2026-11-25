import os

os.environ["DATABASE_URL"] = "sqlite://"
os.environ["SIGNALS_DRY_RUN"] = "true"
os.environ["TELEGRAM_BOT_TOKEN"] = ""
os.environ["TELEGRAM_CHANNEL_ID"] = ""
os.environ["ADMIN_TOKEN"] = "secret"

import pytest

from app.db.session import Base, SessionLocal, engine


@pytest.fixture()
def db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
