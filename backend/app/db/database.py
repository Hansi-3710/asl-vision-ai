"""
db/database.py
===============
SQLAlchemy engine + session factory. SQLite for local development (per the
tech stack spec); swap DATABASE_URL to a Postgres/MySQL URL for production
without touching any other file, since nothing else references SQLite
directly.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.core.config import get_settings

settings = get_settings()

# check_same_thread=False is required for SQLite when used with FastAPI's
# threaded request handling (each request may run in a different thread
# than the one that created the connection). Not needed/used for
# Postgres/MySQL URLs.
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency: yields a DB session, always closed after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Creates all tables that don't exist yet. Called once at app startup.
    For anything beyond this simple create-if-missing behavior (real
    migrations, schema changes over time), add Alembic -- intentionally
    kept out of scope here since this project has a single, stable table."""
    from app.db import models  # noqa: F401 (import registers models on Base.metadata)

    Base.metadata.create_all(bind=engine)
