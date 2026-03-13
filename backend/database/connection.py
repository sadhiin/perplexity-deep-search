import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_SQLITE_PATH = os.path.join(BASE_DIR, "data", "deep_research.db")

DATABASE_URL = os.environ.get("DATABASE_URL") or f"sqlite:///{DEFAULT_SQLITE_PATH}"

# Ensure the directory exists for SQLite files
if DATABASE_URL.startswith("sqlite:///"):
    os.makedirs(os.path.dirname(DEFAULT_SQLITE_PATH), exist_ok=True)

ENGINE = create_engine(
    DATABASE_URL,
    future=True,
    echo=False,
    connect_args={"check_same_thread": False}
    if DATABASE_URL.startswith("sqlite")
    else {},
)

SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, expire_on_commit=False, future=True)


class Base(DeclarativeBase):
    """Base class for declarative models."""


def init_db() -> None:
    """Create database tables for all registered models."""
    Base.metadata.create_all(bind=ENGINE)


@contextmanager
def get_db_session():
    """Yield a database session for use with context managers."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
