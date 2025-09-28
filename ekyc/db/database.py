from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from utils.config import Config
from appcontext import ctx

config = Config()
SQLALCHEMY_DATABASE_URL = config.SQLALCHEMY_DATABASE_URL

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_postgres_db():
    """
    Get a new database session for PostgreSQL.
    """
    if not ctx.db_session_factory:
        raise RuntimeError("Database session factory is not initialized.")
    db = ctx.get_db_session()
    try:
        yield db
    finally:
        db.close()