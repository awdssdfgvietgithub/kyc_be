from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from . import config
from ..common.test_model_v2 import logger

DB_USERNAME = config.DB_USERNAME
DB_PASSWORD = config.DB_PASSWORD
DB_HOST = config.DB_HOST
DB_NAME = config.DB_NAME

SQLALCHEMY_DB_URL = f"postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

engine = create_engine(SQLALCHEMY_DB_URL, echo=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    try:
        db = SessionLocal()
        logger.info(f"Connected to DB {DB_HOST}/{DB_NAME}")
        yield db
    except Exception as e:
        logger.error(f"Error connecting to DB: {str(e)}")
        raise e
    finally:
        db.close()