# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import DATABASE

DATABASE_URL = (
    f"{DATABASE['DB_DRIVER']}://{DATABASE['DB_USER']}:{DATABASE['DB_PASSWORD']}"
    f"@{DATABASE['DB_HOST']}:{DATABASE['DB_PORT']}/{DATABASE['DB_NAME']}"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)