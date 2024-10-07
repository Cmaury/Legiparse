from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from models import Base  # Replace with your model file
import database

# Database connection string
DATABASE_URL = database.DATABASE_URL

# Step 1: Connect to the database
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

# Step 2: Drop all tables
meta = MetaData()
meta.reflect(bind=engine)
meta.drop_all(bind=engine)

# Step 3: Recreate tables from the model
Base.metadata.create_all(engine)

# Close session
session.close()