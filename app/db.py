from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# SQLite database (local file)
DATABASE_URL = "sqlite:///./car_damage.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 🔑 Add init_db function here
def init_db():
    # Import models inside function to avoid circular imports
    from app import models
    Base.metadata.create_all(bind=engine)
