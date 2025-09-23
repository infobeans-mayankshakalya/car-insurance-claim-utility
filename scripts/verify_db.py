from sqlalchemy import inspect
from app.db import engine, SessionLocal
from app import models

def list_tables():
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print("✅ Tables in the database:", tables)

def check_sample_insert():
    db = SessionLocal()
    try:
        # Insert a sample car
        car = models.Car(make="Honda", model="Civic", year=2020)
        db.add(car)
        db.commit()
        db.refresh(car)
        print(f"✅ Inserted Car: id={car.id}, make={car.make}, model={car.model}, year={car.year}")
    finally:
        db.close()

if __name__ == "__main__":
    list_tables()
    check_sample_insert()
