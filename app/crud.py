from sqlalchemy.orm import Session
from . import models

def create_user(db: Session, name: str, email: str):
    user = models.User(name=name, email=email)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def create_car(db: Session, user_id: str, make: str, model: str, year: int):
    car = models.Car(user_id=user_id, make=make, model=model, year=year)
    db.add(car)
    db.commit()
    db.refresh(car)
    return car

def create_claim(db: Session, user_id: str, car_id: str, description: str = None):
    claim = models.Claim(user_id=user_id, car_id=car_id, description=description)
    db.add(claim)
    db.commit()
    db.refresh(claim)
    return claim
