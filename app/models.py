from sqlalchemy import Column, String, Integer, ForeignKey, Text, DateTime, Numeric
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from .db import Base

def gen_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=gen_uuid)
    name = Column(String)
    email = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    cars = relationship("Car", back_populates="owner")
    claims = relationship("Claim", back_populates="user")

class Car(Base):
    __tablename__ = "cars"
    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"))
    make = Column(String)
    model = Column(String)
    year = Column(Integer)

    owner = relationship("User", back_populates="cars")
    claims = relationship("Claim", back_populates="car")

class Claim(Base):
    __tablename__ = "claims"
    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"))
    car_id = Column(String, ForeignKey("cars.id"))
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="claims")
    car = relationship("Car", back_populates="claims")
    images = relationship("ClaimImage", back_populates="claim")
    results = relationship("InferenceResult", back_populates="claim")

class ClaimImage(Base):
    __tablename__ = "claim_images"
    id = Column(String, primary_key=True, default=gen_uuid)
    claim_id = Column(String, ForeignKey("claims.id"))
    path = Column(String)
    width = Column(Integer)
    height = Column(Integer)

    claim = relationship("Claim", back_populates="images")

class InferenceResult(Base):
    __tablename__ = "inference_results"
    id = Column(String, primary_key=True, default=gen_uuid)
    claim_id = Column(String, ForeignKey("claims.id"))
    result = Column(Text)  # JSON string
    cost_estimate = Column(Numeric)
    severity = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    claim = relationship("Claim", back_populates="results")
