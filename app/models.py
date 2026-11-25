
from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, String, ForeignKey,
    DateTime, Float, Text
)

from app.database import Base


# 1. User accounts
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="patient")
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc)
    )


# 2. Patient profiles
class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)


# 3. Doctor-patient assignments
class DoctorPatient(Base):
    __tablename__ = "doctor_patient"

    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)


# 4. ECG analysis history
class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    ts = Column(Float, nullable=False)
    hr = Column(Float)
    risk = Column(String(30))
    abnormal = Column(Float)
    counts = Column(Text)
    reasons = Column(Text)
