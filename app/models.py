
from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    ForeignKey,
    DateTime,
    Float,
    JSON
)

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="patient")
    created_at = Column(DateTime, default=datetime.utcnow)


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)


class ECGAnalysis(Base):
    __tablename__ = "ecg_analyses"

    id = Column(Integer, primary_key=True, index=True)

    patient_id = Column(
        Integer,
        ForeignKey("patients.id"),
        nullable=False
    )

    heart_rate = Column(Float)
    total_beats = Column(Integer)

    abnormal_beat_burden = Column(String(50))
    risk_level = Column(String(50))
    result_summary = Column(String(500))

    # Detailed ECG analysis results
    total_samples = Column(Integer)
    sampling_rate = Column(Integer)

    class_counts = Column(JSON)
    class_percentages = Column(JSON)

    abnormal_beats = Column(Integer)
    average_confidence = Column(Float)

    beat_predictions = Column(JSON)
    r_peak_indices = Column(JSON)

    # Filtered ECG waveform for PDF visualization
    filtered_signal = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)
