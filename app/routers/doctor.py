
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, Patient, ECGAnalysis


router = APIRouter(
    prefix="/api/doctor",
    tags=["Doctor Dashboard"]
)


def verify_doctor(request: Request, db: Session):

    user_id = request.session.get("user_id")

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Please login first"
        )

    user = db.get(User, user_id)

    if not user:
        request.session.clear()
        raise HTTPException(
            status_code=401,
            detail="User not found"
        )

    if user.role not in ["doctor", "admin"]:
        raise HTTPException(
            status_code=403,
            detail="Doctor or admin access required"
        )

    return user


@router.get("/dashboard")
def doctor_dashboard(
    request: Request,
    db: Session = Depends(get_db)
):

    doctor = verify_doctor(request, db)

    patients = (
        db.query(Patient)
        .order_by(Patient.created_at.desc())
        .all()
    )

    patient_records = []

    for patient in patients:

        user = db.get(User, patient.user_id)

        latest = (
            db.query(ECGAnalysis)
            .filter(ECGAnalysis.patient_id == patient.id)
            .order_by(ECGAnalysis.created_at.desc())
            .first()
        )

        patient_records.append({
            "patient_id": patient.id,
            "name": patient.name,
            "email": user.email if user else None,
            "age": patient.age,
            "gender": patient.gender,
            "total_analyses": (
                db.query(ECGAnalysis)
                .filter(ECGAnalysis.patient_id == patient.id)
                .count()
            ),
            "latest_analysis": {
                "analysis_id": latest.id,
                "heart_rate_bpm": latest.heart_rate,
                "total_beats": latest.total_beats,
                "abnormal_beat_burden": latest.abnormal_beat_burden,
                "risk_level": latest.risk_level,
                "created_at": latest.created_at.isoformat()
            } if latest else None
        })

    return {
        "message": "Doctor dashboard data retrieved successfully",
        "doctor": {
            "id": doctor.id,
            "name": doctor.name,
            "email": doctor.email
        },
        "total_patients": len(patient_records),
        "total_analyses": db.query(ECGAnalysis).count(),
        "patients": patient_records
    }


@router.get("/patients/{patient_id}/history")
def get_patient_history(
    patient_id: int,
    request: Request,
    db: Session = Depends(get_db)
):

    verify_doctor(request, db)

    patient = db.get(Patient, patient_id)

    if not patient:
        raise HTTPException(
            status_code=404,
            detail="Patient not found"
        )

    records = (
        db.query(ECGAnalysis)
        .filter(ECGAnalysis.patient_id == patient_id)
        .order_by(ECGAnalysis.created_at.desc())
        .all()
    )

    history = []

    for record in records:
        history.append({
            "analysis_id": record.id,
            "heart_rate_bpm": record.heart_rate,
            "total_beats": record.total_beats,
            "abnormal_beat_burden": record.abnormal_beat_burden,
            "risk_level": record.risk_level,
            "result_summary": record.result_summary,
            "created_at": record.created_at.isoformat()
        })

    return {
        "patient": {
            "id": patient.id,
            "name": patient.name,
            "age": patient.age,
            "gender": patient.gender
        },
        "total_records": len(history),
        "history": history
    }
