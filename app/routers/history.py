
from fastapi import APIRouter, Request, HTTPException, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Patient, ECGAnalysis


router = APIRouter(
    prefix="/api/patient",
    tags=["Patient ECG History"]
)


@router.get("/history")
def get_ecg_history(
    request: Request,
    db: Session = Depends(get_db)
):

    # Check login
    user_id = request.session.get("user_id")

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Please login first"
        )

    # Find patient
    patient = db.query(Patient).filter(
        Patient.user_id == user_id
    ).first()

    if not patient:
        raise HTTPException(
            status_code=404,
            detail="Patient profile not found"
        )

    # Fetch patient's ECG history
    records = (
        db.query(ECGAnalysis)
        .filter(ECGAnalysis.patient_id == patient.id)
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
        "message": "ECG history retrieved successfully",
        "total_records": len(history),
        "history": history
    }
