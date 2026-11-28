
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, Patient, ECGAnalysis

router = APIRouter(prefix="/api/admin", tags=["Admin Dashboard"])


def verify_admin(request: Request, db: Session):
    user_id = request.session.get("user_id")

    if not user_id:
        raise HTTPException(status_code=401, detail="Please login first")

    user = db.get(User, user_id)

    if not user:
        request.session.clear()
        raise HTTPException(status_code=401, detail="User not found")

    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    return user


@router.get("/dashboard")
def admin_dashboard(
    request: Request,
    db: Session = Depends(get_db)
):
    admin = verify_admin(request, db)

    total_users = db.query(User).count()
    total_patients = db.query(User).filter(User.role == "patient").count()
    total_doctors = db.query(User).filter(User.role == "doctor").count()
    total_analyses = db.query(ECGAnalysis).count()

    recent_users = db.query(User).order_by(
        User.created_at.desc()
    ).limit(10).all()

    recent_analyses = (
        db.query(ECGAnalysis, Patient)
        .join(Patient, ECGAnalysis.patient_id == Patient.id)
        .order_by(ECGAnalysis.created_at.desc())
        .limit(10)
        .all()
    )

    risk_counts = {}
    for risk in ["High", "Medium", "Low", "Experimental classification",
                 "Insufficient beats"]:
        risk_counts[risk] = (
            db.query(ECGAnalysis)
            .filter(ECGAnalysis.risk_level == risk)
            .count()
        )

    users = [
        {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "role": user.role,
            "created_at": user.created_at.isoformat()
            if user.created_at else None
        }
        for user in recent_users
    ]

    analyses = [
        {
            "analysis_id": analysis.id,
            "patient_name": patient.name,
            "heart_rate_bpm": analysis.heart_rate,
            "total_beats": analysis.total_beats,
            "abnormal_beat_burden": analysis.abnormal_beat_burden,
            "risk_level": analysis.risk_level,
            "created_at": analysis.created_at.isoformat()
            if analysis.created_at else None
        }
        for analysis, patient in recent_analyses
    ]

    return {
        "admin": {
            "id": admin.id,
            "name": admin.name,
            "email": admin.email
        },
        "statistics": {
            "total_users": total_users,
            "total_patients": total_patients,
            "total_doctors": total_doctors,
            "total_analyses": total_analyses
        },
        "risk_overview": risk_counts,
        "recent_users": users,
        "recent_analyses": analyses
    }
