
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, Patient


router = APIRouter(
    prefix="/api/patient",
    tags=["Patient Profile"]
)


# CREATE MISSING PATIENT PROFILE
@router.post("/create-profile")
def create_profile(
    request: Request,
    db: Session = Depends(get_db)
):

    user_id = request.session.get("user_id")

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Please login first"
        )

    user = db.get(User, user_id)

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    existing_patient = db.query(Patient).filter(
        Patient.user_id == user_id
    ).first()

    if existing_patient:
        return {
            "message": "Patient profile already exists"
        }

    new_patient = Patient(
        user_id=user.id,
        name=user.name,
        age=0,
        gender=None
    )

    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)

    return {
        "message": "Patient profile created successfully",
        "patient_id": new_patient.id,
        "name": new_patient.name
    }


# GET PATIENT PROFILE
@router.get("/profile")
def get_profile(
    request: Request,
    db: Session = Depends(get_db)
):

    user_id = request.session.get("user_id")

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Please login first"
        )

    patient = db.query(Patient).filter(
        Patient.user_id == user_id
    ).first()

    if not patient:
        raise HTTPException(
            status_code=404,
            detail="Patient profile not found"
        )

    user = db.get(User, user_id)

    return {
        "id": patient.id,
        "name": patient.name,
        "email": user.email,
        "age": patient.age,
        "gender": patient.gender
    }


# UPDATE PATIENT PROFILE
@router.put("/profile")
def update_profile(
    data: dict,
    request: Request,
    db: Session = Depends(get_db)
):

    user_id = request.session.get("user_id")

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Please login first"
        )

    patient = db.query(Patient).filter(
        Patient.user_id == user_id
    ).first()

    if not patient:
        raise HTTPException(
            status_code=404,
            detail="Patient profile not found"
        )

    if "name" in data:
        patient.name = data["name"]

    if "age" in data:
        age = data["age"]

        if not isinstance(age, int) or isinstance(age, bool) or not 1 <= age <= 120:
            raise HTTPException(
                status_code=400,
                detail="Age must be between 1 and 120"
            )

        patient.age = age

    if "gender" in data:
        gender = data["gender"]

        if gender not in ["Male", "Female", "Other"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid gender"
            )

        patient.gender = gender

    db.commit()
    db.refresh(patient)

    return {
        "message": "Profile updated successfully",
        "name": patient.name,
        "age": patient.age,
        "gender": patient.gender
    }
