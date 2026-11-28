
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from sqlalchemy import select
from pwdlib import PasswordHash

from app.database import get_db
from app.models import User, Patient
from app.schemas import UserSignup, UserLogin, UserResponse


router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"]
)

password_hash = PasswordHash.recommended()


# SIGNUP API
@router.post("/signup", response_model=UserResponse)
def signup(data: UserSignup, db: Session = Depends(get_db)):

    existing_user = db.scalar(
        select(User).where(User.email == data.email)
    )

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    hashed_password = password_hash.hash(data.password)

    new_user = User(
        name=data.name,
        email=data.email,
        hashed_password=hashed_password,
        role="patient"
    )

    try:
        db.add(new_user)
        db.flush()

        new_patient = Patient(
            user_id=new_user.id,
            name=new_user.name,
            age=0,
            gender=None
        )

        db.add(new_patient)
        db.commit()
        db.refresh(new_user)

    except Exception:
        db.rollback()
        raise

    return new_user


# LOGIN API
@router.post("/login")
def login(
    request: Request,
    data: UserLogin,
    db: Session = Depends(get_db)
):

    user = db.scalar(
        select(User).where(User.email == data.email)
    )

    if not user or not password_hash.verify(
        data.password,
        user.hashed_password
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    request.session.clear()
    request.session["user_id"] = user.id
    request.session["name"] = user.name
    request.session["role"] = user.role

    return {
        "message": "Login successful",
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "role": user.role
        }
    }


# LOGOUT API
@router.post("/logout")
def logout(request: Request):

    request.session.clear()

    return {
        "message": "Logout successful"
    }


# CURRENT USER API
@router.get("/me")
def get_current_user(
    request: Request,
    db: Session = Depends(get_db)
):

    user_id = request.session.get("user_id")

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Please login first"
        )

    user = db.get(User, user_id)

    if not user:
        request.session.clear()

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role
    }
