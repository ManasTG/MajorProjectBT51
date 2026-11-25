
import os
from fastapi import APIRouter, HTTPException, Response, Request
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.database import SessionLocal
from app.models import User, Patient
from app.auth import hash_password, verify_password
from app.schemas import SignupRequest, LoginRequest

router = APIRouter(prefix="/api", tags=["Authentication"])

SECRET_KEY = os.getenv(
    "SECRET_KEY",
    "local-development-secret-change-before-deploy"
)

serializer = URLSafeTimedSerializer(
    SECRET_KEY,
    salt="patient-health-session"
)

SESSION_MAX_AGE = 86400


# SIGNUP
@router.post("/signup")
def signup(data: SignupRequest):
    db = SessionLocal()

    try:
        existing_user = db.scalar(
            select(User).where(User.email == data.email.lower())
        )

        if existing_user:
            raise HTTPException(409, "Email already registered")

        user = User(
            name=data.name.strip(),
            email=data.email.lower(),
            password_hash=hash_password(data.password),
            role="patient"
        )

        db.add(user)
        db.flush()

        patient = Patient(
            name=data.name.strip(),
            age=data.age,
            user_id=user.id
        )

        db.add(patient)
        db.commit()

        return {
            "message": "Registration successful",
            "user": {
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "role": user.role
            }
        }

    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Email already registered")

    finally:
        db.close()


# LOGIN
@router.post("/login")
def login(data: LoginRequest, response: Response):
    db = SessionLocal()

    try:
        user = db.scalar(
            select(User).where(User.email == data.email.lower())
        )

        if not user or not verify_password(
            data.password,
            user.password_hash
        ):
            raise HTTPException(401, "Invalid email or password")

        token = serializer.dumps({
            "user_id": user.id,
            "role": user.role
        })

        response.set_cookie(
            key="session",
            value=token,
            httponly=True,
            samesite="lax",
            secure=os.getenv("ENVIRONMENT") == "production",
            max_age=SESSION_MAX_AGE,
            path="/"
        )

        return {
            "message": "Login successful",
            "user": {
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "role": user.role
            }
        }

    finally:
        db.close()


# GET CURRENT LOGGED-IN USER
def get_current_user(request: Request):
    token = request.cookies.get("session")

    if not token:
        raise HTTPException(401, "Please login first")

    try:
        payload = serializer.loads(
            token,
            max_age=SESSION_MAX_AGE
        )
    except (BadSignature, SignatureExpired):
        raise HTTPException(401, "Session expired. Please login again.")

    db = SessionLocal()

    try:
        user = db.scalar(
            select(User).where(User.id == payload.get("user_id"))
        )

        if not user:
            raise HTTPException(401, "User account not found")

        return {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "role": user.role
        }

    finally:
        db.close()


# CHECK LOGIN SESSION
@router.get("/me")
def current_user(request: Request):
    user = get_current_user(request)

    return {
        "logged_in": True,
        "user": user
    }


# LOGOUT
@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(
        key="session",
        path="/",
        httponly=True,
        samesite="lax",
        secure=os.getenv("ENVIRONMENT") == "production"
    )

    return {
        "message": "Logout successful"
    }
