
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse

from app.database import Base, engine, SessionLocal
from app import models

from app.routers.auth import router as auth_router
from app.routers.patient import router as patient_router
from app.routers.ecg import router as ecg_router
from app.routers.history import router as history_router
from app.routers.doctor import router as doctor_router
from app.routers.admin import router as admin_router
from app.routers.reports import router as reports_router


# Project directory
BASE_DIR = Path(__file__).resolve().parent


# FastAPI application
app = FastAPI(
    title="PulseAI - Patient Health Monitoring",
    description="AI-Powered Patient Health Monitoring and Early Disease Prediction",
    version="1.0.0"
)


# Create database tables
Base.metadata.create_all(bind=engine)


# Session configuration (development only)
app.add_middleware(
    SessionMiddleware,
    secret_key="pulseai-development-secret-change-before-deployment",
    max_age=3600,
    same_site="lax",
    https_only=False
)


# Static files
app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static"
)


# HTML templates
templates = Jinja2Templates(
    directory=BASE_DIR / "templates"
)


# Register API routers
app.include_router(auth_router)
app.include_router(patient_router)
app.include_router(ecg_router)
app.include_router(history_router)
app.include_router(doctor_router)
app.include_router(admin_router)
app.include_router(reports_router)


# ---------------------------------
# HOME PAGE
# ---------------------------------

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# ---------------------------------
# AUTHENTICATION PAGE
# ---------------------------------

@app.get("/auth")
def auth_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="auth.html"
    )


# ---------------------------------
# PATIENT DASHBOARD
# ---------------------------------

@app.get("/dashboard")
def dashboard(request: Request):

    user_id = request.session.get("user_id")

    if not user_id:
        return RedirectResponse(
            url="/auth",
            status_code=303
        )

    db = SessionLocal()

    try:
        user = db.get(models.User, user_id)
    finally:
        db.close()

    if not user:
        request.session.clear()

        return RedirectResponse(
            url="/auth",
            status_code=303
        )

    if user.role != "patient":
        if user.role == "doctor":
            return RedirectResponse(
                url="/doctor",
                status_code=303
            )

        if user.role == "admin":
            return RedirectResponse(
                url="/admin",
                status_code=303
            )

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"user": user}
    )


# ---------------------------------
# DOCTOR DASHBOARD
# ---------------------------------

@app.get("/doctor")
def doctor_page(request: Request):

    user_id = request.session.get("user_id")

    if not user_id:
        return RedirectResponse(
            url="/auth",
            status_code=303
        )

    db = SessionLocal()

    try:
        user = db.get(models.User, user_id)
    finally:
        db.close()

    if not user:
        request.session.clear()

        return RedirectResponse(
            url="/auth",
            status_code=303
        )

    if user.role not in ["doctor", "admin"]:
        return RedirectResponse(
            url="/dashboard",
            status_code=303
        )

    return templates.TemplateResponse(
        request=request,
        name="doctor.html",
        context={"user": user}
    )


# ---------------------------------
# ADMIN DASHBOARD
# ---------------------------------

@app.get("/admin")
def admin_page(request: Request):

    user_id = request.session.get("user_id")

    if not user_id:
        return RedirectResponse(
            url="/auth",
            status_code=303
        )

    db = SessionLocal()

    try:
        user = db.get(models.User, user_id)
    finally:
        db.close()

    if not user:
        request.session.clear()

        return RedirectResponse(
            url="/auth",
            status_code=303
        )

    if user.role != "admin":
        return RedirectResponse(
            url="/dashboard",
            status_code=303
        )

    return templates.TemplateResponse(
        request=request,
        name="admin.html",
        context={"user": user}
    )


# ---------------------------------
# ECG MONITORING PAGE
# ---------------------------------

@app.get("/ecg")
def ecg_page(request: Request):

    user_id = request.session.get("user_id")

    if not user_id:
        return RedirectResponse(
            url="/auth",
            status_code=303
        )

    db = SessionLocal()

    try:
        user = db.get(models.User, user_id)
    finally:
        db.close()

    if not user:
        request.session.clear()

        return RedirectResponse(
            url="/auth",
            status_code=303
        )

    if user.role != "patient":
        if user.role == "doctor":
            return RedirectResponse(
                url="/doctor",
                status_code=303
            )

        if user.role == "admin":
            return RedirectResponse(
                url="/admin",
                status_code=303
            )

    return templates.TemplateResponse(
        request=request,
        name="ecg.html",
        context={"user": user}
    )


# ---------------------------------
# HEALTH CHECK
# ---------------------------------

@app.get("/health")
def health_check():

    return {
        "message": "Welcome to PulseAI",
        "status": "Backend is running successfully"
    }
