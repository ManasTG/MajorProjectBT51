
"""Run: uvicorn app.main:app --reload"""

import json
import sqlite3
import time

from pathlib import Path

import joblib

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ml.pipeline import analyze, FS
from ml.synth import synth_record

from app.database import engine, Base
from app import models
from app.auth_routes import (
    router as auth_router,
    get_current_user
)


# APPLICATION SETUP

app = FastAPI(title="AI-Powered Patient Health Monitoring")

Base.metadata.create_all(bind=engine)

app.include_router(auth_router)


# LOAD MACHINE LEARNING MODEL

MODEL = Path("models/arrhythmia_rf.joblib")
model = joblib.load(MODEL) if MODEL.exists() else None


# DATABASE CONNECTION

db = sqlite3.connect("data.db", check_same_thread=False)
db.row_factory = sqlite3.Row

db.executescript("""
CREATE TABLE IF NOT EXISTS patients(
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INT,
    user_id INTEGER
);

CREATE TABLE IF NOT EXISTS analyses(
    id INTEGER PRIMARY KEY,
    patient_id INT,
    ts REAL,
    hr REAL,
    risk TEXT,
    abnormal REAL,
    counts TEXT,
    reasons TEXT
);
""")

db.commit()


# REQUEST MODELS

class Patient(BaseModel):
    name: str
    age: int


class Ecg(BaseModel):
    patient_id: int
    fs: float = FS
    samples: list[float]


# HELPER: CHECK PATIENT OWNERSHIP

def get_owned_patient(pid: int, current_user: dict):

    if current_user["role"] != "patient":
        raise HTTPException(
            status_code=403,
            detail="Patient access only"
        )

    patient = db.execute(
        "SELECT * FROM patients WHERE id=? AND user_id=?",
        (pid, current_user["id"])
    ).fetchone()

    if not patient:
        raise HTTPException(
            status_code=404,
            detail="Patient not found"
        )

    return patient


# HOME PAGE

@app.get("/")
def index():
    return FileResponse("app/static/index.html")


# ADD PATIENT

@app.post("/api/patients")
def add_patient(
    p: Patient,
    current_user: dict = Depends(get_current_user)
):

    if current_user["role"] != "patient":
        raise HTTPException(403, "Patient access only")

    cur = db.execute(
        """
        INSERT INTO patients(name, age, user_id)
        VALUES(?,?,?)
        """,
        (p.name, p.age, current_user["id"])
    )

    db.commit()

    return {
        "id": cur.lastrowid,
        **p.model_dump()
    }


# GET LOGGED-IN PATIENT'S RECORDS

@app.get("/api/patients")
def patients(
    current_user: dict = Depends(get_current_user)
):

    if current_user["role"] != "patient":
        raise HTTPException(403, "Patient access only")

    records = db.execute(
        """
        SELECT id, name, age
        FROM patients
        WHERE user_id=?
        ORDER BY id DESC
        """,
        (current_user["id"],)
    )

    return [dict(r) for r in records]


# UPDATE PATIENT

@app.put("/api/patients/{pid}")
def update_patient(
    pid: int,
    p: Patient,
    current_user: dict = Depends(get_current_user)
):

    get_owned_patient(pid, current_user)

    db.execute(
        """
        UPDATE patients
        SET name=?, age=?
        WHERE id=? AND user_id=?
        """,
        (p.name, p.age, pid, current_user["id"])
    )

    db.commit()

    return {
        "message": "Patient updated successfully"
    }


# DELETE PATIENT

@app.delete("/api/patients/{pid}")
def delete_patient(
    pid: int,
    current_user: dict = Depends(get_current_user)
):

    get_owned_patient(pid, current_user)

    db.execute(
        """
        DELETE FROM analyses
        WHERE patient_id=?
        """,
        (pid,)
    )

    db.execute(
        """
        DELETE FROM patients
        WHERE id=? AND user_id=?
        """,
        (pid, current_user["id"])
    )

    db.commit()

    return {
        "message": "Patient deleted successfully"
    }


# ANALYZE ECG

@app.post("/api/analyze")
def run(
    e: Ecg,
    current_user: dict = Depends(get_current_user)
):

    get_owned_patient(e.patient_id, current_user)

    if model is None:
        raise HTTPException(
            503,
            "No trained model. Run: python -m ml.train"
        )

    try:
        res = analyze(e.samples, e.fs, model)

    except ValueError as err:
        raise HTTPException(422, str(err))

    db.execute(
        """
        INSERT INTO analyses(
            patient_id, ts, hr, risk,
            abnormal, counts, reasons
        )
        VALUES(?,?,?,?,?,?,?)
        """,
        (
            e.patient_id,
            time.time(),
            res["hr"],
            res["risk"],
            res["abnormal_fraction"],
            json.dumps(res["counts"]),
            json.dumps(res["reasons"])
        )
    )

    db.commit()

    return res


# PATIENT ECG HISTORY

@app.get("/api/patients/{pid}/history")
def history(
    pid: int,
    current_user: dict = Depends(get_current_user)
):

    get_owned_patient(pid, current_user)

    records = db.execute(
        """
        SELECT *
        FROM analyses
        WHERE patient_id=?
        ORDER BY ts DESC
        LIMIT 50
        """,
        (pid,)
    )

    return [
        {
            **dict(r),
            "counts": json.loads(r["counts"]),
            "reasons": json.loads(r["reasons"])
        }
        for r in records
    ]


# SYNTHETIC ECG DEMO

@app.get("/api/demo")
def demo(kind: str = "normal"):

    x, _, _ = synth_record(
        n_beats=25,
        seed=int(time.time()) % 1000,
        p_abn=0.0 if kind == "normal" else 0.4
    )

    return {
        "fs": FS,
        "samples": [round(float(v), 4) for v in x]
    }
