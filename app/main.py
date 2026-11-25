"""Run: uvicorn app.main:app --reload"""
import json, sqlite3, time
from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ml.pipeline import analyze, FS
from ml.synth import synth_record

app = FastAPI(title="AI-Powered Patient Health Monitoring")
MODEL = Path("models/arrhythmia_rf.joblib")
model = joblib.load(MODEL) if MODEL.exists() else None
db = sqlite3.connect("data.db", check_same_thread=False)
db.row_factory = sqlite3.Row
db.executescript("""CREATE TABLE IF NOT EXISTS patients(id INTEGER PRIMARY KEY, name TEXT, age INT);
CREATE TABLE IF NOT EXISTS analyses(id INTEGER PRIMARY KEY, patient_id INT, ts REAL, hr REAL, risk TEXT,
  abnormal REAL, counts TEXT, reasons TEXT);""")


class Patient(BaseModel):
    name: str
    age: int


class Ecg(BaseModel):
    patient_id: int
    fs: float = FS
    samples: list[float]


@app.get("/")
def index():
    return FileResponse("app/static/index.html")


@app.post("/api/patients")
def add_patient(p: Patient):
    cur = db.execute("INSERT INTO patients(name, age) VALUES(?,?)", (p.name, p.age)); db.commit()
    return {"id": cur.lastrowid, **p.model_dump()}


@app.get("/api/patients")
def patients():
    return [dict(r) for r in db.execute("SELECT * FROM patients ORDER BY id DESC")]

# UPDATE PATIENT
@app.put("/api/patients/{pid}")
def update_patient(pid: int, p: Patient):

    existing = db.execute(
        "SELECT * FROM patients WHERE id=?", (pid,)
    ).fetchone()

    if not existing:
        raise HTTPException(404, "Patient not found")

    db.execute(
        "UPDATE patients SET name=?, age=? WHERE id=?",
        (p.name, p.age, pid)
    )

    db.commit()

    return {"message": "Patient updated successfully"}


# DELETE PATIENT
@app.delete("/api/patients/{pid}")
def delete_patient(pid: int):

    existing = db.execute(
        "SELECT * FROM patients WHERE id=?", (pid,)
    ).fetchone()

    if not existing:
        raise HTTPException(404, "Patient not found")

    # Delete associated analysis history first
    db.execute("DELETE FROM analyses WHERE patient_id=?", (pid,))

    # Delete patient
    db.execute("DELETE FROM patients WHERE id=?", (pid,))

    db.commit()

    return {"message": "Patient deleted successfully"}    


@app.post("/api/analyze")
def run(e: Ecg):
    if model is None:
        raise HTTPException(503, "No trained model. Run: python -m ml.train")
    if not db.execute("SELECT 1 FROM patients WHERE id=?", (e.patient_id,)).fetchone():
        raise HTTPException(404, "Patient not found")
    try:
        res = analyze(e.samples, e.fs, model)
    except ValueError as err:
        raise HTTPException(422, str(err))
    db.execute("INSERT INTO analyses(patient_id, ts, hr, risk, abnormal, counts, reasons) VALUES(?,?,?,?,?,?,?)",
               (e.patient_id, time.time(), res["hr"], res["risk"], res["abnormal_fraction"],
                json.dumps(res["counts"]), json.dumps(res["reasons"])))
    db.commit()
    return res


@app.get("/api/patients/{pid}/history")
def history(pid: int):
    return [{**dict(r), "counts": json.loads(r["counts"]), "reasons": json.loads(r["reasons"])}
            for r in db.execute("SELECT * FROM analyses WHERE patient_id=? ORDER BY ts DESC LIMIT 50", (pid,))]


@app.get("/api/demo")
def demo(kind: str = "normal"):
    x, _, _ = synth_record(n_beats=25, seed=int(time.time()) % 1000, p_abn=0.0 if kind == "normal" else 0.4)
    return {"fs": FS, "samples": [round(float(v), 4) for v in x]}
