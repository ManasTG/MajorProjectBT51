
from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Request,
    HTTPException,
    Form,
    Depends
)

from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Patient, ECGAnalysis

from app.services.ecg_service import read_ecg_csv
from app.services.ecg_visualization import generate_ecg_graph
from ml.ecg_processing import process_ecg


router = APIRouter(
    prefix="/api/ecg",
    tags=["ECG Monitoring"]
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 10 * 1024 * 1024


def check_login(request):
    user_id = request.session.get("user_id")

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Please login first"
        )

    return user_id


def get_current_patient(user_id, db):
    patient = db.query(Patient).filter(
        Patient.user_id == user_id
    ).first()

    if not patient:
        raise HTTPException(
            status_code=404,
            detail="Patient profile not found. Please create your profile."
        )

    return patient


def json_safe(value):
    """Convert NumPy values into JSON-compatible Python values."""

    if hasattr(value, "tolist"):
        return json_safe(value.tolist())

    if isinstance(value, dict):
        return {
            str(key): json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if hasattr(value, "item"):
        return value.item()

    return value


def save_temp_file(contents):
    temp_path = UPLOAD_DIR / f"{uuid4().hex}.csv"
    temp_path.write_bytes(contents)
    return temp_path


async def read_uploaded_csv(file):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are allowed"
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File size must not exceed 10 MB"
        )

    return contents


@router.post("/upload")
async def upload_ecg(
    request: Request,
    file: UploadFile = File(...)
):
    check_login(request)

    contents = await read_uploaded_csv(file)
    file_path = save_temp_file(contents)

    try:
        ecg_data = read_ecg_csv(file_path)

        unique_filename = f"{uuid4().hex}.csv"
        permanent_path = UPLOAD_DIR / unique_filename

        file_path.replace(permanent_path)

        return {
            "message": "ECG file uploaded and validated successfully",
            "filename": unique_filename,
            "original_filename": file.filename,
            "total_samples": ecg_data["total_samples"],
            "column_used": ecg_data["column_used"],
            "signal_preview": ecg_data["signal"][:10].tolist()
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to process ECG file"
        )

    finally:
        if file_path.exists():
            file_path.unlink()


@router.post("/analyze")
async def analyze_ecg(
    request: Request,
    file: UploadFile = File(...),
    sampling_rate: int = Form(360),
    db: Session = Depends(get_db)
):
    user_id = check_login(request)
    patient = get_current_patient(user_id, db)

    if sampling_rate != 360:
        raise HTTPException(
            status_code=400,
            detail="The currently integrated ML model requires 360 Hz ECG data."
        )

    contents = await read_uploaded_csv(file)
    temp_path = save_temp_file(contents)

    try:
        ecg_data = read_ecg_csv(temp_path)

        result = process_ecg(
            ecg_data["signal"],
            sampling_rate
        )

        total_beats = int(result["total_beats"])
        abnormal_burden = float(result["abnormal_burden"])

        class_counts = json_safe(result["class_counts"])
        class_percentages = json_safe(result["class_percentages"])

        abnormal_beats = int(result["abnormal_beats"])
        average_confidence = float(result["average_confidence"])

        beat_predictions = json_safe(result["beat_predictions"])
        r_peak_indices = json_safe(result["r_peaks"])

        # Store a downsampled waveform to keep the database compact.
        filtered_signal = result["filtered_signal"]
        max_waveform_points = 2000

        if len(filtered_signal) > max_waveform_points:
            step = max(
                1,
                len(filtered_signal) // max_waveform_points
            )
            filtered_signal = filtered_signal[::step][:max_waveform_points]

        filtered_signal = json_safe(filtered_signal)

        if beat_predictions:
            risk_level = "Experimental classification"
        else:
            risk_level = "Insufficient beats"

        summary = (
            "Experimental ML heartbeat classification completed. "
            f"Detected {total_beats} beats. "
            f"Predicted abnormal-class burden: {abnormal_burden}%. "
            "This output is not a clinical diagnosis."
        )

        analysis = ECGAnalysis(
            patient_id=patient.id,
            heart_rate=float(result["heart_rate"]),
            total_beats=total_beats,
            abnormal_beat_burden=f"{abnormal_burden}%",
            risk_level=risk_level,
            result_summary=summary,

            total_samples=int(ecg_data["total_samples"]),
            sampling_rate=sampling_rate,

            class_counts=class_counts,
            class_percentages=class_percentages,

            abnormal_beats=abnormal_beats,
            average_confidence=average_confidence,

            beat_predictions=beat_predictions,
            r_peak_indices=r_peak_indices,
            filtered_signal=filtered_signal
        )

        db.add(analysis)
        db.commit()
        db.refresh(analysis)

        return {
            "message": "ECG and experimental ML analysis completed",
            "analysis_id": analysis.id,
            "patient_id": patient.id,
            "total_samples": ecg_data["total_samples"],
            "sampling_rate": sampling_rate,
            "total_beats": total_beats,
            "heart_rate_bpm": result["heart_rate"],
            "class_counts": class_counts,
            "class_percentages": class_percentages,
            "abnormal_beats": abnormal_beats,
            "abnormal_burden_percent": abnormal_burden,
            "average_model_confidence_percent": average_confidence,
            "beat_predictions": beat_predictions,
            "r_peak_indices": r_peak_indices,
            "filtered_signal_preview": filtered_signal[:10],
            "risk_level": risk_level,
            "disclaimer": (
                "Experimental research output only. "
                "Not intended for medical diagnosis or treatment."
            )
        }

    except ValueError as error:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        db.rollback()
        print("ECG ANALYSIS ERROR:", repr(error))
        raise HTTPException(
            status_code=500,
            detail="ECG analysis failed"
        )

    finally:
        temp_path.unlink(missing_ok=True)


@router.post("/visualize")
async def visualize_ecg(
    request: Request,
    file: UploadFile = File(...),
    sampling_rate: int = Form(360)
):
    check_login(request)

    if sampling_rate != 360:
        raise HTTPException(
            status_code=400,
            detail="Sampling rate must be 360 Hz for this model."
        )

    contents = await read_uploaded_csv(file)
    temp_path = save_temp_file(contents)

    try:
        ecg_data = read_ecg_csv(temp_path)

        graph = generate_ecg_graph(
            ecg_data["signal"],
            sampling_rate
        )

        return FileResponse(
            path=str(graph["graph_path"]),
            media_type="image/png",
            filename=graph["graph_filename"]
        )

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        print("ECG VISUALIZATION ERROR:", repr(error))
        raise HTTPException(
            status_code=500,
            detail="Unable to generate ECG visualization"
        )

    finally:
        temp_path.unlink(missing_ok=True)
