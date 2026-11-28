
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "arrhythmia_rf_random_oversampling.joblib"

MODEL_DATA = joblib.load(MODEL_PATH)
MODEL = MODEL_DATA["model"]

INPUT_LENGTH = MODEL_DATA["input_length"]
RR_FEATURE_COUNT = MODEL_DATA["rr_feature_count"]
EXPECTED_FEATURE_COUNT = MODEL_DATA["feature_count"]
MODEL_SAMPLING_RATE = MODEL_DATA["sampling_rate"]


def bandpass_filter(signal, sampling_rate=360):

    signal = np.asarray(signal, dtype=float)

    if len(signal) < 20:
        raise ValueError("ECG signal is too short for filtering.")

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")

    low_cutoff = 0.5
    high_cutoff = 40

    if high_cutoff >= sampling_rate / 2:
        raise ValueError("Sampling rate is too low for this filter.")

    nyquist = sampling_rate / 2

    low = low_cutoff / nyquist
    high = high_cutoff / nyquist

    b, a = butter(4, [low, high], btype="band")

    return filtfilt(b, a, signal)


def detect_r_peaks(signal, sampling_rate=360):

    signal = np.asarray(signal, dtype=float)

    if len(signal) < 3:
        raise ValueError("ECG signal is too short.")

    min_distance = int(0.25 * sampling_rate)

    peaks, _ = find_peaks(
        signal,
        distance=min_distance,
        prominence=np.std(signal) * 0.5
    )

    return peaks


def calculate_heart_rate(r_peaks, sampling_rate=360):

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")

    if len(r_peaks) < 2:
        return None

    intervals = np.diff(r_peaks) / sampling_rate
    valid_intervals = intervals[intervals > 0]

    if len(valid_intervals) == 0:
        return None

    return round(float(60 / np.mean(valid_intervals)), 2)


def normalize_heartbeat(heartbeat):

    heartbeat = np.asarray(heartbeat, dtype=float)

    mean = np.mean(heartbeat)
    std = np.std(heartbeat)

    if std < 1e-8:
        return np.zeros_like(heartbeat)

    return (heartbeat - mean) / std


def classify_heartbeats(filtered_signal, r_peaks, sampling_rate):

    if sampling_rate != MODEL_SAMPLING_RATE:
        raise ValueError(
            f"The trained model requires {MODEL_SAMPLING_RATE} Hz ECG data."
        )

    predictions = []
    feature_rows = []

    window_before = int(0.2 * sampling_rate)
    window_after = int(0.4 * sampling_rate)

    for i, peak in enumerate(r_peaks):

        start = peak - window_before
        end = peak + window_after

        if start < 0 or end > len(filtered_signal):
            continue

        heartbeat = filtered_signal[start:end]

        if len(heartbeat) != INPUT_LENGTH:
            continue

        previous_rr = 0.0
        next_rr = 0.0

        if i > 0:
            previous_rr = (
                peak - r_peaks[i - 1]
            ) / sampling_rate

        if i < len(r_peaks) - 1:
            next_rr = (
                r_peaks[i + 1] - peak
            ) / sampling_rate

        valid_intervals = [
            value for value in [previous_rr, next_rr]
            if value > 0
        ]

        mean_rr = (
            float(np.mean(valid_intervals))
            if valid_intervals else 0.0
        )

        rr_ratio = (
            previous_rr / mean_rr
            if mean_rr > 0 and previous_rr > 0
            else 0.0
        )

        rr_features = np.array([
            previous_rr,
            next_rr,
            mean_rr,
            rr_ratio
        ])

        if len(rr_features) != RR_FEATURE_COUNT:
            raise ValueError("RR feature count mismatch.")

        normalized = normalize_heartbeat(heartbeat)

        combined_features = np.concatenate([
            normalized,
            rr_features
        ])

        if len(combined_features) != EXPECTED_FEATURE_COUNT:
            raise ValueError("Model feature count mismatch.")

        feature_rows.append(combined_features)

        predictions.append({
            "r_peak_index": int(peak)
        })

    if not feature_rows:
        return {
            "beat_predictions": [],
            "class_counts": {},
            "class_percentages": {},
            "abnormal_beats": 0,
            "abnormal_burden": 0.0,
            "average_confidence": None
        }

    X = np.asarray(feature_rows, dtype=float)

    predicted_classes = MODEL.predict(X)
    probabilities = MODEL.predict_proba(X)

    class_counts = {
        label: int(np.sum(predicted_classes == label))
        for label in ["N", "S", "V", "F", "Q"]
    }

    total = len(predicted_classes)

    class_percentages = {
        label: round(count / total * 100, 2)
        for label, count in class_counts.items()
    }

    beat_predictions = []

    for i, label in enumerate(predicted_classes):

        confidence = float(np.max(probabilities[i]))

        beat_predictions.append({
            **predictions[i],
            "predicted_class": str(label),
            "confidence": round(confidence * 100, 2)
        })

    abnormal_beats = total - class_counts["N"]

    return {
        "beat_predictions": beat_predictions,
        "class_counts": class_counts,
        "class_percentages": class_percentages,
        "abnormal_beats": abnormal_beats,
        "abnormal_burden": round(
            abnormal_beats / total * 100, 2
        ),
        "average_confidence": round(
            float(np.mean(np.max(probabilities, axis=1))) * 100,
            2
        )
    }


def process_ecg(signal, sampling_rate=360):

    filtered_signal = bandpass_filter(
        signal,
        sampling_rate
    )

    r_peaks = detect_r_peaks(
        filtered_signal,
        sampling_rate
    )

    heart_rate = calculate_heart_rate(
        r_peaks,
        sampling_rate
    )

    classification = classify_heartbeats(
        filtered_signal,
        r_peaks,
        sampling_rate
    )

    return {
        "filtered_signal": filtered_signal,
        "r_peaks": r_peaks,
        "heart_rate": heart_rate,
        "total_beats": len(r_peaks),
        **classification
    }
