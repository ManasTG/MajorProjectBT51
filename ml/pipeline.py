"""ECG -> filtered signal -> R-peaks -> beat features -> classifier -> risk level."""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

FS = 360  # MIT-BIH sampling rate; everything is resampled to this
CLASSES = ["N", "S", "V", "F", "Q"]  # AAMI: normal, supraventricular, ventricular, fusion, unknown
NAMES = {"N": "Normal", "S": "Supraventricular", "V": "Ventricular", "F": "Fusion", "Q": "Unknown"}


def resample(x, fs):
    x = np.asarray(x, float)
    if fs == FS:
        return x
    return np.interp(np.linspace(0, len(x) - 1, int(len(x) * FS / fs)), np.arange(len(x)), x)


def bandpass(x, lo, hi):
    b, a = butter(2, [lo / (FS / 2), hi / (FS / 2)], btype="band")
    return filtfilt(b, a, x)


def clean(x):
    """Removes baseline wander (<0.5 Hz) and high-frequency noise (>40 Hz)."""
    return bandpass(x, 0.5, 40)


def detect_r_peaks(x):
    qrs = bandpass(x, 5, 20)
    w = int(0.1 * FS)
    env = np.convolve(qrs ** 2, np.ones(w) / w, mode="same")
    p, _ = find_peaks(env, height=0.3 * np.percentile(env, 99), distance=int(0.25 * FS))
    r = int(0.06 * FS)
    return np.array([max(0, i - r) + np.argmax(np.abs(x[max(0, i - r):i + r])) for i in p], int)


def beat_features(x, peaks):
    """Per beat: 40-point normalised shape + RR-interval features. Returns (X, peak indices used)."""
    pre, post = int(0.2 * FS), int(0.4 * FS)
    rr = np.diff(peaks) / FS
    rows, used = [], []
    for k in range(1, len(peaks) - 1):
        i = peaks[k]
        if i - pre < 0 or i + post > len(x):
            continue
        raw = x[i - pre:i + post]
        seg = (raw - raw.mean()) / (raw.std() + 1e-8)
        shape = np.interp(np.linspace(0, len(seg) - 1, 40), np.arange(len(seg)), seg)
        pre_rr, post_rr = rr[k - 1], rr[k]
        local = rr[max(0, k - 8):k].mean()
        rows.append(np.r_[shape, pre_rr, post_rr, pre_rr / local, post_rr / local, np.abs(raw).max()])
        used.append(k)
    return np.array(rows), np.array(used)


def assess_risk(abnormal_frac, hr, labels):
    reasons, level = [], 0
    run = best = 0
    for l in labels:
        run = run + 1 if l == "V" else 0
        best = max(best, run)
    if best >= 3:
        level = 2; reasons.append(f"Run of {best} consecutive ventricular beats")
    if abnormal_frac >= 0.20:
        level = 2; reasons.append(f"{abnormal_frac:.0%} of beats look abnormal")
    elif abnormal_frac >= 0.05:
        level = max(level, 1); reasons.append(f"{abnormal_frac:.0%} of beats look abnormal")
    if hr < 40 or hr > 130:
        level = 2; reasons.append(f"Heart rate {hr:.0f} bpm is far outside the normal range")
    elif hr < 50 or hr > 100:
        level = max(level, 1); reasons.append(f"Heart rate {hr:.0f} bpm is outside 50-100 bpm")
    return ["Low", "Moderate", "High"][level], reasons


def analyze(signal, fs, model):
    x = clean(resample(signal, fs))
    peaks = detect_r_peaks(x)
    if len(peaks) < 6:
        raise ValueError("Too few heartbeats found. Upload at least ~10 seconds of ECG from a single lead.")
    X, used = beat_features(x, peaks)
    labels = model.predict(X)
    counts = {c: int((labels == c).sum()) for c in CLASSES if (labels == c).any()}
    hr = float(60 / np.median(np.diff(peaks) / FS))
    abnormal = 1 - counts.get("N", 0) / len(labels)
    risk, reasons = assess_risk(abnormal, hr, labels)
    n = min(len(x), 8 * FS)  # first ~8 s for the dashboard chart
    beats = [{"i": int(peaks[k]), "label": str(l)} for k, l in zip(used, labels) if peaks[k] < n]
    return {"hr": round(hr, 1), "risk": risk, "reasons": reasons, "abnormal_fraction": round(float(abnormal), 3),
            "counts": counts, "n_beats": int(len(labels)), "fs": FS,
            "waveform": [round(float(v), 3) for v in x[:n]], "beats": beats}
