
import numpy as np
import wfdb
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets" / "mitbih"

AAMI_MAPPING = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    "V": "V",
    "E": "V",
    "F": "F",
    "/": "Q",
    "f": "Q",
    "Q": "Q"
}


def extract_heartbeats(record_name):

    record_path = DATASET_DIR / record_name

    record = wfdb.rdrecord(str(record_path))
    annotation = wfdb.rdann(str(record_path), "atr")

    lead_names = [name.upper() for name in record.sig_name]

    if "MLII" in lead_names:
        lead_index = lead_names.index("MLII")
    else:
        lead_index = 0

    signal = record.p_signal[:, lead_index]
    sampling_rate = record.fs

    beat_samples = []
    beat_symbols = []

    for sample, symbol in zip(annotation.sample, annotation.symbol):
        if symbol in AAMI_MAPPING:
            beat_samples.append(sample)
            beat_symbols.append(symbol)

    beat_samples = np.array(beat_samples)

    heartbeats = []
    labels = []
    rr_features = []

    window_before = int(0.2 * sampling_rate)
    window_after = int(0.4 * sampling_rate)

    for i, (sample, symbol) in enumerate(
        zip(beat_samples, beat_symbols)
    ):

        start = sample - window_before
        end = sample + window_after

        if start < 0 or end > len(signal):
            continue

        heartbeat = signal[start:end]

        if len(heartbeat) != 216:
            continue

        previous_rr = 0.0
        next_rr = 0.0

        if i > 0:
            previous_rr = (
                sample - beat_samples[i - 1]
            ) / sampling_rate

        if i < len(beat_samples) - 1:
            next_rr = (
                beat_samples[i + 1] - sample
            ) / sampling_rate

        valid_intervals = [
            value for value in [previous_rr, next_rr]
            if value > 0
        ]

        mean_rr = (
            np.mean(valid_intervals)
            if valid_intervals else 0.0
        )

        rr_ratio = (
            previous_rr / mean_rr
            if mean_rr > 0 and previous_rr > 0
            else 0.0
        )

        heartbeats.append(heartbeat)
        labels.append(AAMI_MAPPING[symbol])

        rr_features.append([
            previous_rr,
            next_rr,
            mean_rr,
            rr_ratio
        ])

    if not heartbeats:
        return (
            np.empty((0, 216)),
            np.array([]),
            np.empty((0, 4))
        )

    return (
        np.array(heartbeats),
        np.array(labels),
        np.array(rr_features)
    )


if __name__ == "__main__":

    X, y, rr = extract_heartbeats("100")

    print("Heartbeat Extraction Successful")
    print("Heartbeat Shape:", X.shape)
    print("Labels Shape:", y.shape)
    print("RR Features Shape:", rr.shape)
    print("All RR Values Finite:", np.isfinite(rr).all())

    classes, counts = np.unique(y, return_counts=True)

    print("\nClass Distribution:")

    for label, count in zip(classes, counts):
        print(label, ":", count)
