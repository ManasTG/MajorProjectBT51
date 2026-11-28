
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from uuid import uuid4

from ml.ecg_processing import process_ecg


BASE_DIR = Path(__file__).resolve().parent.parent.parent
GRAPH_DIR = BASE_DIR / "uploads" / "graphs"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)


def generate_ecg_graph(signal, sampling_rate=360):
    """
    Generate original and filtered ECG signal graphs
    with detected R-peaks.
    """

    signal = np.asarray(signal, dtype=float)

    result = process_ecg(
        signal,
        sampling_rate
    )

    filtered_signal = result["filtered_signal"]
    r_peaks = result["r_peaks"]

    time = np.arange(len(signal)) / sampling_rate

    filename = f"ecg_{uuid4().hex}.png"
    graph_path = GRAPH_DIR / filename

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8)
    )

    # Original ECG signal
    axes[0].plot(
        time,
        signal,
        linewidth=0.8
    )

    axes[0].set_title("Original ECG Signal")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    # Filtered ECG signal
    axes[1].plot(
        time,
        filtered_signal,
        linewidth=0.8,
        label="Filtered ECG"
    )

    axes[1].scatter(
        time[r_peaks],
        filtered_signal[r_peaks],
        label="Detected R-peaks",
        zorder=3
    )

    axes[1].set_title("Filtered ECG with R-Peaks")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    fig.savefig(
        graph_path,
        dpi=120,
        bbox_inches="tight"
    )

    plt.close(fig)

    return {
        "graph_filename": filename,
        "graph_path": graph_path,
        "heart_rate": result["heart_rate"],
        "total_beats": result["total_beats"]
    }
