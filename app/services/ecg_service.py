
import pandas as pd
import numpy as np


def read_ecg_csv(file_path):
    """
    Read an ECG CSV file and validate its signal data.
    """

    try:
        df = pd.read_csv(file_path, header=None)

    except Exception as error:
        raise ValueError(f"Unable to read CSV file: {error}")

    if df.empty:
        raise ValueError("CSV file is empty.")

    if df.shape[1] == 0:
        raise ValueError("No columns found in CSV file.")

    # Convert columns to numeric where possible
    numeric_df = df.apply(pd.to_numeric, errors="coerce")

    # Keep columns containing numeric ECG values
    numeric_df = numeric_df.dropna(axis=1, how="all")

    if numeric_df.empty:
        raise ValueError("No numeric ECG signal data found.")

    # Use the first numeric column as ECG signal
    signal = numeric_df.iloc[:, 0].dropna().to_numpy(dtype=float)

    if len(signal) < 100:
        raise ValueError("ECG signal must contain at least 100 samples.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("ECG signal contains invalid values.")

    return {
        "signal": signal,
        "total_samples": len(signal),
        "column_used": str(numeric_df.columns[0])
    }
