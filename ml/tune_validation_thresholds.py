
import numpy as np
import joblib

from pathlib import Path

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

from ml.preprocessing import normalize_dataset

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "datasets" / "mitbih" / "validation_split.npz"

MODEL_PATH = BASE_DIR / "models" / "arrhythmia_validation_rf.joblib"

print("Loading validation dataset...")

data = np.load(DATA_PATH)

X_val = data["X_val"]
y_val = data["y_val"]
rr_val = data["rr_val"]

print("Normalizing validation heartbeats...")

X_normalized = normalize_dataset(X_val)

X_combined = np.hstack((X_normalized, rr_val))

print("\nLoading validation model...")

saved_data = joblib.load(MODEL_PATH)
model = saved_data["model"]

classes = list(model.classes_)

print("Model classes:", classes)

print("\nGenerating probabilities...")

probabilities = model.predict_proba(X_combined)

thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

for target in ["S", "V", "F"]:

    class_index = classes.index(target)

    actual = (y_val == target).astype(int)

    class_probabilities = probabilities[:, class_index]

    print("\n" + "=" * 55)
    print("Class:", target)
    print("=" * 55)

    for threshold in thresholds:

        predictions = (
            class_probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            actual,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            actual,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            actual,
            predictions,
            zero_division=0
        )

        print(
            f"Threshold: {threshold:.2f} | "
            f"Precision: {precision:.3f} | "
            f"Recall: {recall:.3f} | "
            f"F1: {f1:.3f}"
        )

print("\nValidation threshold analysis completed!")
