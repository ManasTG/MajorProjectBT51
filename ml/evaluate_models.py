
import numpy as np
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from ml.preprocessing import normalize_dataset

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "datasets" / "mitbih" / "heartbeat_dataset.npz"

MODEL_DIR = BASE_DIR / "models"

MODEL_FILES = {
    "SMOTE Only": "arrhythmia_rf.joblib",
    "Random Oversampling": "arrhythmia_rf_random_oversampling.joblib",
    "Balanced Random Forest": "arrhythmia_balanced_rf.joblib"
}

print("Loading test dataset...")

data = np.load(DATASET_PATH)

X = data["X"]
rr_features = data["rr_features"]
y = data["y"]
splits = data["splits"]

test_mask = splits == "DS2"

X_test = X[test_mask]
rr_test = rr_features[test_mask]
y_test = y[test_mask]

print("Normalizing test heartbeats...")

X_test_normalized = normalize_dataset(X_test)

X_test_combined = np.hstack(
    (X_test_normalized, rr_test)
)

classes = ["N", "S", "V", "F", "Q"]

results = []

for model_name, filename in MODEL_FILES.items():

    model_path = MODEL_DIR / filename

    print("\n" + "=" * 60)
    print("Evaluating:", model_name)
    print("=" * 60)

    if not model_path.exists():
        print("Model file not found:", filename)
        continue

    saved_data = joblib.load(model_path)

    model = saved_data["model"]

    predictions = model.predict(X_test_combined)

    accuracy = accuracy_score(y_test, predictions)

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    macro_precision = precision_score(
        y_test,
        predictions,
        labels=classes,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        predictions,
        labels=classes,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=classes,
        average="macro",
        zero_division=0
    )

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Balanced Accuracy:", round(balanced_accuracy * 100, 2), "%")
    print("Macro Precision:", round(macro_precision, 4))
    print("Macro Recall:", round(macro_recall, 4))
    print("Macro F1:", round(macro_f1, 4))

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=classes,
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=classes
        )
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1
    })

print("\n" + "=" * 75)
print("FINAL MODEL COMPARISON")
print("=" * 75)

results.sort(
    key=lambda item: item["Macro F1"],
    reverse=True
)

for result in results:

    print("\nModel:", result["Model"])

    for metric, value in result.items():

        if metric == "Model":
            continue

        print(
            metric + ":",
            round(value * 100, 2), "%"
        )

print("\nEvaluation completed successfully!")
