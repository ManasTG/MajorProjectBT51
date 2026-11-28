
import numpy as np
import joblib

from pathlib import Path
from collections import Counter

from imblearn.ensemble import BalancedRandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)

from ml.preprocessing import normalize_dataset

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "datasets" / "mitbih" / "heartbeat_dataset.npz"

MODEL_PATH = BASE_DIR / "models" / "arrhythmia_balanced_rf.joblib"

print("Loading MIT-BIH dataset...")

data = np.load(DATASET_PATH)

X = data["X"]
rr_features = data["rr_features"]
y = data["y"]
splits = data["splits"]
record_ids = data["record_ids"]

print("Normalizing ECG heartbeats...")

X_normalized = normalize_dataset(X)

X_combined = np.hstack((X_normalized, rr_features))

train_mask = splits == "DS1"
test_mask = splits == "DS2"

X_train = X_combined[train_mask]
X_test = X_combined[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]

train_records = np.unique(record_ids[train_mask])
test_records = np.unique(record_ids[test_mask])

overlap = len(set(train_records) & set(test_records))

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Feature count:", X_combined.shape[1])
print("Records overlap:", overlap)

if overlap != 0:
    raise ValueError("Data leakage detected!")

print("\nTraining distribution:")
print(Counter(y_train))

print("\nTraining Balanced Random Forest...")

model = BalancedRandomForestClassifier(
    n_estimators=300,
    sampling_strategy="all",
    replacement=True,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2
)

model.fit(X_train, y_train)

print("\nTraining completed!")

predictions = model.predict(X_test)

classes = ["N", "S", "V", "F", "Q"]

print(
    "\nAccuracy:",
    round(accuracy_score(y_test, predictions) * 100, 2),
    "%"
)

print(
    "Balanced Accuracy:",
    round(balanced_accuracy_score(y_test, predictions) * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=classes,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=classes))

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "classes": np.array(classes),
        "input_length": 216,
        "rr_feature_count": 4,
        "feature_count": 220,
        "sampling_rate": 360,
        "training_protocol": "MIT-BIH DS1/DS2 inter-patient split",
        "feature_type": "Normalized ECG plus RR intervals",
        "balancing": "Balanced Random Forest internal class sampling"
    },
    MODEL_PATH
)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)
