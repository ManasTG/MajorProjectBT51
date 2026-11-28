
import numpy as np
import joblib

from pathlib import Path
from collections import Counter

from imblearn.ensemble import BalancedRandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from ml.preprocessing import normalize_dataset

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "datasets" / "mitbih" / "validation_split.npz"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "arrhythmia_validation_rf.joblib"

print("Loading training and validation data...")

data = np.load(DATA_PATH)

X_train = data["X_train"]
y_train = data["y_train"]
rr_train = data["rr_train"]

X_val = data["X_val"]
y_val = data["y_val"]
rr_val = data["rr_val"]

print("\nTraining Distribution:")
print(Counter(y_train))

print("\nValidation Distribution:")
print(Counter(y_val))

print("\nNormalizing ECG heartbeats...")

X_train_normalized = normalize_dataset(X_train)
X_val_normalized = normalize_dataset(X_val)

X_train_combined = np.hstack(
    (X_train_normalized, rr_train)
)

X_val_combined = np.hstack(
    (X_val_normalized, rr_val)
)

print("\nTraining Balanced Random Forest...")

model = BalancedRandomForestClassifier(
    n_estimators=300,
    sampling_strategy="all",
    replacement=True,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2
)

model.fit(X_train_combined, y_train)

print("\nTraining completed!")

print("\nGenerating validation predictions...")

predictions = model.predict(X_val_combined)

accuracy = accuracy_score(y_val, predictions)

balanced_accuracy = balanced_accuracy_score(
    y_val,
    predictions
)

macro_f1 = f1_score(
    y_val,
    predictions,
    average="macro",
    zero_division=0
)

classes = ["N", "S", "V", "F", "Q"]

print("\nVALIDATION RESULTS")
print("=" * 55)

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Balanced Accuracy:", round(balanced_accuracy * 100, 2), "%")
print("Macro F1:", round(macro_f1, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_val,
        predictions,
        labels=classes,
        zero_division=0
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_val,
        predictions,
        labels=classes
    )
)

joblib.dump(
    {
        "model": model,
        "feature_count": X_train_combined.shape[1],
        "classes": classes
    },
    MODEL_PATH
)

print("\nModel saved:", MODEL_PATH)
print("Validation training experiment completed!")
