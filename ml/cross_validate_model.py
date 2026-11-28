
import numpy as np

from pathlib import Path
from collections import Counter

from imblearn.ensemble import BalancedRandomForestClassifier

from sklearn.model_selection import StratifiedGroupKFold

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

DATA_PATH = BASE_DIR / "datasets" / "mitbih" / "heartbeat_dataset.npz"

print("Loading MIT-BIH dataset...")

data = np.load(DATA_PATH)

X = data["X"]
y = data["y"]
rr_features = data["rr_features"]
record_ids = data["record_ids"]
splits = data["splits"]

ds1_mask = splits == "DS1"

X = X[ds1_mask]
y = y[ds1_mask]
rr_features = rr_features[ds1_mask]
record_ids = record_ids[ds1_mask]

classes = ["N", "S", "V", "F", "Q"]

print("\nDS1 Samples:", len(X))
print("DS1 Records:", len(np.unique(record_ids)))
print("Class Distribution:", Counter(y))

cv = StratifiedGroupKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

all_predictions = np.empty(len(y), dtype=object)

fold_number = 1

for train_idx, val_idx in cv.split(
    X,
    y,
    groups=record_ids
):

    print("\n" + "=" * 60)
    print("FOLD:", fold_number)
    print("=" * 60)

    X_train = X[train_idx]
    y_train = y[train_idx]
    rr_train = rr_features[train_idx]
    records_train = record_ids[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]
    rr_val = rr_features[val_idx]
    records_val = record_ids[val_idx]

    train_records = set(records_train)
    val_records = set(records_val)

    print("Training Records:", sorted(train_records))
    print("Validation Records:", sorted(val_records))

    print("Record Overlap:", train_records.intersection(val_records))

    print("Training Distribution:", Counter(y_train))
    print("Validation Distribution:", Counter(y_val))

    print("\nNormalizing ECG data...")

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

    predictions = model.predict(X_val_combined)

    all_predictions[val_idx] = predictions

    print("\nFold Accuracy:",
          round(accuracy_score(y_val, predictions) * 100, 2), "%")

    print("Fold Balanced Accuracy:",
          round(balanced_accuracy_score(y_val, predictions) * 100, 2), "%")

    print("Fold Macro F1:",
          round(f1_score(
              y_val,
              predictions,
              labels=classes,
              average="macro",
              zero_division=0
          ), 4))

    fold_number += 1

print("\n" + "=" * 65)
print("COMBINED OUT-OF-FOLD RESULTS")
print("=" * 65)

all_predictions = np.asarray(all_predictions)

print("Accuracy:",
      round(accuracy_score(y, all_predictions) * 100, 2), "%")

print("Balanced Accuracy:",
      round(balanced_accuracy_score(y, all_predictions) * 100, 2), "%")

print("Macro Precision:",
      round(precision_score(
          y,
          all_predictions,
          labels=classes,
          average="macro",
          zero_division=0
      ), 4))

print("Macro Recall:",
      round(recall_score(
          y,
          all_predictions,
          labels=classes,
          average="macro",
          zero_division=0
      ), 4))

print("Macro F1:",
      round(f1_score(
          y,
          all_predictions,
          labels=classes,
          average="macro",
          zero_division=0
      ), 4))

print("\nClassification Report:")

print(
    classification_report(
        y,
        all_predictions,
        labels=classes,
        zero_division=0
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y,
        all_predictions,
        labels=classes
    )
)

print("\nCross-validation completed successfully!")
