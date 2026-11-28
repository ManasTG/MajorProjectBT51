
import numpy as np

from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "datasets" / "mitbih" / "heartbeat_dataset.npz"

OUTPUT_PATH = BASE_DIR / "datasets" / "mitbih" / "validation_split.npz"

VALIDATION_RECORDS = [
    "201",
    "203",
    "207",
    "209",
    "223"
]

print("Loading MIT-BIH dataset...")

data = np.load(DATA_PATH)

X = data["X"]
y = data["y"]
rr_features = data["rr_features"]
record_ids = data["record_ids"]
splits = data["splits"]

ds1_mask = splits == "DS1"

X_ds1 = X[ds1_mask]
y_ds1 = y[ds1_mask]
rr_ds1 = rr_features[ds1_mask]
records_ds1 = record_ids[ds1_mask]

available_records = set(np.unique(records_ds1))

if not set(VALIDATION_RECORDS).issubset(available_records):
    raise ValueError("Some validation records are missing!")

validation_mask = np.isin(
    records_ds1,
    VALIDATION_RECORDS
)

training_mask = ~validation_mask

X_train = X_ds1[training_mask]
y_train = y_ds1[training_mask]
rr_train = rr_ds1[training_mask]
records_train = records_ds1[training_mask]

X_val = X_ds1[validation_mask]
y_val = y_ds1[validation_mask]
rr_val = rr_ds1[validation_mask]
records_val = records_ds1[validation_mask]

train_records = set(np.unique(records_train))
val_records = set(np.unique(records_val))

print("\nTRAINING DATA")
print("Samples:", len(X_train))
print("Records:", sorted(train_records))
print("Distribution:", Counter(y_train))

print("\nVALIDATION DATA")
print("Samples:", len(X_val))
print("Records:", sorted(val_records))
print("Distribution:", Counter(y_val))

print("\nRecord Overlap:", train_records.intersection(val_records))

np.savez_compressed(
    OUTPUT_PATH,
    X_train=X_train,
    y_train=y_train,
    rr_train=rr_train,
    records_train=records_train,
    X_val=X_val,
    y_val=y_val,
    rr_val=rr_val,
    records_val=records_val
)

print("\nValidation split saved:", OUTPUT_PATH)
print("Split creation completed successfully!")
