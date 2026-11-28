
import numpy as np
from collections import Counter
from pathlib import Path

from ml.heartbeat_extraction import extract_heartbeats

DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets" / "mitbih"

DS1 = [
    "101", "106", "108", "109", "112", "114",
    "115", "116", "118", "119", "122", "124",
    "201", "203", "205", "207", "208", "209",
    "215", "220", "223", "230"
]

DS2 = [
    "100", "103", "105", "111", "113", "117",
    "121", "123", "200", "202", "210", "212",
    "213", "214", "219", "221", "222", "228",
    "231", "232", "233", "234"
]

all_heartbeats = []
all_labels = []
all_rr_features = []
all_records = []
all_splits = []

print("Preparing MIT-BIH DS1/DS2 dataset with RR features...")

for split_name, records in [("DS1", DS1), ("DS2", DS2)]:

    print(f"\nProcessing {split_name}...")

    for record_name in records:

        try:
            X, y, rr = extract_heartbeats(record_name)

            if len(X) == 0:
                print("No heartbeats:", record_name)
                continue

            all_heartbeats.append(X)
            all_labels.extend(y)
            all_rr_features.append(rr)
            all_records.extend([record_name] * len(y))
            all_splits.extend([split_name] * len(y))

            print(record_name, ":", len(X), "heartbeats")

        except Exception as error:
            print("Error:", record_name, error)
            raise

X = np.vstack(all_heartbeats)
y = np.array(all_labels)
rr_features = np.vstack(all_rr_features)
record_ids = np.array(all_records)
splits = np.array(all_splits)

output_path = DATASET_DIR / "heartbeat_dataset.npz"

np.savez_compressed(
    output_path,
    X=X,
    y=y,
    rr_features=rr_features,
    record_ids=record_ids,
    splits=splits
)

print("\nDataset Preparation Completed")
print("Total Heartbeats:", len(X))
print("Heartbeat Shape:", X.shape)
print("RR Features Shape:", rr_features.shape)
print("Unique Records:", len(np.unique(record_ids)))

for split_name in ["DS1", "DS2"]:

    mask = splits == split_name

    print(f"\n{split_name} Distribution:")
    print("Total:", np.sum(mask))

    for label, count in Counter(y[mask]).items():
        print(label, ":", count)

print("\nSaved:", output_path)
