
import numpy as np

from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "datasets" / "mitbih" / "heartbeat_dataset.npz"

data = np.load(DATA_PATH)

y = data["y"]
record_ids = data["record_ids"]
splits = data["splits"]

ds1_mask = splits == "DS1"

records = record_ids[ds1_mask]
labels = y[ds1_mask]

print("DS1 RECORD-WISE CLASS DISTRIBUTION")
print("=" * 75)

for record in sorted(np.unique(records)):

    mask = records == record

    counts = Counter(labels[mask])

    print(
        f"{record}: "
        f"N={counts.get('N', 0):5d} | "
        f"S={counts.get('S', 0):4d} | "
        f"V={counts.get('V', 0):4d} | "
        f"F={counts.get('F', 0):4d} | "
        f"Q={counts.get('Q', 0):3d}"
    )

print("\nRARE CLASS RECORD COVERAGE")
print("=" * 75)

for target in ["S", "V", "F", "Q"]:

    print("\nClass:", target)

    for record in sorted(np.unique(records)):

        count = np.sum(
            (records == record) & (labels == target)
        )

        if count > 0:
            print(record, ":", count)

print("\nAnalysis completed!")
