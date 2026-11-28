
import numpy as np
from collections import Counter, defaultdict

data = np.load("datasets/mitbih/heartbeat_dataset.npz")

y = data["y"]
record_ids = data["record_ids"]

record_classes = defaultdict(Counter)

for record, label in zip(record_ids, y):
    record_classes[str(record)][str(label)] += 1

print("RECORD-WISE F CLASS DISTRIBUTION\n")

for record in sorted(record_classes):
    counts = record_classes[record]

    if counts["F"] > 0:
        print(
            "Record:", record,
            "| F:", counts["F"],
            "| N:", counts["N"],
            "| S:", counts["S"],
            "| V:", counts["V"],
            "| Q:", counts["Q"]
        )

print("\nTotal F samples:", sum(
    counts["F"] for counts in record_classes.values()
))
