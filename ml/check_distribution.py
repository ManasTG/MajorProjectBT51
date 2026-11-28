
import numpy as np
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit

data = np.load("datasets/mitbih/heartbeat_dataset.npz")

X = data["X"]
y = data["y"]
record_ids = data["record_ids"]

splitter = GroupShuffleSplit(
    n_splits=100,
    test_size=0.20,
    random_state=42
)

classes = np.unique(y)

best_split = None
best_score = float("inf")

for train_idx, test_idx in splitter.split(X, y, groups=record_ids):

    if not all(label in y[train_idx] for label in classes):
        continue

    if not all(label in y[test_idx] for label in classes):
        continue

    train_dist = np.array([
        np.mean(y[train_idx] == label) for label in classes
    ])

    test_dist = np.array([
        np.mean(y[test_idx] == label) for label in classes
    ])

    score = np.sum(np.abs(train_dist - test_dist))

    if score < best_score:
        best_score = score
        best_split = (train_idx, test_idx)

train_idx, test_idx = best_split

print("CLASS DISTRIBUTION")
print("\nTraining:")

for label, count in Counter(y[train_idx]).items():
    print(label, ":", count)

print("\nTesting:")

for label, count in Counter(y[test_idx]).items():
    print(label, ":", count)

print("\nTesting Records:")
print(sorted(np.unique(record_ids[test_idx])))
