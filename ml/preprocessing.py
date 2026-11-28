
import numpy as np


def normalize_heartbeat(heartbeat):

    heartbeat = np.asarray(heartbeat, dtype=float)

    mean = np.mean(heartbeat)
    std = np.std(heartbeat)

    if std < 1e-8:
        return np.zeros_like(heartbeat)

    normalized = (heartbeat - mean) / std

    return normalized


def normalize_dataset(X):

    normalized_data = np.array([
        normalize_heartbeat(heartbeat)
        for heartbeat in X
    ])

    return normalized_data


def extract_features(X, sampling_rate=360):

    X = np.asarray(X, dtype=float)

    features = []

    for heartbeat in X:

        heartbeat = normalize_heartbeat(heartbeat)

        mean_value = np.mean(heartbeat)
        std_value = np.std(heartbeat)
        min_value = np.min(heartbeat)
        max_value = np.max(heartbeat)
        median_value = np.median(heartbeat)

        peak_index = np.argmax(np.abs(heartbeat))
        peak_value = heartbeat[peak_index]

        energy = np.sum(heartbeat ** 2)

        q25 = np.percentile(heartbeat, 25)
        q75 = np.percentile(heartbeat, 75)

        features.append([
            mean_value,
            std_value,
            min_value,
            max_value,
            median_value,
            peak_index,
            peak_value,
            energy,
            q25,
            q75
        ])

    return np.array(features)


if __name__ == "__main__":

    sample = np.random.randn(216)

    normalized = normalize_heartbeat(sample)

    features = extract_features([sample])

    print("Preprocessing Test Successful")
    print("Original Shape:", sample.shape)
    print("Normalized Shape:", normalized.shape)
    print("Feature Shape:", features.shape)
    print("Normalized Mean:", round(np.mean(normalized), 4))
    print("Normalized Std:", round(np.std(normalized), 4))
