"""Train the beat classifier.  python -m ml.train --source synthetic|mitdb"""
import argparse, json
from pathlib import Path
import joblib, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from .pipeline import clean, beat_features, CLASSES

AAMI = {**dict.fromkeys("NLRej", "N"), **dict.fromkeys(["A", "a", "J", "S"], "S"), **dict.fromkeys("VE", "V"),
        "F": "F", **dict.fromkeys(["/", "f", "Q"], "Q")}
# Standard inter-patient split (de Chazal et al.): train and test on different patients
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]


def from_signal(x, peaks, labels):
    X, used = beat_features(clean(x), peaks)
    return X, labels[used]


def load_mitdb(records, d="data/mitdb"):
    import wfdb
    if not Path(d).exists():
        wfdb.dl_database("mitdb", d)
    Xs, ys = [], []
    for r in records:
        rec, ann = wfdb.rdrecord(f"{d}/{r}", channels=[0]), wfdb.rdann(f"{d}/{r}", "atr")
        keep = [i for i, s in enumerate(ann.symbol) if s in AAMI]
        peaks = ann.sample[keep]
        labels = np.array([AAMI[ann.symbol[i]] for i in keep])
        X, y = from_signal(rec.p_signal[:, 0], peaks, labels)
        Xs.append(X); ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def load_synth(seeds):
    from .synth import synth_record
    parts = [from_signal(*synth_record(seed=s)) for s in seeds]
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--source", default="synthetic", choices=["synthetic", "mitdb"])
    src = ap.parse_args().source
    Xtr, ytr, Xte, yte = (*load_mitdb(DS1), *load_mitdb(DS2)) if src == "mitdb" else (*load_synth(range(20)), *load_synth(range(100, 106)))
    model = RandomForestClassifier(300, class_weight="balanced_subsample", n_jobs=-1, random_state=0).fit(Xtr, ytr)
    pred = model.predict(Xte)
    labels = [c for c in CLASSES if c in set(yte) | set(pred)]
    rep = classification_report(yte, pred, labels=labels, output_dict=True, zero_division=0)
    print(classification_report(yte, pred, labels=labels, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):", labels); print(confusion_matrix(yte, pred, labels=labels))
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/arrhythmia_rf.joblib")
    json.dump({"source": src, "report": rep}, open("models/metrics.json", "w"), indent=1)
