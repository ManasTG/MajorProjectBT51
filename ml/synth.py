"""Synthetic ECG generator so the full pipeline runs without downloading MIT-BIH."""
import numpy as np
from .pipeline import FS


def synth_record(n_beats=300, seed=0, p_abn=0.2):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.7, 1.0)
    times, labels, t, pause = [], [], 1.0, False
    for _ in range(n_beats):
        lab = rng.choice(["N", "S", "V"], p=[1 - p_abn, p_abn / 2, p_abn / 2])
        rr = base * rng.normal(1, 0.03) * {"N": 1, "S": 0.7, "V": 0.75}[lab] * (1.3 if pause else 1)
        pause = lab == "V"
        t += rr; times.append(t); labels.append(lab)
    tt = np.arange(int((t + 1) * FS)) / FS
    g = lambda c, a, w: a * np.exp(-((tt - c) / w) ** 2 / 2)
    x = 0.1 * np.sin(2 * np.pi * 0.3 * tt) + rng.normal(0, 0.03, len(tt))
    for c, lab in zip(times, labels):
        if lab == "V":
            x += g(c, 1.4, 0.04) - g(c + 0.08, 0.3, 0.03) - g(c + 0.3, 0.4, 0.07)
        else:
            x += g(c - 0.16, 0.15 if lab == "N" else 0.06, 0.025) + g(c, 1.0, 0.012) - g(c + 0.03, 0.15, 0.01) + g(c + 0.25, 0.3, 0.05)
    return x, np.array([int(c * FS) for c in times]), np.array(labels)
