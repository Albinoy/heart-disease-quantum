"""Train a classical Random Forest and a TUNED quantum-kernel SVM on the UCI
heart disease dataset, then save everything the Streamlit app needs.

The quantum model chooses its angle scale and SVM C by 3-fold cross-validation
on the training rows only (the test set is never used for tuning), matching
the method in benchmark.py.

Usage:  python train.py          (expects heart.csv in the same folder)
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from data_prep import load_data

SEED = 42
N_QUBITS = 4      # = number of PCA components fed into the circuit
MAX_TRAIN = 120   # quantum kernel cost grows with n^2 (same as benchmark.py)
SCALES = [0.05, 0.1, 0.25, 0.5, 1.0]   # angle scale, as multiples of pi
C_GRID = [0.1, 1.0, 10.0]

df = load_data("heart.csv")
X = df.drop("target", axis=1)
y = df["target"]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)


def scores(y_true, y_pred, y_score, seconds):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
        "train_seconds": seconds,
    }


# ---------- Classical model ----------
t0 = time.time()
rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED)
rf.fit(X_train, y_train)
rf_time = time.time() - t0
rf_metrics = scores(y_test, rf.predict(X_test), rf.predict_proba(X_test)[:, 1], rf_time)
print("Random Forest:", rf_metrics)

# ---------- Quantum kernel model (tuned) ----------
if len(X_train) > MAX_TRAIN:
    X_q, _, y_q, _ = train_test_split(
        X_train, y_train, train_size=MAX_TRAIN, random_state=SEED, stratify=y_train
    )
else:
    X_q, y_q = X_train, y_train
y_q = np.asarray(y_q)

# Features scaled to [0, 1]; the angle scale then multiplies by scale * pi.
prep = Pipeline([
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=N_QUBITS, random_state=SEED)),
    ("unit", MinMaxScaler(feature_range=(0, 1))),
])
A1 = prep.fit_transform(X_q)
B1 = prep.transform(X_test)

kernel = FidelityQuantumKernel(feature_map=ZZFeatureMap(feature_dimension=N_QUBITS, reps=2))
inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

t0 = time.time()
kernels, best = {}, None
for scale in SCALES:
    print(f"  tuning: angle scale {scale}*pi ...", flush=True)
    K = kernel.evaluate(x_vec=A1 * scale * np.pi)
    kernels[scale] = K
    for C in C_GRID:
        aucs = []
        for tr, va in inner.split(A1, y_q):
            m = SVC(kernel="precomputed", C=C, class_weight="balanced")
            m.fit(K[np.ix_(tr, tr)], y_q[tr])
            aucs.append(roc_auc_score(y_q[va], m.decision_function(K[np.ix_(va, tr)])))
        auc = float(np.mean(aucs))
        if best is None or auc > best[0]:
            best = (auc, scale, C)

cv_auc, best_scale, best_C = best
print(f"Chosen: angle scale {best_scale}*pi, C = {best_C} (inner CV AUC {cv_auc:.3f})")

svc = SVC(kernel="precomputed", C=best_C, class_weight="balanced",
          probability=True, random_state=SEED)
svc.fit(kernels[best_scale], y_q)
Xq_train = A1 * best_scale * np.pi
q_time = time.time() - t0

K_test = kernel.evaluate(x_vec=B1 * best_scale * np.pi, y_vec=Xq_train)
q_metrics = scores(y_test, svc.predict(K_test), svc.decision_function(K_test), q_time)
print("Quantum kernel SVM (tuned):", q_metrics)

# ---------- Save ----------
joblib.dump(rf, "rf_model.joblib")
joblib.dump(
    {"prep": prep, "svc": svc, "Xq_train": Xq_train, "n_qubits": N_QUBITS,
     "angle_scale": best_scale, "C": best_C},
    "quantum_model.joblib",
)
with open("metrics.json", "w") as f:
    json.dump(
        {
            "Random Forest": rf_metrics,
            "Quantum kernel SVM": q_metrics,
            "features": feature_names,
            "angle_scale": best_scale,
            "svm_C": best_C,
            "n_train_classical": len(X_train),
            "n_train_quantum": len(X_q),
            "n_test": len(X_test),
        },
        f,
        indent=2,
    )
print("Saved rf_model.joblib, quantum_model.joblib, metrics.json")
