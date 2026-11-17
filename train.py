"""Train a classical Random Forest and a quantum-kernel SVM on the UCI heart
disease dataset, then save everything the Streamlit app needs.

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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from data_prep import load_data

SEED = 42
N_QUBITS = 4      # = number of PCA components fed into the circuit
MAX_TRAIN = 200   # quantum kernel cost grows with n^2, so cap the training size

df = load_data("heart.csv")
X = df.drop("target", axis=1)
y = df["target"]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)


def scores(y_true, y_pred, seconds):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "train_seconds": seconds,
    }


# ---------- Classical model ----------
t0 = time.time()
rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED)
rf.fit(X_train, y_train)
rf_time = time.time() - t0
rf_metrics = scores(y_test, rf.predict(X_test), rf_time)
print("Random Forest:", rf_metrics)

# ---------- Quantum kernel model ----------
# Use a stratified subset for the quantum model (kernel matrix is n x n).
if len(X_train) > MAX_TRAIN:
    X_q, _, y_q, _ = train_test_split(
        X_train, y_train, train_size=MAX_TRAIN, random_state=SEED, stratify=y_train
    )
else:
    X_q, y_q = X_train, y_train

prep = Pipeline([
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=N_QUBITS, random_state=SEED)),
    ("angles", MinMaxScaler(feature_range=(0, np.pi))),
])
Xq_train = prep.fit_transform(X_q)
Xq_test = prep.transform(X_test)

feature_map = ZZFeatureMap(feature_dimension=N_QUBITS, reps=2)
kernel = FidelityQuantumKernel(feature_map=feature_map)

t0 = time.time()
K_train = kernel.evaluate(x_vec=Xq_train)
svc = SVC(kernel="precomputed", class_weight="balanced", probability=True, random_state=SEED)
svc.fit(K_train, y_q)
q_time = time.time() - t0

K_test = kernel.evaluate(x_vec=Xq_test, y_vec=Xq_train)
q_metrics = scores(y_test, svc.predict(K_test), q_time)
print("Quantum kernel SVM:", q_metrics)

# ---------- Save ----------
joblib.dump(rf, "rf_model.joblib")
joblib.dump(
    {"prep": prep, "svc": svc, "Xq_train": Xq_train, "n_qubits": N_QUBITS},
    "quantum_model.joblib",
)
with open("metrics.json", "w") as f:
    json.dump(
        {
            "Random Forest": rf_metrics,
            "Quantum kernel SVM": q_metrics,
            "features": feature_names,
            "n_train_classical": len(X_train),
            "n_train_quantum": len(X_q),
            "n_test": len(X_test),
        },
        f,
        indent=2,
    )
print("Saved rf_model.joblib, quantum_model.joblib, metrics.json")
