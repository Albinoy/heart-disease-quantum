"""Train CardiaQ's two models and save everything the app needs.

- Classical: Random Forest on all usable features.
- Quantum: quantum kernel SVM (simulated with Qiskit) on 4 PCA components.
  Its angle scale and C are chosen by 3-fold cross-validation on the
  training rows only, and its scores are turned into probabilities with a
  calibration step fitted on out-of-fold scores (no test data used).

Data: every .csv in data/ (combined, duplicates removed), or heart.csv.
Missing values (e.g. exercise-test results the Iranian dataset never recorded)
are left as gaps for the Random Forest, which handles them itself; the quantum
pipeline fills them with the training median, because PCA cannot use gaps.
The test split is stratified by data source and outcome, and test scores are
also reported for each source separately.
Usage:  python train.py
"""
import json
import time

import joblib
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from data_prep import load_data

SEED = 42
N_QUBITS = 4
# Quantum kernel cost grows with n^2. 240 rows (up from 120) was needed once the
# Iranian patients were added: at 120 the test AUC fell to 0.73 (0.59 on Iran).
MAX_TRAIN = 240
SCALES = [0.05, 0.1, 0.25, 0.5, 1.0]   # angle scale, multiples of pi
C_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]   # 1000 added to test the top of the range; 100 is still chosen
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]

df = load_data(verbose=True, with_source=True, fill_missing=False)
info = df.attrs["info"]
features = info["features"]
X, y, source = df[features], df["target"], df["source"]

X_train, X_test, y_train, y_test, src_train, src_test = train_test_split(
    X, y, source, test_size=0.2, random_state=SEED, stratify=source + "_" + y.astype(str)
)


def by_source(prob):
    """Test ROC AUC for each data source separately."""
    out = {}
    for s in sorted(src_test.unique()):
        m = (src_test == s).values
        out[s] = {"n_test": int(m.sum()), "roc_auc": float(roc_auc_score(y_test[m], prob[m]))}
    return out


def report(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, prob),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# ---------- Classical model ----------
t0 = time.time()
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED)
rf.fit(X_train, y_train)
rf_prob = rf.predict_proba(X_test)[:, 1]
rf_metrics = {**report(y_test, rf_prob), "train_seconds": time.time() - t0, "by_source": by_source(rf_prob)}
print("Classical (Random Forest):", {k: v for k, v in rf_metrics.items() if k != "confusion"})

# ---------- Quantum model (tuned + calibrated) ----------
if len(X_train) > MAX_TRAIN:
    X_q, _, y_q, _ = train_test_split(
        X_train, y_train, train_size=MAX_TRAIN, random_state=SEED, stratify=y_train
    )
else:
    X_q, y_q = X_train, y_train
y_q = np.asarray(y_q)

prep = Pipeline([
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=N_QUBITS, random_state=SEED)),
    ("unit", MinMaxScaler(feature_range=(0, 1))),
])
A1, B1 = prep.fit_transform(X_q), prep.transform(X_test)

kernel = FidelityQuantumKernel(feature_map=ZZFeatureMap(feature_dimension=N_QUBITS, reps=2))
splits = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(A1, y_q))

t0 = time.time()
kernels, best = {}, None
for scale in SCALES:
    print(f"  tuning quantum model: angle scale {scale}*pi", flush=True)
    K = kernel.evaluate(x_vec=A1 * scale * np.pi)
    kernels[scale] = K
    for C in C_GRID:
        oof = np.zeros(len(y_q))
        for tr, va in splits:
            m = SVC(kernel="precomputed", C=C, class_weight="balanced")
            m.fit(K[np.ix_(tr, tr)], y_q[tr])
            oof[va] = m.decision_function(K[np.ix_(va, tr)])
        auc = roc_auc_score(y_q, oof)
        if best is None or auc > best[0]:
            best = (auc, scale, C, oof)

cv_auc, best_scale, best_C, oof_scores = best
print(f"Chosen: angle scale {best_scale}*pi, C = {best_C} (inner CV AUC {cv_auc:.3f})")

svc = SVC(kernel="precomputed", C=best_C, class_weight="balanced").fit(kernels[best_scale], y_q)
calib = LogisticRegression().fit(oof_scores.reshape(-1, 1), y_q)   # scores -> probabilities
Xq_train = A1 * best_scale * np.pi
q_time = time.time() - t0

K_test = kernel.evaluate(x_vec=B1 * best_scale * np.pi, y_vec=Xq_train)
q_prob = calib.predict_proba(svc.decision_function(K_test).reshape(-1, 1))[:, 1]
q_metrics = {**report(y_test, q_prob), "train_seconds": q_time, "by_source": by_source(q_prob)}
print("Quantum kernel SVM:", {k: v for k, v in q_metrics.items() if k != "confusion"})

# ---------- Reference values for explaining a single prediction ----------
train_df = X_train.assign(target=y_train.values)
reference = {}
for f in features:
    col = train_df[f].dropna()
    typical = float(col.median()) if f in NUMERIC else float(col.mode().iloc[0])
    entry = {"typical": typical}
    if f in NUMERIC:
        entry["quantiles"] = [float(v) for v in np.quantile(col, np.linspace(0, 1, 101))]
        entry["median_healthy"] = float(train_df.loc[train_df.target == 0, f].median())
        entry["median_disease"] = float(train_df.loc[train_df.target == 1, f].median())
    reference[f] = entry

# ---------- Save ----------
joblib.dump(rf, "rf_model.joblib")
joblib.dump(
    {"prep": prep, "svc": svc, "calib": calib, "Xq_train": Xq_train,
     "n_qubits": N_QUBITS, "angle_scale": best_scale, "C": best_C},
    "quantum_model.joblib",
)
with open("metrics.json", "w") as f:
    json.dump(
        {
            "Random Forest": rf_metrics,
            "Quantum kernel SVM": q_metrics,
            "features": features,
            "reference": reference,
            "data": info,
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
