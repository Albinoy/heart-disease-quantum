"""Fair benchmark of classical vs quantum models on the heart disease data.

Usage:
    python benchmark.py            # full run (can take 15-40 minutes on a laptop)
    python benchmark.py --quick    # small, fast run to check everything works

Outputs go in the results/ folder (CSV tables and PNG charts).

Part A: repeated cross-validation of four models, all given the SAME 4 PCA
        features and the SAME training rows (except the first RF baseline):
          1. Random Forest, 13 features, all rows   (strong classical baseline)
          2. Random Forest, 4 PCA features
          3. RBF-kernel SVM, 4 PCA features         (classical kernel)
          4. Quantum kernel SVM, 4 qubits           (quantum kernel)
        Comparing 3 vs 4 isolates the one thing being tested: classical kernel
        vs quantum kernel, with everything else held equal.
Part B: ROC curves on one held-out split.
Part C: quantum experiments (feature map, qubits, training size, angle scaling,
        SVM C) on one held-out split.
"""
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit.circuit.library import PauliFeatureMap, ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from data_prep import load_data

# ---------------- Settings ----------------
QUICK = "--quick" in sys.argv
SEED = 42
N_QUBITS = 4
N_Q = 60 if QUICK else 120          # training rows given to the 4-feature models
N_SPLITS = 3 if QUICK else 5
N_REPEATS = 1 if QUICK else 2
TRAIN_SIZES = [20, 40, 60] if QUICK else [40, 80, 120, 200]
QUBIT_COUNTS = [2, 4] if QUICK else [2, 4, 6, 8]

M_RF = "RF (13 feat, all rows)"
M_RF4 = "RF (4 PCA feat)"
M_SVM = "RBF-SVM (4 PCA feat)"
M_Q = "Quantum kernel SVM"
MODEL_ORDER = [M_RF, M_RF4, M_SVM, M_Q]

os.makedirs("results", exist_ok=True)

df = load_data("heart.csv")
X = df.drop(columns="target").to_numpy(dtype=float)
y = df["target"].to_numpy()
print(f"Data: {len(df)} rows, {X.shape[1]} features, {y.mean():.0%} positive", flush=True)


# ---------------- Helpers ----------------
def make_prep(n_components, scale=1.0):
    """Standardise -> PCA -> scale to [0, scale*pi] (angles for the circuit)."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=SEED)),
        ("angles", MinMaxScaler(feature_range=(0, scale * np.pi))),
    ])


def subsample(Xa, ya, n, seed):
    if len(Xa) <= n:
        return Xa, ya
    Xs, _, ys, _ = train_test_split(Xa, ya, train_size=n, random_state=seed, stratify=ya)
    return Xs, ys


def metrics(y_true, y_pred, score):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, score),
    }


def fit_quantum(A, ya, B, feature_map, C=1.0):
    """Quantum kernel SVM. Returns (scores, predictions, seconds)."""
    t0 = time.time()
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    K_train = kernel.evaluate(x_vec=A)
    svc = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(K_train, ya)
    K_test = kernel.evaluate(x_vec=B, y_vec=A)
    return svc.decision_function(K_test), svc.predict(K_test), time.time() - t0


def run_models(Xtr, ytr, Xte, yte, seed):
    """Train and score all four models on one split."""
    out, scores = {}, {}

    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed)
    rf.fit(Xtr, ytr)
    s = rf.predict_proba(Xte)[:, 1]
    out[M_RF] = {**metrics(yte, rf.predict(Xte), s), "seconds": time.time() - t0}
    scores[M_RF] = s

    # Everything below shares the same rows and the same 4 features.
    Xs, ys = subsample(Xtr, ytr, N_Q, seed)
    prep = make_prep(N_QUBITS)
    A, B = prep.fit_transform(Xs), prep.transform(Xte)

    t0 = time.time()
    rf4 = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed)
    rf4.fit(A, ys)
    s = rf4.predict_proba(B)[:, 1]
    out[M_RF4] = {**metrics(yte, rf4.predict(B), s), "seconds": time.time() - t0}
    scores[M_RF4] = s

    t0 = time.time()
    svm = SVC(kernel="rbf", class_weight="balanced", random_state=seed).fit(A, ys)
    s = svm.decision_function(B)
    out[M_SVM] = {**metrics(yte, svm.predict(B), s), "seconds": time.time() - t0}
    scores[M_SVM] = s

    s, p, secs = fit_quantum(A, ys, B, ZZFeatureMap(N_QUBITS, reps=2))
    out[M_Q] = {**metrics(yte, p, s), "seconds": secs}
    scores[M_Q] = s
    return out, scores


# ---------------- Part A: repeated cross-validation ----------------
print("\n== Part A: cross-validation ==", flush=True)
rows = []
cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
for i, (tr, te) in enumerate(cv.split(X, y)):
    res, _ = run_models(X[tr], y[tr], X[te], y[te], seed=SEED + i)
    for name, m in res.items():
        rows.append({"fold": i, "model": name, **m})
    print(f"  fold {i + 1}/{N_SPLITS * N_REPEATS} done", flush=True)

cv_df = pd.DataFrame(rows)
cv_df.to_csv("results/cv_folds.csv", index=False)
summary = cv_df.groupby("model")[["accuracy", "recall", "roc_auc", "seconds"]].agg(["mean", "std"])
summary = summary.reindex(MODEL_ORDER)
summary.to_csv("results/cv_summary.csv")
print("\n" + summary.round(3).to_string(), flush=True)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
for ax, m in zip(axes, ["accuracy", "recall", "roc_auc"]):
    means, stds = summary[m]["mean"], summary[m]["std"]
    ax.bar(range(len(means)), means.values, yerr=stds.values, capsize=4)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(means.index, rotation=30, ha="right")
    ax.set_title(f"{m} (mean ± std over {N_SPLITS * N_REPEATS} folds)")
    ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("results/cv_comparison.png", dpi=150)
plt.close()

# ---------------- Part B: ROC curves on one split ----------------
print("\n== Part B: ROC curves ==", flush=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
_, scores = run_models(Xtr, ytr, Xte, yte, seed=SEED)
plt.figure(figsize=(6, 5))
for name in MODEL_ORDER:
    fpr, tpr, _ = roc_curve(yte, scores[name])
    plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(yte, scores[name]):.2f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate (recall)")
plt.title("ROC curves (single held-out split)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("results/roc_curves.png", dpi=150)
plt.close()


# ---------------- Part C: quantum experiments ----------------
print("\n== Part C: quantum experiments (single split) ==", flush=True)


def quantum_run(nq=N_QUBITS, n=N_Q, scale=1.0, C=1.0, fmap="zz", reps=2):
    Xs, ys = subsample(Xtr, ytr, n, SEED)
    prep = make_prep(nq, scale)
    A, B = prep.fit_transform(Xs), prep.transform(Xte)
    if fmap == "zz":
        fm = ZZFeatureMap(nq, reps=reps)
    else:
        fm = PauliFeatureMap(nq, reps=reps, paulis=["Z", "ZZ"])
    s, p, secs = fit_quantum(A, ys, B, fm, C)
    return {**metrics(yte, p, s), "seconds": secs}


experiments = []
for reps in (1, 2, 3):
    experiments.append(("feature_map", f"ZZ reps={reps}", dict(fmap="zz", reps=reps)))
experiments.append(("feature_map", "Pauli(Z,ZZ) reps=2", dict(fmap="pauli", reps=2)))
for nq in QUBIT_COUNTS:
    experiments.append(("qubits", str(nq), dict(nq=nq)))
for n in TRAIN_SIZES:
    experiments.append(("train_size", str(n), dict(n=n)))
for scale in (0.25, 0.5, 1.0, 2.0):
    experiments.append(("angle_scale", f"{scale}*pi", dict(scale=scale)))
for C in (0.1, 1.0, 10.0):
    experiments.append(("svm_C", str(C), dict(C=C)))

exp_rows = []
for k, (group, label, kwargs) in enumerate(experiments):
    r = quantum_run(**kwargs)
    exp_rows.append({"experiment": group, "setting": label, **r})
    print(f"  [{k + 1}/{len(experiments)}] {group}={label}: "
          f"acc {r['accuracy']:.3f}, recall {r['recall']:.3f}, "
          f"auc {r['roc_auc']:.3f}, {r['seconds']:.0f}s", flush=True)

exp_df = pd.DataFrame(exp_rows)
exp_df.to_csv("results/quantum_experiments.csv", index=False)

for group, sub in exp_df.groupby("experiment", sort=False):
    ax = sub.set_index("setting")[["accuracy", "recall", "roc_auc"]].plot.bar(
        figsize=(6, 4), ylim=(0, 1), rot=20
    )
    ax.set_title(f"Quantum kernel SVM: effect of {group}")
    plt.tight_layout()
    plt.savefig(f"results/exp_{group}.png", dpi=150)
    plt.close()

q_sub = exp_df[exp_df["experiment"] == "qubits"]
plt.figure(figsize=(6, 4))
plt.plot(q_sub["setting"], q_sub["seconds"], marker="o")
plt.yscale("log")
plt.xlabel("Number of qubits")
plt.ylabel("Seconds (log scale)")
plt.title("Simulation time vs qubits")
plt.tight_layout()
plt.savefig("results/exp_qubits_time.png", dpi=150)
plt.close()

print("\nDone. See the results/ folder.", flush=True)
