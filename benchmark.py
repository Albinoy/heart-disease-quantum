"""Fair benchmark of classical vs quantum models on the heart disease data.

Usage:
    python benchmark.py            # full run (can take 45-90 minutes on a laptop)
    python benchmark.py --quick    # small, fast run to check everything works

Outputs go in the results/ folder (CSV tables and PNG charts).

Part A: repeated cross-validation of six models. Everything except the first
        model gets the SAME 4 PCA features and the SAME training rows:
          1. Random Forest, all features (11), all rows
          2. Random Forest, 4 PCA features
          3. RBF-SVM, default settings
          4. RBF-SVM, tuned (C and gamma, chosen by inner CV on training rows)
          5. Quantum kernel SVM, default settings (angle scale = pi, C = 1)
          6. Quantum kernel SVM, tuned (angle scale and C, chosen by inner CV
             on training rows only)
        Compare 4 vs 6 for the fairest classical-vs-quantum test: both get a
        similar tuning budget, and neither sees the test fold while tuning.
Part B: ROC curves on one held-out test split (tuning still uses training
        rows only).
Part C: quantum experiments (angle scale, feature map, qubits, training size,
        SVM C). These are scored on a VALIDATION split carved out of the
        training data, so the test set is never used to pick settings.
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
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
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
SCALES = [0.1, 0.5] if QUICK else [0.05, 0.1, 0.25, 0.5, 1.0]   # multiples of pi
C_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]   # same range as train.py

M_RF = "RF (all features, all rows)"
M_RF4 = "RF (4 PCA feat)"
M_SVM = "RBF-SVM (default)"
M_SVMT = "RBF-SVM (tuned)"
M_Q = "Quantum SVM (default)"
M_QT = "Quantum SVM (tuned)"
MODEL_ORDER = [M_RF, M_RF4, M_SVM, M_SVMT, M_Q, M_QT]

os.makedirs("results", exist_ok=True)

df = load_data(verbose=True)
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


def make_fm(nq, kind="zz", reps=2):
    if kind == "zz":
        return ZZFeatureMap(nq, reps=reps)
    # A genuinely different map (ZZFeatureMap is already the Z,ZZ Pauli map)
    return PauliFeatureMap(nq, reps=reps, paulis=["Y", "ZZ"])


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


def calibrated_preds(oof_scores, y_train, test_scores):
    """Yes/no predictions from a calibrated probability >= 0.5.

    A raw SVM cut-off (decision value 0) can put every patient on one side when
    training data is small, giving recall 0 even with a good AUC. Instead, fit a
    logistic curve (Platt scaling) on out-of-fold scores from the TRAINING rows
    only, as train.py does for the app's quantum model. AUC is unaffected.
    """
    lr = LogisticRegression().fit(np.asarray(oof_scores).reshape(-1, 1), y_train)
    return (lr.predict_proba(np.asarray(test_scores).reshape(-1, 1))[:, 1] >= 0.5).astype(int)


def oof_precomputed(K, y, C, inner):
    """Out-of-fold decision scores for an SVM on a precomputed kernel matrix."""
    oof = np.zeros(len(y))
    for tr, va in inner.split(K, y):
        svc = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(K[np.ix_(tr, tr)], y[tr])
        oof[va] = svc.decision_function(K[np.ix_(va, tr)])
    return oof


def fit_quantum(A, ya, B, feature_map, C=1.0, seed=SEED):
    """Quantum kernel SVM with fixed settings. Returns (scores, preds, seconds)."""
    t0 = time.time()
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    K_train = kernel.evaluate(x_vec=A)
    svc = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(K_train, ya)
    K_test = kernel.evaluate(x_vec=B, y_vec=A)
    s = svc.decision_function(K_test)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    preds = calibrated_preds(oof_precomputed(K_train, ya, C, inner), ya, s)
    return s, preds, time.time() - t0


def fit_quantum_tuned(A1, ya, B1, seed, nq=N_QUBITS):
    """Choose angle scale and C by 3-fold CV on the TRAINING rows only.

    A1, B1 are features scaled to [0, 1]; the angle scale multiplies them by
    scale*pi. The kernel is computed once per scale, and the inner CV just
    slices that matrix, so tuning C is free.
    Returns (scores, preds, seconds, best_scale, best_C).
    """
    t0 = time.time()
    kernel = FidelityQuantumKernel(feature_map=make_fm(nq))
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    kernels, best = {}, None
    for scale in SCALES:
        K = kernel.evaluate(x_vec=A1 * scale * np.pi)
        kernels[scale] = K
        for C in C_GRID:
            aucs = []
            for tr, va in inner.split(A1, ya):
                svc = SVC(kernel="precomputed", C=C, class_weight="balanced")
                svc.fit(K[np.ix_(tr, tr)], ya[tr])
                aucs.append(roc_auc_score(ya[va], svc.decision_function(K[np.ix_(va, tr)])))
            m = float(np.mean(aucs))
            if best is None or m > best[0]:
                best = (m, scale, C)
    _, scale, C = best
    svc = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(kernels[scale], ya)
    K_test = kernel.evaluate(x_vec=B1 * scale * np.pi, y_vec=A1 * scale * np.pi)
    s = svc.decision_function(K_test)
    preds = calibrated_preds(oof_precomputed(kernels[scale], ya, C, inner), ya, s)
    return s, preds, time.time() - t0, scale, C


def run_models(Xtr, ytr, Xte, yte, seed):
    """Train and score all six models on one split. Tuning never sees Xte."""
    out, scores = {}, {}

    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed)
    rf.fit(Xtr, ytr)
    s = rf.predict_proba(Xte)[:, 1]
    out[M_RF] = {**metrics(yte, rf.predict(Xte), s), "seconds": time.time() - t0}
    scores[M_RF] = s

    # Everything below shares the same rows and the same 4 features.
    Xs, ys = subsample(Xtr, ytr, N_Q, seed)
    prep = make_prep(N_QUBITS, scale=1 / np.pi)      # features in [0, 1]
    A1, B1 = prep.fit_transform(Xs), prep.transform(Xte)
    A, B = A1 * np.pi, B1 * np.pi                    # angles in [0, pi]

    t0 = time.time()
    rf4 = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed)
    rf4.fit(A, ys)
    s = rf4.predict_proba(B)[:, 1]
    out[M_RF4] = {**metrics(yte, rf4.predict(B), s), "seconds": time.time() - t0}
    scores[M_RF4] = s

    t0 = time.time()
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    svm = SVC(kernel="rbf", class_weight="balanced", random_state=seed).fit(A, ys)
    s = svm.decision_function(B)
    oof = cross_val_predict(clone(svm), A, ys, cv=inner, method="decision_function")
    out[M_SVM] = {**metrics(yte, calibrated_preds(oof, ys, s), s), "seconds": time.time() - t0}
    scores[M_SVM] = s

    t0 = time.time()
    grid = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced", random_state=seed),
        {"C": C_GRID, "gamma": ["scale", 0.03, 0.1, 0.3, 1.0]},
        cv=3, scoring="roc_auc",
    ).fit(A, ys)
    s = grid.decision_function(B)
    oof = cross_val_predict(clone(grid.best_estimator_), A, ys, cv=inner, method="decision_function")
    out[M_SVMT] = {**metrics(yte, calibrated_preds(oof, ys, s), s), "seconds": time.time() - t0}
    scores[M_SVMT] = s

    s, p, secs = fit_quantum(A, ys, B, make_fm(N_QUBITS), seed=seed)
    out[M_Q] = {**metrics(yte, p, s), "seconds": secs}
    scores[M_Q] = s

    s, p, secs, best_scale, best_C = fit_quantum_tuned(A1, ys, B1, seed)
    out[M_QT] = {**metrics(yte, p, s), "seconds": secs,
                 "best_scale": best_scale, "best_C": best_C}
    scores[M_QT] = s
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

chosen = cv_df[cv_df["model"] == M_QT][["best_scale", "best_C"]]
print("\nScale chosen by tuning (multiples of pi):", dict(chosen["best_scale"].value_counts()))
print("C chosen by tuning:", dict(chosen["best_C"].value_counts()), flush=True)

fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
for ax, m in zip(axes, ["accuracy", "recall", "roc_auc"]):
    means, stds = summary[m]["mean"], summary[m]["std"]
    ax.bar(range(len(means)), means.values, yerr=stds.values, capsize=4)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(means.index, rotation=35, ha="right")
    ax.set_title(f"{m} (mean ± std, {N_SPLITS * N_REPEATS} folds)")
    ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("results/cv_comparison.png", dpi=150)
plt.close()

# ---------------- Part B: ROC curves on one test split ----------------
print("\n== Part B: ROC curves ==", flush=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
_, scores = run_models(Xtr, ytr, Xte, yte, seed=SEED)
plt.figure(figsize=(6.5, 5.5))
for name in MODEL_ORDER:
    fpr, tpr, _ = roc_curve(yte, scores[name])
    plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(yte, scores[name]):.2f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate (recall)")
plt.title("ROC curves (single held-out test split)")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig("results/roc_curves.png", dpi=150)
plt.close()

# ---------------- Part C: quantum experiments (validation split) ----------------
print("\n== Part C: quantum experiments (validation split, test set untouched) ==", flush=True)
Xfit, Xval, yfit, yval = train_test_split(
    Xtr, ytr, test_size=0.25, random_state=SEED, stratify=ytr
)


def quantum_run(nq=N_QUBITS, n=N_Q, scale=1.0, C=1.0, fmap="zz", reps=2):
    Xs, ys = subsample(Xfit, yfit, n, SEED)
    prep = make_prep(nq, scale)
    A, B = prep.fit_transform(Xs), prep.transform(Xval)
    s, p, secs = fit_quantum(A, ys, B, make_fm(nq, fmap, reps), C)
    return {**metrics(yval, p, s), "seconds": secs}


total = (len(SCALES) + 1) + 4 + len(QUBIT_COUNTS) + len(TRAIN_SIZES) + len(C_GRID)
exp_rows, counter = [], [0]


def run_experiment(group, label, **kwargs):
    r = quantum_run(**kwargs)
    exp_rows.append({"experiment": group, "setting": label, **r})
    counter[0] += 1
    print(f"  [{counter[0]}/{total}] {group}={label}: acc {r['accuracy']:.3f}, "
          f"recall {r['recall']:.3f}, auc {r['roc_auc']:.3f}, {r['seconds']:.0f}s", flush=True)
    return r


# Angle scale first; its best value becomes the base setting for the rest, so
# the other experiments are not confounded by a badly chosen scale.
scale_results = {s: run_experiment("angle_scale", f"{s}*pi", scale=s) for s in SCALES + [2.0]}
base_scale = max(scale_results, key=lambda s: scale_results[s]["roc_auc"])
print(f"  -> base angle scale for remaining experiments: {base_scale}*pi", flush=True)

for reps in (1, 2, 3):
    run_experiment("feature_map", f"ZZ reps={reps}", scale=base_scale, reps=reps)
run_experiment("feature_map", "Pauli(Y,ZZ) reps=2", scale=base_scale, fmap="pauli", reps=2)
for nq in QUBIT_COUNTS:
    run_experiment("qubits", str(nq), scale=base_scale, nq=nq)
for n in TRAIN_SIZES:
    run_experiment("train_size", str(n), scale=base_scale, n=n)
for C in C_GRID:
    run_experiment("svm_C", str(C), scale=base_scale, C=C)

exp_df = pd.DataFrame(exp_rows)
exp_df.to_csv("results/quantum_experiments.csv", index=False)

for group, sub in exp_df.groupby("experiment", sort=False):
    ax = sub.set_index("setting")[["accuracy", "recall", "roc_auc"]].plot.bar(
        figsize=(6, 4), ylim=(0, 1), rot=20
    )
    ax.set_title(f"Quantum kernel SVM: effect of {group} (validation)")
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
