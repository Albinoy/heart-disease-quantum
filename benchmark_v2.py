"""Second benchmark: can the quantum kernel be tuned further?

Usage:
    python benchmark_v2.py            # full run (can take 45-90 minutes)
    python benchmark_v2.py --quick    # small, fast run to check it works

Writes to results_v2/ (your original results/ folder is left untouched).

Ideas tested, all with NESTED cross-validation (every setting is chosen by an
inner CV on the training rows of each outer fold; the outer test fold is never
used for tuning):
  1. Feature choice: PCA (4 mixed components) vs the 4 most important ORIGINAL
     features (picked by Random Forest importance on the training fold only).
  2. A finer angle-scale grid, with 5-fold inner CV.
  3. A hybrid kernel: w * quantum kernel + (1 - w) * RBF kernel.

Models (all except the first use the same training rows):
  RF (13 feat)             strong classical reference
  RBF-SVM tuned (PCA4)     classical kernel, PCA features
  Quantum tuned (PCA4)     your current best quantum model
  RBF-SVM tuned (top-4)    classical kernel, selected features
  Quantum tuned (top-4)    quantum kernel, selected features
  Hybrid Q+RBF (top-4)     mix of both kernels, selected features
"""
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
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
N_Q = 60 if QUICK else 120          # training rows for the 4-feature models
N_SPLITS = 3 if QUICK else 5
N_REPEATS = 1 if QUICK else 2
N_INNER = 3 if QUICK else 5         # inner CV folds used for tuning
SCALES = [0.1, 0.3] if QUICK else [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75]  # x pi
C_GRID = [1.0, 10.0] if QUICK else [0.1, 1.0, 10.0, 100.0]
W_GRID = [0.5] if QUICK else [0.25, 0.5, 0.75]   # quantum share in hybrid kernel

M_RF = "RF (13 feat)"
M_SVM_P = "RBF-SVM tuned (PCA4)"
M_Q_P = "Quantum tuned (PCA4)"
M_SVM_T = "RBF-SVM tuned (top-4)"
M_Q_T = "Quantum tuned (top-4)"
M_H_T = "Hybrid Q+RBF (top-4)"
MODEL_ORDER = [M_RF, M_SVM_P, M_Q_P, M_SVM_T, M_Q_T, M_H_T]

OUT = "results_v2"
os.makedirs(OUT, exist_ok=True)

df = load_data(verbose=True)
FEATURES = [c for c in df.columns if c != "target"]
X = df[FEATURES].to_numpy(dtype=float)
y = df["target"].to_numpy()
print(f"Data: {len(df)} rows, {X.shape[1]} features, {y.mean():.0%} positive", flush=True)


# ---------------- Helpers ----------------
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


def cv_auc(K, ya, splits, C):
    """Mean inner-CV AUC of an SVM on a precomputed kernel matrix K."""
    aucs = []
    for tr, va in splits:
        svc = SVC(kernel="precomputed", C=C, class_weight="balanced")
        svc.fit(K[np.ix_(tr, tr)], ya[tr])
        aucs.append(roc_auc_score(ya[va], svc.decision_function(K[np.ix_(va, tr)])))
    return float(np.mean(aucs))


def tune_rbf(A, ya, B, seed):
    grid = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced", random_state=seed),
        {"C": C_GRID, "gamma": ["scale", 0.03, 0.1, 0.3, 1.0]},
        cv=StratifiedKFold(N_INNER, shuffle=True, random_state=seed),
        scoring="roc_auc",
    ).fit(A, ya)
    return grid.decision_function(B), grid.predict(B)


def tune_kernel_models(A1, ya, B1, seed, with_hybrid):
    """Tune the quantum kernel (and optionally a hybrid) on training rows only.

    A1, B1 are features in [0, 1]. The quantum angle scale multiplies them by
    scale * pi. Each scale's kernel is computed once; inner CV just slices it.
    Returns a dict of results for 'quantum' (and 'hybrid').
    """
    t0 = time.time()
    nq = A1.shape[1]
    kernel = FidelityQuantumKernel(feature_map=ZZFeatureMap(nq, reps=2))
    splits = list(StratifiedKFold(N_INNER, shuffle=True, random_state=seed).split(A1, ya))
    gamma = 1.0 / (nq * A1.var())
    K_rbf = rbf_kernel(A1, A1, gamma=gamma)
    K_rbf_test = rbf_kernel(B1, A1, gamma=gamma)

    Kq, best_q, best_h = {}, None, None
    for scale in SCALES:
        K = kernel.evaluate(x_vec=A1 * scale * np.pi)
        Kq[scale] = K
        for C in C_GRID:
            auc = cv_auc(K, ya, splits, C)
            if best_q is None or auc > best_q[0]:
                best_q = (auc, scale, C)
            if with_hybrid:
                for w in W_GRID:
                    auc_h = cv_auc(w * K + (1 - w) * K_rbf, ya, splits, C)
                    if best_h is None or auc_h > best_h[0]:
                        best_h = (auc_h, scale, C, w)

    test_cache = {}

    def test_kernel(scale):
        if scale not in test_cache:
            test_cache[scale] = kernel.evaluate(
                x_vec=B1 * scale * np.pi, y_vec=A1 * scale * np.pi
            )
        return test_cache[scale]

    results = {}
    _, scale, C = best_q
    svc = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(Kq[scale], ya)
    Kt = test_kernel(scale)
    results["quantum"] = dict(score=svc.decision_function(Kt), pred=svc.predict(Kt),
                              best_scale=scale, best_C=C)
    if with_hybrid:
        _, scale, C, w = best_h
        svc = SVC(kernel="precomputed", C=C, class_weight="balanced")
        svc.fit(w * Kq[scale] + (1 - w) * K_rbf, ya)
        Kt = w * test_kernel(scale) + (1 - w) * K_rbf_test
        results["hybrid"] = dict(score=svc.decision_function(Kt), pred=svc.predict(Kt),
                                 best_scale=scale, best_C=C, best_w=w)
    secs = time.time() - t0     # tuning is shared, so both quantum models report it
    for r in results.values():
        r["seconds"] = secs
    return results


def run_models(Xtr, ytr, Xte, yte, seed):
    """Train and score all six models on one split. Tuning never sees Xte."""
    out, scores = {}, {}

    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed)
    rf.fit(Xtr, ytr)
    s = rf.predict_proba(Xte)[:, 1]
    out[M_RF] = {**metrics(yte, rf.predict(Xte), s), "seconds": time.time() - t0}
    scores[M_RF] = s
    top = np.argsort(rf.feature_importances_)[::-1][:N_QUBITS]   # chosen on training data
    top_names = [FEATURES[i] for i in top]

    Xs, ys = subsample(Xtr, ytr, N_Q, seed)

    # --- PCA features ---
    prep = Pipeline([("scale", StandardScaler()),
                     ("pca", PCA(n_components=N_QUBITS, random_state=SEED)),
                     ("unit", MinMaxScaler())])
    A1, B1 = prep.fit_transform(Xs), prep.transform(Xte)
    t0 = time.time()
    s, p = tune_rbf(A1 * np.pi, ys, B1 * np.pi, seed)
    out[M_SVM_P] = {**metrics(yte, p, s), "seconds": time.time() - t0}
    scores[M_SVM_P] = s
    r = tune_kernel_models(A1, ys, B1, seed, with_hybrid=False)["quantum"]
    out[M_Q_P] = {**metrics(yte, r["pred"], r["score"]), "seconds": r["seconds"],
                  "best_scale": r["best_scale"], "best_C": r["best_C"]}
    scores[M_Q_P] = r["score"]

    # --- Selected original features ---
    prep = Pipeline([("scale", StandardScaler()), ("unit", MinMaxScaler())])
    A1, B1 = prep.fit_transform(Xs[:, top]), prep.transform(Xte[:, top])
    t0 = time.time()
    s, p = tune_rbf(A1 * np.pi, ys, B1 * np.pi, seed)
    out[M_SVM_T] = {**metrics(yte, p, s), "seconds": time.time() - t0}
    scores[M_SVM_T] = s
    res = tune_kernel_models(A1, ys, B1, seed, with_hybrid=True)
    q, h = res["quantum"], res["hybrid"]
    out[M_Q_T] = {**metrics(yte, q["pred"], q["score"]), "seconds": q["seconds"],
                  "best_scale": q["best_scale"], "best_C": q["best_C"],
                  "features": ", ".join(top_names)}
    out[M_H_T] = {**metrics(yte, h["pred"], h["score"]), "seconds": h["seconds"],
                  "best_scale": h["best_scale"], "best_C": h["best_C"], "best_w": h["best_w"]}
    scores[M_Q_T], scores[M_H_T] = q["score"], h["score"]
    return out, scores


# ---------------- Nested cross-validation ----------------
print("\n== Nested cross-validation ==", flush=True)
rows = []
cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
for i, (tr, te) in enumerate(cv.split(X, y)):
    res, _ = run_models(X[tr], y[tr], X[te], y[te], seed=SEED + i)
    for name, m in res.items():
        rows.append({"fold": i, "model": name, **m})
    print(f"  fold {i + 1}/{N_SPLITS * N_REPEATS} done", flush=True)

cv_df = pd.DataFrame(rows)
cv_df.to_csv(f"{OUT}/cv_folds.csv", index=False)
summary = cv_df.groupby("model")[["accuracy", "recall", "roc_auc", "seconds"]].agg(["mean", "std"])
summary = summary.reindex(MODEL_ORDER)
summary.to_csv(f"{OUT}/cv_summary.csv")
print("\n" + summary.round(3).to_string(), flush=True)

# What did tuning choose?
for m in [M_Q_P, M_Q_T, M_H_T]:
    sub = cv_df[cv_df["model"] == m]
    print(f"\n{m}: scale chosen {dict(sub['best_scale'].value_counts())}, "
          f"C chosen {dict(sub['best_C'].value_counts())}")
print(f"Hybrid quantum share w chosen: {dict(cv_df[cv_df['model'] == M_H_T]['best_w'].value_counts())}")
feat_counts = (cv_df[cv_df["model"] == M_Q_T]["features"].str.split(", ").explode().value_counts())
print(f"Features picked (count over folds): {dict(feat_counts)}", flush=True)

# Paired differences: the same folds, so the comparison is fairer than eyeballing means.
auc = cv_df.pivot(index="fold", columns="model", values="roc_auc")
pairs = [(M_Q_T, M_SVM_T), (M_Q_P, M_SVM_P), (M_Q_T, M_Q_P), (M_H_T, M_SVM_T), (M_H_T, M_Q_T)]
diffs = []
print("\nPaired AUC differences (A minus B, mean ± std across folds):")
for a, b in pairs:
    d = auc[a] - auc[b]
    diffs.append({"A": a, "B": b, "mean_diff": d.mean(), "std_diff": d.std(),
                  "A_better_in_folds": int((d > 0).sum()), "folds": len(d)})
    print(f"  {a} - {b}: {d.mean():+.3f} ± {d.std():.3f}  (A better in {(d > 0).sum()}/{len(d)} folds)")
pd.DataFrame(diffs).to_csv(f"{OUT}/paired_differences.csv", index=False)
print("  Rule of thumb: if |mean| is smaller than the std, treat it as no real difference.",
      flush=True)

fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
for ax, m in zip(axes, ["accuracy", "recall", "roc_auc"]):
    means, stds = summary[m]["mean"], summary[m]["std"]
    ax.bar(range(len(means)), means.values, yerr=stds.values, capsize=4)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(means.index, rotation=35, ha="right")
    ax.set_title(f"{m} (mean ± std, {N_SPLITS * N_REPEATS} folds)")
    ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(f"{OUT}/cv_comparison_v2.png", dpi=150)
plt.close()

print(f"\nDone. See the {OUT}/ folder.", flush=True)
