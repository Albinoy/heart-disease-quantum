"""Leave-one-hospital-out validation: how well do CardiaQ's models work at a hospital
they have never seen?

train.py reports scores on a random 20% of patients, so test patients come from the same
hospitals as training patients. That measures "new patients from familiar hospitals". This
script measures "a new hospital": for each of the five hospitals, both models are trained
on the other four and tested on the held-out one, using exactly train.py's settings.

Hospitals: Cleveland, Hungary (Budapest), Switzerland (Zurich/Basel) and Long Beach VA from
data/uci_heart_disease.csv (its original "source" column), and Tehran from
data/z_alizadeh_sani.csv. Duplicates are removed as in data_prep.load_data.

Reported per held-out hospital and model: ROC AUC with a 95% bootstrap interval, Brier score,
mean estimate vs observed disease rate (calibration-in-the-large), and recall / specificity
at the app's 50% cut-off.

Usage:
    python validate_hospitals.py            # both models (~25 min; quantum is simulated)
    python validate_hospitals.py --rf-only  # classical model only (~1 min)
Writes results/hospital_validation.csv and results/hospital_validation.json.

Settings below copy train.py (proposal P6 would move them into one shared config).
"""
import argparse
import json
import time
import warnings
from datetime import date

import numpy as np
import pandas as pd
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from data_prep import _normalise

warnings.filterwarnings("ignore")

# --- same settings as train.py ---
SEED = 42
N_QUBITS = 4
MAX_TRAIN = 240
SCALES = [0.05, 0.1, 0.25, 0.5, 1.0]
C_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]
INCREASING = ["trestbps", "chol"]
N_BOOT = 1000

HOSPITAL_NAMES = {"cleveland": "Cleveland (USA)", "hungarian": "Budapest (Hungary)",
                  "switzerland": "Zurich/Basel (Switzerland)", "va": "Long Beach VA (USA)",
                  "tehran": "Tehran (Iran)"}


def load_with_hospitals(features):
    uci_raw = pd.read_csv("data/uci_heart_disease.csv")
    uci = _normalise(uci_raw, "uci_heart_disease.csv")
    uci["hospital"] = uci_raw["source"].str.strip().str.lower().values
    zas = _normalise(pd.read_csv("data/z_alizadeh_sani.csv"), "z_alizadeh_sani.csv")
    zas["hospital"] = "tehran"
    df = pd.concat([uci, zas], ignore_index=True, sort=False).reindex(columns=features + ["target", "hospital"])
    return df.drop_duplicates(subset=features + ["target"]).reset_index(drop=True)


def fit_rf(X, y, features):
    return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED,
                                  monotonic_cst=[1 if f in INCREASING else 0 for f in features]).fit(X, y)


def fit_predict_quantum(Xtr, ytr, Xte):
    """train.py's quantum pipeline: subsample, impute, PCA to 4, tune angle scale and C by
    3-fold CV on training rows, calibrate out-of-fold scores, predict probabilities."""
    if len(Xtr) > MAX_TRAIN:
        Xq, _, yq, _ = train_test_split(Xtr, ytr, train_size=MAX_TRAIN, random_state=SEED, stratify=ytr)
    else:
        Xq, yq = Xtr, ytr
    yq = np.asarray(yq)
    prep = Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("pca", PCA(n_components=N_QUBITS, random_state=SEED)),
                     ("unit", MinMaxScaler(feature_range=(0, 1)))])
    A, B = prep.fit_transform(Xq), prep.transform(Xte)
    kernel = FidelityQuantumKernel(feature_map=ZZFeatureMap(feature_dimension=N_QUBITS, reps=2))
    splits = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(A, yq))
    best, kernels = None, {}
    for scale in SCALES:
        K = kernel.evaluate(x_vec=A * scale * np.pi)
        kernels[scale] = K
        for C in C_GRID:
            oof = np.zeros(len(yq))
            for tr, va in splits:
                m = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(K[np.ix_(tr, tr)], yq[tr])
                oof[va] = m.decision_function(K[np.ix_(va, tr)])
            auc = roc_auc_score(yq, oof)
            if best is None or auc > best[0]:
                best = (auc, scale, C, oof)
    _, scale, C, oof = best
    svc = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(kernels[scale], yq)
    calib = LogisticRegression().fit(oof.reshape(-1, 1), yq)
    K_test = kernel.evaluate(x_vec=B * scale * np.pi, y_vec=A * scale * np.pi)
    return calib.predict_proba(svc.decision_function(K_test).reshape(-1, 1))[:, 1], scale, C


def bootstrap_auc(y, p, rng):
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarise(model, hospital, y, p, rng, extra=None):
    y, p = np.asarray(y), np.asarray(p)
    lo, hi = bootstrap_auc(y, p, rng)
    return {"model": model, "hospital": hospital, "n": int(len(y)), "disease_rate": float(y.mean()),
            "roc_auc": float(roc_auc_score(y, p)), "auc_ci_low": lo, "auc_ci_high": hi,
            "brier": float(brier_score_loss(y, p)), "mean_estimate": float(p.mean()),
            "recall": float(recall_score(y, p >= 0.5)),
            "specificity": float(((p < 0.5) & (y == 0)).sum() / max((y == 0).sum(), 1)),
            **(extra or {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf-only", action="store_true", help="skip the (slow) quantum model")
    args = ap.parse_args()
    features = json.load(open("metrics.json"))["features"]
    df = load_with_hospitals(features)
    rng = np.random.default_rng(SEED)
    rows = []
    for h in ["cleveland", "hungarian", "switzerland", "va", "tehran"]:
        tr, te = df[df.hospital != h], df[df.hospital == h]
        t0 = time.time()
        p_rf = fit_rf(tr[features], tr.target, features).predict_proba(te[features])[:, 1]
        rows.append(summarise("Random Forest", h, te.target, p_rf, rng))
        msg = f"{HOSPITAL_NAMES[h]:28s} n={len(te):3d}  classical AUC {rows[-1]['roc_auc']:.3f}"
        if not args.rf_only:
            p_q, scale, C = fit_predict_quantum(tr[features], tr.target, te[features])
            rows.append(summarise("Quantum kernel SVM", h, te.target, p_q, rng,
                                  {"angle_scale": scale, "svm_C": C}))
            msg += f"  quantum AUC {rows[-1]['roc_auc']:.3f} (scale {scale}*pi, C {C})"
        print(msg + f"  [{time.time() - t0:.0f}s]", flush=True)

    out = pd.DataFrame(rows)
    out["hospital_name"] = out["hospital"].map(HOSPITAL_NAMES)
    mean_auc = out.groupby("model")["roc_auc"].mean().to_dict()
    print("\nMean AUC across held-out hospitals:", {k: round(v, 3) for k, v in mean_auc.items()})
    if not args.rf_only:
        out.to_csv("results/hospital_validation.csv", index=False)
        json.dump({"date": str(date.today()), "features": features, "n_patients": int(len(df)),
                   "bootstrap_resamples": N_BOOT, "mean_auc": mean_auc,
                   "rows": out.to_dict(orient="records")},
                  open("results/hospital_validation.json", "w"), indent=2)
        print("Saved results/hospital_validation.csv and results/hospital_validation.json")
    else:
        print("(--rf-only: results not saved)")


if __name__ == "__main__":
    main()
