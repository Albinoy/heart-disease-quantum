"""Check freshly trained models before they replace the live ones.

Usage:
    python check_model.py [--baseline path/to/old/metrics.json]
Reads rf_model.joblib, quantum_model.joblib and metrics.json in this folder.
Exit code: 0 = PASS or WARN, 1 = FAIL.

Checks
  1. Neither model predicts one class for everyone (recall and precision > 0).
  2. Test ROC AUC is reasonable overall and for every data source.
  3. Known risk factors move the classical estimate the medically expected way
     for a typical patient (exercise angina, ST depression, low peak heart rate,
     older age, male sex all raise it).
  4. Optional: no data source got clearly worse than in --baseline.
"""
import argparse
import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

warnings.filterwarnings("ignore")

BASE = dict(age=55, sex=1, cp=3, trestbps=130, chol=240, fbs=0, restecg=0,
            thalach=150, exang=0, oldpeak=1.0, slope=1, ca=0, thal=2)
EXPECT_UP = [("exercise angina", dict(exang=1)), ("ST depression 3.0", dict(oldpeak=3.0)),
             ("max heart rate 110", dict(thalach=110)), ("age 70", dict(age=70))]
EXPECT_DOWN = [("female instead of male", dict(sex=0))]
MIN_AUC, MAX_DROP = 0.65, 0.03

results = []


def record(level, check, message):
    results.append((level, check, message))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", help="metrics.json from the previous models")
    a = ap.parse_args()

    m = json.load(open("metrics.json"))
    rf, q = joblib.load("rf_model.joblib"), joblib.load("quantum_model.joblib")
    feats = m["features"]

    for name in ("Random Forest", "Quantum kernel SVM"):
        r = m[name]
        level = "FAIL" if r["recall"] == 0 or r["precision"] == 0 else "PASS"
        record(level, f"{name}: predicts both classes", f"recall {r['recall']:.2f}, precision {r['precision']:.2f}")
        aucs = {"overall": r["roc_auc"], **{s: v["roc_auc"] for s, v in r.get("by_source", {}).items()}}
        low = {s: v for s, v in aucs.items() if v < MIN_AUC}
        record("FAIL" if low else "PASS", f"{name}: ROC AUC",
               ", ".join(f"{s} {v:.3f}" for s, v in aucs.items()))

    row = lambda **kw: pd.DataFrame([{**BASE, **kw}])[feats].astype(float)
    p = lambda **kw: float(rf.predict_proba(row(**kw))[0][1])
    p0 = p()
    for desc, kw in EXPECT_UP + EXPECT_DOWN:
        up = (desc, kw) in EXPECT_UP
        d = p(**kw) - p0
        ok = d > 0.02 if up else d < -0.02
        level = "PASS" if ok else ("FAIL" if (d < -0.02 if up else d > 0.02) else "WARN")
        record(level, f"direction: {desc}", f"{p0:.0%} -> {p0 + d:.0%} (expected {'higher' if up else 'lower'})")

    kernel = FidelityQuantumKernel(feature_map=ZZFeatureMap(feature_dimension=q["n_qubits"], reps=2))
    x = q["prep"].transform(row()) * q["angle_scale"] * np.pi
    score = q["svc"].decision_function(kernel.evaluate(x_vec=x, y_vec=q["Xq_train"]))
    pq = float(q["calib"].predict_proba(np.asarray(score).reshape(-1, 1))[0][1])
    record("PASS" if 0.02 < pq < 0.98 else "WARN", "quantum: typical patient",
           f"estimate {pq:.0%} (classical {p0:.0%})")

    if a.baseline:
        b = json.load(open(a.baseline))
        for name in ("Random Forest", "Quantum kernel SVM"):
            new, old = m[name].get("by_source", {}), b[name].get("by_source", {})
            for s in new:
                if s in old:
                    drop = old[s]["roc_auc"] - new[s]["roc_auc"]
                    level = "FAIL" if drop > MAX_DROP else "PASS"
                    record(level, f"{name}: vs baseline on {s}",
                           f"{old[s]['roc_auc']:.3f} -> {new[s]['roc_auc']:.3f}")
            if not old:
                record("WARN", f"{name}: vs baseline", "baseline has no per-source scores; compare overall only: "
                       f"{b[name]['roc_auc']:.3f} -> {m[name]['roc_auc']:.3f} (different test sets)")

    for level, check, message in results:
        print(f"  [{level}] {check}: {message}")
    worst = "FAIL" if any(r[0] == "FAIL" for r in results) else (
        "WARN" if any(r[0] == "WARN" for r in results) else "PASS")
    print(f"\nOVERALL: {worst}" + ("  -> do not deploy these models" if worst == "FAIL" else ""))
    return 1 if worst == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
