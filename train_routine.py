"""Train CardiaQ's second model: heart-disease risk from routine checks only.

Inputs: age, sex, resting systolic BP, total cholesterol, fasting blood sugar > 120.
Data:   data_routine/routine_all.csv from prepare_routine_data.py (5 real datasets,
        13,190 patients). Random Forest; gaps (e.g. Duke never recorded BP) are left
        as missing values, which scikit-learn's Random Forest handles itself.

Why the sample weights: the sources differ hugely in size and disease rate (Soochow is
58% of patients with 20% disease; the angiography sources have 55-72%), and each source
leaves different inputs blank. Unweighted, the model would learn "which hospital is this"
(e.g. "BP missing -> Duke -> high risk") instead of medicine. So every source gets the
same total weight, and within each source disease and no-disease are weighted equally.
Consequence: the output is a balanced score where 50% is the average patient across the
five hospital groups, not the probability of disease in any real population.

Evaluation:
  - held-out test set (20%, stratified by source and outcome), ROC AUC overall and
    per source;
  - leave-one-source-out: train on four sources, test on the fifth, to show how well
    the model transfers to a hospital it has never seen;
  - direction checks: older age, male sex, higher BP, higher cholesterol and high
    fasting sugar should each raise the estimate.

Usage: python train_routine.py  -> routine_model.joblib, routine_metrics.json
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
FEATURES = ["age", "sex", "trestbps", "chol", "fbs"]
DATA = "data_routine/routine_all.csv"


def weights(df):
    """Each source gets equal total weight; classes balanced within each source."""
    n_src = df["source"].nunique()
    counts = df.groupby(["source", "target"])["target"].transform("size")
    return (1.0 / (n_src * 2 * counts)).to_numpy() * len(df)


def make_model():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=10,
                                  random_state=SEED, n_jobs=-1)


def main():
    df = pd.read_csv(DATA)
    train, test = train_test_split(df, test_size=0.2, random_state=SEED,
                                   stratify=df["source"] + "_" + df["target"].astype(str))

    t0 = time.time()
    model = make_model().fit(train[FEATURES], train["target"], sample_weight=weights(train))
    secs = time.time() - t0
    p = model.predict_proba(test[FEATURES])[:, 1]

    by_source = {}
    for s, g in test.groupby("source"):
        m = test["source"].eq(s).to_numpy()
        by_source[s] = {"n_test": int(m.sum()), "disease_rate": round(float(g["target"].mean()), 3),
                        "roc_auc": float(roc_auc_score(g["target"], p[m])),
                        "recall": float(recall_score(g["target"], p[m] >= 0.5))}
        print(f"  test {s:11s} n={m.sum():5d}  AUC {by_source[s]['roc_auc']:.3f}  recall {by_source[s]['recall']:.2f}")
    overall = float(roc_auc_score(test["target"], p))
    # The pooled AUC mixes sources with very different disease rates; the mean of the
    # per-source AUCs is the fairer single number.
    mean_src = float(np.mean([v["roc_auc"] for v in by_source.values()]))
    print(f"  pooled test AUC {overall:.3f}; mean of per-source AUCs {mean_src:.3f}")

    loso = {}
    for s in sorted(df["source"].unique()):
        tr, te = df[df["source"] != s], df[df["source"] == s]
        m = make_model().fit(tr[FEATURES], tr["target"], sample_weight=weights(tr))
        loso[s] = float(roc_auc_score(te["target"], m.predict_proba(te[FEATURES])[:, 1]))
        print(f"  leave-one-source-out: trained without {s:11s} -> AUC on {s}: {loso[s]:.3f}")

    base = dict(age=55, sex=1, trestbps=130, chol=240, fbs=0)
    est = lambda **kw: float(model.predict_proba(pd.DataFrame([{**base, **kw}])[FEATURES])[0][1])
    p0 = est()
    directions = {
        "age 70": est(age=70) - p0, "female": est(sex=0) - p0, "BP 160": est(trestbps=160) - p0,
        "cholesterol 300": est(chol=300) - p0, "fasting sugar > 120": est(fbs=1) - p0,
    }
    print(f"  typical patient {p0:.0%}; changes: " + ", ".join(f"{k} {v:+.0%}" for k, v in directions.items()))

    joblib.dump(model, "routine_model.joblib", compress=3)   # ~19 MB uncompressed
    json.dump({
        "features": FEATURES,
        "n_patients": int(len(df)), "n_train": int(len(train)), "n_test": int(len(test)),
        "sources": {s: {"patients": int(len(g)), "disease_rate": round(float(g["target"].mean()), 3)}
                    for s, g in df.groupby("source")},
        "test_roc_auc_pooled": overall, "test_roc_auc_mean_per_source": mean_src,
        "by_source": by_source, "leave_one_source_out_auc": loso,
        "typical_patient_estimate": p0, "direction_changes": directions,
        "train_seconds": secs,
    }, open("routine_metrics.json", "w"), indent=2)
    print("Saved routine_model.joblib, routine_metrics.json")


if __name__ == "__main__":
    main()
