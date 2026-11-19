"""Check that a heart-disease dataset looks like real patients before training on it.

Encodes the checks that caught a non-real dataset (Mendeley "Cardiovascular
Disease Dataset", India): measurements spread flat instead of clustering,
known risk factors unrelated to disease, and a model that scores near-perfectly
on routine test results.

Usage:
    python check_dataset.py path/to/file.csv [--name NAME]
The file can be in any format data_prep.py understands (e.g. CardiaQ columns).
Exit code: 0 = PASS or WARN, 1 = FAIL.

Checks
  1. Value ranges are physiologically possible.
  2. Overlap with datasets already in data/ (exact duplicate patients).
  3. Numeric measurements are not flat/uniform (real data clusters).
  4. Known risk factors point the right way (older age, male sex, exercise
     angina, ST depression, low peak heart rate -> more disease).
  5. Age vs maximum heart rate is negative (physiology), when both exist.
  6. A Random Forest within the dataset is not implausibly accurate.
"""
import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from data_prep import FEATURES, _normalise

warnings.filterwarnings("ignore")

RANGES = {"age": (18, 100), "trestbps": (70, 250), "chol": (80, 700),
          "thalach": (50, 230), "oldpeak": (-3, 10)}
CODES = {"sex": {0, 1}, "cp": {0, 1, 2, 3}, "fbs": {0, 1}, "restecg": {0, 1, 2},
         "exang": {0, 1}, "slope": {0, 1, 2}}
# (feature, description, rule) where rule(df) -> boolean mask of the "higher risk" group
RISK_RULES = [
    ("age", "age >= 60", lambda d: d["age"] >= 60),
    ("sex", "male", lambda d: d["sex"] == 1),
    ("exang", "exercise angina", lambda d: d["exang"] == 1),
    ("oldpeak", "ST depression >= 2", lambda d: d["oldpeak"] >= 2),
    ("thalach", "max heart rate < 130", lambda d: d["thalach"] < 130),
]

results = []


def record(level, check, message):
    results.append((level, check, message))


def check_ranges(df):
    """A few impossible values (<= 2% of rows) is a data-entry quirk -> WARN;
    more, or undefined category codes, means the file is not what it claims -> FAIL."""
    bad, worst_share = [], 0.0
    for c, (lo, hi) in RANGES.items():
        if c in df:
            x = df[c].dropna()
            n = int(((x < lo) | (x > hi)).sum())
            if n:
                bad.append(f"{c}: {n} outside {lo}-{hi}")
                worst_share = max(worst_share, n / len(df))
    codes_bad = False
    for c, ok in CODES.items():
        if c in df:
            extra = set(df[c].dropna().astype(int).unique()) - ok
            if extra:
                bad.append(f"{c}: undefined codes {sorted(extra)}")
                codes_bad = True
    level = "FAIL" if codes_bad or worst_share > 0.02 else ("WARN" if bad else "PASS")
    record(level, "value ranges", "; ".join(bad) or "all values possible")


def check_overlap(df, path):
    existing = [f for f in glob.glob(os.path.join("data", "*.csv"))
                if os.path.abspath(f) != os.path.abspath(path)]
    if not existing:
        record("PASS", "overlap", "no other datasets in data/")
        return
    other = pd.concat([_normalise(pd.read_csv(f), os.path.basename(f)) for f in existing])
    cols = [c for c in FEATURES if c in df and c in other and df[c].notna().mean() > 0.9
            and other[c].notna().mean() > 0.5]
    if len(cols) < 5:
        record("WARN", "overlap", f"only {len(cols)} shared complete features; cannot check duplicates well")
        return
    key = lambda d: [tuple(r) for r in d[cols + ["target"]].round(2).fillna(-999).to_numpy()]
    seen = set(key(other))
    share = np.mean([k in seen for k in key(df)])
    level = "FAIL" if share > 0.2 else ("WARN" if share > 0.02 else "PASS")
    record(level, "overlap", f"{share:.0%} of rows exactly match patients already in data/ ({', '.join(cols)})")


def check_uniform(df):
    flat = []
    for c in ["age", "trestbps", "chol", "thalach"]:
        if c in df:
            x = df[c].dropna()
            x = x[x > 0]
            if len(x) >= 50 and x.nunique() > 10:
                p = stats.kstest(x, "uniform", args=(x.min(), x.max() - x.min())).pvalue
                if p > 0.05:
                    flat.append(f"{c} (p={p:.2f})")
    level = "FAIL" if len(flat) >= 2 else ("WARN" if flat else "PASS")
    record(level, "not flat", ("looks uniformly spread: " + ", ".join(flat)) if flat
           else "measurements cluster like real patients")


def check_directions(df):
    for feat, desc, rule in RISK_RULES:
        if feat not in df or df[feat].notna().sum() < 60:
            continue
        d = df.dropna(subset=[feat])
        m = rule(d)
        if m.sum() < 20 or (~m).sum() < 20:
            continue
        hi, lo = d.loc[m, "target"].mean(), d.loc[~m, "target"].mean()
        diff = hi - lo
        level = "FAIL" if diff < -0.05 else ("WARN" if diff < 0.03 else "PASS")
        record(level, f"risk: {desc}", f"disease rate {hi:.0%} vs {lo:.0%} (expected higher)")


def check_physiology(df):
    if "age" in df and "thalach" in df and df[["age", "thalach"]].dropna().shape[0] >= 60:
        r = df[["age", "thalach"]].dropna().corr().iloc[0, 1]
        record("PASS" if r < -0.1 else "WARN", "age vs max heart rate",
               f"correlation {r:.2f} (real patients: clearly negative, about -0.4)")


def check_too_clean(df):
    cols = [c for c in FEATURES if c in df and df[c].notna().mean() > 0.5]
    X, y = df[cols], df["target"]
    if y.nunique() < 2 or y.value_counts().min() < 15:
        record("WARN", "too clean", "too few cases of one class to test")
        return
    auc = cross_val_score(RandomForestClassifier(300, random_state=0), X, y, cv=5, scoring="roc_auc").mean()
    level = "FAIL" if auc > 0.97 else ("WARN" if auc > 0.94 else "PASS")
    record(level, "too clean", f"Random Forest 5-fold AUC {auc:.3f} (routine test results rarely exceed ~0.93)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    df = _normalise(pd.read_csv(a.path), a.name or os.path.basename(a.path))
    print(f"Checking {a.path}: {len(df)} rows, {df['target'].mean():.0%} with disease\n")
    for check in (check_ranges, lambda d: check_overlap(d, a.path), check_uniform,
                  check_directions, check_physiology, check_too_clean):
        check(df)
    for level, check, message in results:
        print(f"  [{level}] {check}: {message}")
    worst = "FAIL" if any(r[0] == "FAIL" for r in results) else (
        "WARN" if any(r[0] == "WARN" for r in results) else "PASS")
    print(f"\nOVERALL: {worst}" + ("  -> do not train on this dataset" if worst == "FAIL" else ""))
    return 1 if worst == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
