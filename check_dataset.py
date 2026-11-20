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
  6. A Random Forest within the dataset is not implausibly accurate (files over
     MAX_ROWS rows are subsampled for speed).

Synthetic-data checks (added after testing on realistic-looking fakes: columns
shuffled independently, and rows drawn from a multivariate Gaussian fitted to
real patients). Thresholds were chosen on 5 real files (UCI, Z-Alizadeh Sani,
Duke acath, Soochow, St. Gallen) and 7 known fakes/copies; see the margins below.
  7. BP terminal digits: clinic blood pressures are read by eye and end in 0 or
     5 far more often than the 20% expected by chance (real files 67-99%;
     generated files 18-26%). Below 40% -> WARN (automatic cuffs also lack the
     preference, so alone it is not proof).
  8. Physiological correlations (Spearman, p < 0.01): age vs max heart rate (-),
     age vs systolic BP (+), age vs ST depression (+), max HR vs exercise angina
     (-), max HR vs ST depression (-), exercise angina vs ST depression (+).
     Any pair significantly the wrong way, or none of >= 2 testable pairs
     present -> FAIL (columns generated independently); some missing -> WARN.
  9. Distribution shape: systolic BP and cholesterol are right-skewed in real
     patients (skewness 0.57-1.31 in the real files) and ST depression has a
     large spike at exactly 0 (20-64% of patients); Gaussian generators give
     skewness ~0 and few exact zeros. Skew <= 0.25 or zero-share < 0.15 -> WARN
     per item (needs >= 200 values).
 10. Duplicate rows beyond chance: exact duplicate share minus the share expected
     if columns were independent (permutation baseline). > 5% WARN, > 20% FAIL.
 11. Near-copies of data/: rows matching a patient already in data/ within
     age +-2, BP +-5, chol +-10, max HR +-5, ST depression +-0.3 and identical
     categories, minus the chance rate (permutation baseline). Catches copies
     with small noise added. > 5% WARN, > 20% FAIL.
 12. Signal present: Random Forest AUC (check 6) below 0.58 means the recorded
     risk factors say nothing about disease (real files: 0.65 with only age,
     sex and chest pain, up to 0.88; independently shuffled columns 0.50) -> WARN.
 13. Round numbers: a row count that is a multiple of 100 (>= 500) or exactly
     50.0% disease -> WARN. Weak: real case-control studies are often balanced.
 14. Joint distribution: a classifier tries to tell this file's rows from the
     patients in data/ (shared features). Real hospitals differ too (UCI vs
     Z-Alizadeh Sani AUC 0.94), so AUC >= 0.97 is only a WARN, and only when
     another synthetic-data check also warned; otherwise reported as PASS.
 15. Synthetic evidence: two or more independent signals (WARN or FAIL) from
     checks 7, 8, 9 (each item counts), 10, 11 and 12 -> FAIL. No real file
     tested showed any; the Gaussian fake showed four.
A check that cannot run on a file (missing fields) is shown as N/A and does not
affect the verdict.
"""
import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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

MAX_ROWS = 5000          # larger files are subsampled for the model-based checks
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
NEAR_TOL = {"age": 2, "trestbps": 5, "chol": 10, "thalach": 5, "oldpeak": 0.3}
# (a, b, expected sign) for check 8
PHYSIO_PAIRS = [("age", "thalach", -1), ("age", "trestbps", +1), ("age", "oldpeak", +1),
                ("thalach", "exang", -1), ("thalach", "oldpeak", -1), ("exang", "oldpeak", +1)]
# check names whose WARN or FAIL counts as one signal of generated data (check 15)
SOFT_SIGNALS = ("BP terminal digits", "physiological correlations", "duplicates",
                "near-copies of data/", "signal present")

results = []
state = {}               # values shared between checks (e.g. the Random Forest AUC)


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


def _sample(df, n=MAX_ROWS):
    return df.sample(n, random_state=0) if len(df) > n else df


def check_too_clean(df):
    cols = [c for c in FEATURES if c in df and df[c].notna().mean() > 0.5]
    d = _sample(df)
    X, y = d[cols], d["target"]
    if y.nunique() < 2 or y.value_counts().min() < 15:
        record("WARN", "too clean", "too few cases of one class to test")
        return
    auc = cross_val_score(RandomForestClassifier(300, random_state=0), X, y, cv=5, scoring="roc_auc").mean()
    state["auc"] = auc
    level = "FAIL" if auc > 0.97 else ("WARN" if auc > 0.94 else "PASS")
    record(level, "too clean", f"Random Forest 5-fold AUC {auc:.3f} (routine test results rarely exceed ~0.93)")


def _reference(path):
    """Patients already in data/, excluding the file being checked."""
    existing = [f for f in glob.glob(os.path.join("data", "*.csv"))
                if os.path.abspath(f) != os.path.abspath(path)]
    if not existing:
        return None
    return pd.concat([_normalise(pd.read_csv(f), os.path.basename(f)) for f in existing])


def check_digits(df):
    """Check 7: clinic BP readings end in 0 or 5 far more than 20% of the time."""
    if "trestbps" not in df or df["trestbps"].notna().sum() < 100:
        record("N/A", "BP terminal digits", "fewer than 100 blood pressures")
        return
    x = df["trestbps"].dropna()
    x = x[x == x.round()].astype(int)
    share = (x % 10).isin([0, 5]).mean()
    extra = ""
    if "thalach" in df and df["thalach"].notna().sum() >= 100:
        h = df["thalach"].dropna()
        extra = f"; max heart rate {(h[h == h.round()].astype(int) % 10).isin([0, 5]).mean():.0%} (info)"
    level = "PASS" if share >= 0.40 else "WARN"
    record(level, "BP terminal digits",
           f"{share:.0%} of BPs end in 0 or 5 (chance 20%, real clinics 67-99%){extra}")


def check_correlations(df):
    """Check 8: physiologically expected correlations between measurements."""
    confirmed, absent, wrong = [], [], []
    for a, b, sign in PHYSIO_PAIRS:
        if a not in df or b not in df:
            continue
        p = df[[a, b]].dropna()
        if len(p) < 60 or p[a].nunique() < 2 or p[b].nunique() < 2:
            continue
        r, pv = stats.spearmanr(p[a], p[b])
        tag = f"{a}-{b} {r:+.2f}"
        if pv < 0.01 and np.sign(r) == sign:
            confirmed.append(tag)
        elif pv < 0.01:
            wrong.append(tag)
        else:
            absent.append(tag)
    tested = len(confirmed) + len(absent) + len(wrong)
    if not tested:
        record("N/A", "physiological correlations", "no testable pair of fields")
        return
    if wrong:
        level = "FAIL"
    elif tested >= 2 and not confirmed:
        level = "FAIL"
    else:
        level = "WARN" if absent else "PASS"
    parts = [f"present: {', '.join(confirmed)}" if confirmed else "",
             f"MISSING: {', '.join(absent)}" if absent else "",
             f"WRONG WAY: {', '.join(wrong)}" if wrong else ""]
    record(level, "physiological correlations", "; ".join(p for p in parts if p))


def check_shape(df):
    """Check 9: right-skewed BP and cholesterol; spike of ST depression at 0."""
    items, notes = 0, []
    for c in ("trestbps", "chol"):
        if c in df and df[c].notna().sum() >= 200:
            sk = stats.skew(df[c].dropna())
            ok = sk > 0.25
            items += 1
            notes.append(f"{c} skew {sk:+.2f}" + ("" if ok else " (TOO SYMMETRIC)"))
            if not ok:
                record("WARN", "shape", f"{c} skewness {sk:+.2f}: real values are right-skewed (0.57-1.31)")
    if "oldpeak" in df and df["oldpeak"].notna().sum() >= 200:
        z = (df["oldpeak"].dropna() == 0).mean()
        items += 1
        notes.append(f"ST depression = 0 in {z:.0%}")
        if z < 0.15:
            record("WARN", "shape", f"only {z:.0%} with ST depression exactly 0 (real: 20-64%)")
    if not items:
        record("N/A", "shape", "fewer than 200 values of BP, cholesterol or ST depression")
    elif not any(r[1] == "shape" for r in results):
        record("PASS", "shape", "; ".join(notes))


def check_duplicates(df):
    """Check 10: exact duplicate rows beyond what chance collisions explain."""
    d = _sample(df)
    cols = [c for c in FEATURES if c in d and d[c].notna().mean() > 0.5] + ["target"]
    obs = d[cols].duplicated().mean()
    rng = np.random.default_rng(0)
    base = np.mean([pd.DataFrame({c: rng.permutation(d[c].to_numpy()) for c in cols})
                    .duplicated().mean() for _ in range(5)])
    excess = obs - base
    level = "FAIL" if excess > 0.20 else ("WARN" if excess > 0.05 else "PASS")
    record(level, "duplicates", f"{obs:.1%} duplicate rows, {base:.1%} expected by chance on "
                                f"{len(cols) - 1} fields (excess {excess:+.1%})")


def _near_share(a, b, tol):
    hits = []
    for i in range(0, len(a), 500):
        diff = np.abs(a[i:i + 500, None, :] - b[None, :, :])
        hits.append((diff <= tol).all(axis=2).any(axis=1))
    return float(np.concatenate(hits).mean())


def check_near_copies(df, ref):
    """Check 11: rows that are noisy copies of patients already in data/."""
    if ref is None:
        record("N/A", "near-copies of data/", "no other datasets in data/")
        return
    sh = [c for c in FEATURES if c in df and c in ref
          and df[c].notna().mean() > 0.75 and ref[c].notna().mean() > 0.5]
    if len(sh) < 5 or sum(c in CONTINUOUS for c in sh) < 3:
        record("N/A", "near-copies of data/", f"only {len(sh)} shared fields ({', '.join(sh)})")
        return
    a = _sample(df)[sh].dropna().to_numpy(float)
    b = ref[sh].dropna().to_numpy(float)
    tol = np.array([NEAR_TOL.get(c, 0) + 1e-9 for c in sh])
    obs = _near_share(a, b, tol)
    rng = np.random.default_rng(0)
    base = _near_share(np.column_stack([rng.permutation(a[:, j]) for j in range(a.shape[1])]), b, tol)
    excess = obs - base
    level = "FAIL" if excess > 0.20 else ("WARN" if excess > 0.05 else "PASS")
    record(level, "near-copies of data/", f"{obs:.0%} of rows nearly match a patient in data/, "
                                          f"{base:.0%} expected by chance ({', '.join(sh)})")


def check_signal():
    """Check 12: the risk factors must carry some information about disease."""
    auc = state.get("auc")
    if auc is None:
        record("N/A", "signal present", "Random Forest not run")
        return
    level = "WARN" if auc < 0.58 else "PASS"
    record(level, "signal present", f"Random Forest AUC {auc:.3f} (real files 0.65-0.88; "
                                    "below 0.58 the features say nothing about disease)")


def check_round(df):
    """Check 13: suspiciously round row count or class balance."""
    n, pos = len(df), int(df["target"].sum())
    flags = []
    if n >= 500 and n % 100 == 0:
        flags.append(f"row count {n} is a round number")
    if 2 * pos == n:
        flags.append("exactly 50.0% with disease")
    record("WARN" if flags else "PASS", "round numbers",
           "; ".join(flags) if flags else f"{n} rows, {pos / n:.1%} with disease")


def check_two_sample(df, ref):
    """Check 14: can a classifier tell these rows from the patients in data/?"""
    if ref is None:
        record("N/A", "joint distribution", "no other datasets in data/")
        return
    sh = [c for c in FEATURES if c in df and c in ref
          and df[c].notna().mean() > 0.5 and ref[c].notna().mean() > 0.5]
    if len(sh) < 2:
        record("N/A", "joint distribution", f"only {len(sh)} shared fields")
        return
    a, b = _sample(df)[sh], _sample(ref)[sh]
    X = pd.concat([a, b])
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    auc = cross_val_score(HistGradientBoostingClassifier(random_state=0), X, y, cv=5,
                          scoring="roc_auc").mean()
    other = any(r[0] in ("WARN", "FAIL") and r[1] in SOFT_SIGNALS + ("shape",) for r in results)
    level = "WARN" if auc >= 0.97 and other else "PASS"
    record(level, "joint distribution",
           f"classifier tells these rows from data/ with AUC {auc:.3f} on {len(sh)} fields "
           "(info; real hospitals differ too, e.g. UCI vs Z-Alizadeh Sani 0.94)")


def check_synthetic_evidence():
    """Check 15: several independent soft signals together -> FAIL."""
    signals = [c for lvl, c, m in results
               if lvl in ("WARN", "FAIL") and (c in SOFT_SIGNALS or c == "shape")]
    if len(signals) >= 2:
        record("FAIL", "synthetic evidence",
               f"{len(signals)} independent signs of generated data ({', '.join(signals)})")
    else:
        record("PASS", "synthetic evidence",
               f"{len(signals)} sign(s) of generated data{' (' + signals[0] + ')' if signals else ''}; FAIL at 2")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    df = _normalise(pd.read_csv(a.path), a.name or os.path.basename(a.path))
    print(f"Checking {a.path}: {len(df)} rows, {df['target'].mean():.0%} with disease\n")
    ref = _reference(a.path)
    for check in (check_ranges, lambda d: check_overlap(d, a.path), check_uniform,
                  check_directions, check_physiology, check_too_clean,
                  check_digits, check_correlations, check_shape, check_duplicates,
                  lambda d: check_near_copies(d, ref), lambda d: check_signal(),
                  check_round, lambda d: check_two_sample(d, ref),
                  lambda d: check_synthetic_evidence()):
        check(df)
    for level, check, message in results:
        print(f"  [{level}] {check}: {message}")
    worst = "FAIL" if any(r[0] == "FAIL" for r in results) else (
        "WARN" if any(r[0] == "WARN" for r in results) else "PASS")
    print(f"\nOVERALL: {worst}" + ("  -> do not train on this dataset" if worst == "FAIL" else ""))
    return 1 if worst == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
