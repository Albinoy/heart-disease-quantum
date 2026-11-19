"""Load one or more heart disease CSV files into one clean numeric table.

Where the data comes from
- If a folder called data/ exists and contains .csv files, every file in it
  is loaded and combined. Put each dataset you download in there.
- Otherwise heart.csv in this folder is used.

Supported formats (column names are matched case-insensitively)
- UCI Cleveland numeric (Kaggle "heart.csv": age, sex, cp, ... target)
- UCI full multi-hospital (text values, 'thalch', 'num', 'dataset' columns)
- Kaggle "Heart Failure Prediction" by fedesoriano (Age, ChestPainType,
  RestingBP, ... HeartDisease)
- Kaggle "Heart Disease Dataset (Comprehensive)" / statlog-cleveland-hungary
  (chest pain type, resting bp s, ... target)

Everything is converted to the same codes the app uses:
  sex 1 = male, 0 = female
  cp 0 typical angina, 1 atypical angina, 2 non-anginal pain, 3 asymptomatic
  restecg 0 normal, 1 ST-T abnormality, 2 LV hypertrophy
  slope 0 upsloping, 1 flat, 2 downsloping
  thal 1 fixed defect, 2 normal, 3 reversible defect
  target 1 = heart disease, 0 = none

Combining datasets
- Exact duplicate patients are removed (several public files contain the same
  Cleveland patients, which inflates accuracy if they land in both the
  training and test sets).
- A feature is dropped if more than MAX_MISSING of rows are missing it after
  combining (e.g. ca and thal, which most datasets do not record). The
  remaining gaps are filled with the median.
"""
import glob
import os

import numpy as np
import pandas as pd

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
MAX_MISSING = 0.5

ALIASES = {
    "chestpaintype": "cp", "chest pain type": "cp",
    "restingbp": "trestbps", "resting bp s": "trestbps",
    "cholesterol": "chol",
    "fastingbs": "fbs", "fasting blood sugar": "fbs",
    "restingecg": "restecg", "resting ecg": "restecg",
    "maxhr": "thalach", "max heart rate": "thalach", "thalch": "thalach",
    "exerciseangina": "exang", "exercise angina": "exang",
    "st_slope": "slope", "st slope": "slope",
    "heartdisease": "target", "num": "target", "condition": "target",
}

TEXT_MAPS = {
    "sex": {"male": 1, "m": 1, "female": 0, "f": 0},
    "cp": {"typical angina": 0, "ta": 0, "atypical angina": 1, "ata": 1,
           "non-anginal": 2, "non-anginal pain": 2, "nap": 2,
           "asymptomatic": 3, "asy": 3},
    "fbs": {"true": 1, "false": 0},
    "restecg": {"normal": 0, "st-t abnormality": 1, "st": 1,
                "lv hypertrophy": 2, "lvh": 2},
    "exang": {"true": 1, "false": 0, "y": 1, "n": 0, "yes": 1, "no": 0},
    "slope": {"upsloping": 0, "up": 0, "flat": 1, "downsloping": 2, "down": 2},
    "thal": {"fixed defect": 1, "normal": 2, "reversable defect": 3,
             "reversible defect": 3},
}


def _normalise(raw, name):
    df = raw.copy()
    df.columns = [ALIASES.get(c.strip().lower(), c.strip().lower()) for c in df.columns]
    if "target" not in df.columns:
        raise ValueError(f"{name}: no target column found. Columns: {list(df.columns)}")

    for col, mapping in TEXT_MAPS.items():
        if col in df.columns:
            text = df[col].astype(str).str.strip().str.lower()
            df[col] = text.map(mapping).fillna(pd.to_numeric(df[col], errors="coerce"))

    keep = [c for c in FEATURES if c in df.columns]
    df = df[keep + ["target"]].apply(pd.to_numeric, errors="coerce")

    # Some files number categories from 1 instead of 0.
    if "cp" in df and df["cp"].max() == 4:
        df["cp"] = df["cp"] - 1
    if "slope" in df and df["slope"].max() == 3:
        df["slope"] = (df["slope"] - 1).clip(lower=0)
    if "thal" in df and df["thal"].max() in (6, 7):     # original UCI codes 3/6/7
        df["thal"] = df["thal"].map({3: 2, 6: 1, 7: 3})

    df["target"] = (df["target"] > 0).astype(int)       # severity 0-4 -> yes/no

    # Zeros here are impossible values that stand for missing data.
    for col in ("chol", "trestbps"):
        if col in df:
            df.loc[df[col] == 0, col] = np.nan

    df["source"] = name
    return df


def find_sources(path=None):
    if path:
        return [path]
    files = sorted(glob.glob(os.path.join("data", "*.csv")))
    return files or ["heart.csv"]


def load_data(path=None, with_source=False, verbose=False):
    """Return one numeric table: the usable features plus 'target'."""
    files = find_sources(path)
    parts = [_normalise(pd.read_csv(f), os.path.basename(f)) for f in files]
    df = pd.concat(parts, ignore_index=True, sort=False)

    missing = df.reindex(columns=FEATURES).isna().mean()
    features = [c for c in FEATURES if missing.get(c, 1.0) <= MAX_MISSING]
    dropped = [c for c in FEATURES if c not in features]

    n_before = len(df)
    df = df.drop_duplicates(subset=features + ["target"]).reset_index(drop=True)
    df[features] = df[features].fillna(df[features].median())

    info = {
        "files": {p["source"].iloc[0]: int(len(p)) for p in parts},
        "rows_before_dedup": int(n_before),
        "rows": int(len(df)),
        "duplicates_removed": int(n_before - len(df)),
        "features": features,
        "dropped_features": {c: round(float(missing.get(c, 1.0)), 2) for c in dropped},
        "positive_rate": round(float(df["target"].mean()), 3),
    }
    if verbose:
        print(f"Loaded {len(files)} file(s): {info['files']}")
        print(f"Rows: {n_before} -> {len(df)} after removing {info['duplicates_removed']} duplicates")
        if dropped:
            print(f"Dropped (more than {MAX_MISSING:.0%} missing): {info['dropped_features']}")
        print(f"Using features: {features}")

    cols = features + ["target"] + (["source"] if with_source else [])
    out = df[cols].copy()
    out.attrs["info"] = info
    return out
