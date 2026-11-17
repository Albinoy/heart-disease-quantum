"""Loads heart.csv and converts it to the 13 numeric features the models expect.

Handles both common versions of the UCI Heart Disease data:
- the clean numeric Kaggle file (columns: age, sex, cp, ... target)
- the full multi-hospital file (text values like 'Male', 'asymptomatic',
  extra 'id' and 'dataset' columns, 'thalch' and 'num' column names,
  and missing values)
"""
import numpy as np
import pandas as pd

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

TEXT_MAPS = {
    "sex": {"male": 1, "female": 0},
    "cp": {"typical angina": 0, "atypical angina": 1,
           "non-anginal": 2, "non-anginal pain": 2, "asymptomatic": 3},
    "fbs": {"true": 1, "false": 0},
    "restecg": {"normal": 0, "st-t abnormality": 1, "lv hypertrophy": 2},
    "exang": {"true": 1, "false": 0},
    "slope": {"upsloping": 0, "flat": 1, "downsloping": 2},
    "thal": {"fixed defect": 1, "normal": 2, "reversible defect": 3},
}


def load_data(path="heart.csv"):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"thalch": "thalach", "num": "target"})

    # Text columns -> the same numeric codes the app uses
    for col, mapping in TEXT_MAPS.items():
        if col in df.columns:
            mapped = df[col].astype(str).str.strip().str.lower().map(mapping)
            df[col] = mapped.fillna(pd.to_numeric(df[col], errors="coerce"))

    missing = [c for c in FEATURES + ["target"] if c not in df.columns]
    if missing:
        raise ValueError(f"heart.csv is missing columns: {missing}. Found: {list(df.columns)}")

    df = df[FEATURES + ["target"]].apply(pd.to_numeric, errors="coerce")

    # Severity 0-4 -> disease yes/no
    df["target"] = (df["target"] > 0).astype(int)

    # Zeros in these columns are impossible values (missing data)
    for col in ["chol", "trestbps"]:
        df.loc[df[col] == 0, col] = np.nan

    # Fill gaps with the median (simple; mention this in your write-up)
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
    return df.drop_duplicates()
