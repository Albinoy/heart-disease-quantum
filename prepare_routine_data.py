"""Build the training table for CardiaQ's second model: routine checks only.

The main model needs exercise-test results, which only the UCI and Z-Alizadeh Sani
patients have. This second model uses the five inputs that most real datasets share,
so it can learn from all five real-patient datasets CardiaQ holds (13,190 patients):

  inputs: age, sex, resting systolic BP (trestbps), total cholesterol (chol),
          fasting blood sugar > 120 mg/dl (fbs)
  target: 1 = coronary disease (definition varies by source, see below)

Chest pain type is NOT used: Duke and Soochow never recorded it (85% missing overall).

Sources (fields a source did not record are left blank, never invented):
  uci        data/uci_heart_disease.csv (via data_prep._normalise). 4 hospitals, 1980s;
             CAD = >50% narrowing on angiography. CC BY 4.0.
  z_alizadeh data/z_alizadeh_sani.csv. Tehran; CAD = >=50% narrowing on angiography.
             CC BY 4.0.
  duke       data/raw/duke_acath/acath.xls. Duke Cardiovascular Disease Databank via the
             Vanderbilt Biostatistics datasets (hbiostat.org/data); 3,504 chest-pain
             patients referred for catheterisation; CAD = >=75% narrowing (sigdz).
             age <- age; sex <- 1 - sex (source 0 = male); chol <- choleste (mg/dl).
             No BP or fasting sugar. No licence stated by the host.
  soochow    data/raw/soochow_ccssscc_fibrinogen/soochow_fibrinogen_raw.xls. First
             Affiliated Hospital of Soochow University, 2010-2013 (Figshare 23898021,
             CC BY 4.0). Acute myocardial infarction (target 1) vs cardiology inpatients
             without CAD (0) - a different label from the angiography sources.
             age; sex (1 male, 2 female); trestbps <- sbp; chol <- tc mmol/L x 38.67;
             fbs <- glucose mmol/L > 6.67. First two data rows are labels/units.
  stgallen   data/raw/stgallen_ica_2015/pone.0222137.s001.xlsx. Kantonsspital St. Gallen,
             elective angiography 2015 (PLoS One 2019, CC BY 4.0); CAD = stenotic CAD on
             angiography (Coro_CAD 1; 0 and 3 = no). Only age and sex are usable here
             (its cholesterol column is a yes/no flag, not mg/dl).

Cleaning (same rules as the main data): BP outside 70-250 mmHg, cholesterol outside
80-700 mg/dl and age outside 18-100 are set to blank (data-entry errors, < 2% per source).

Usage: python prepare_routine_data.py   ->  writes data_routine/routine_all.csv
(data_routine/ is separate from data/, so the main model's training is unaffected.)
"""
import os

import numpy as np
import pandas as pd

from data_prep import _normalise

RAW = "data/raw"
OUT_DIR = "data_routine"
OUT = os.path.join(OUT_DIR, "routine_all.csv")
FEATURES = ["age", "sex", "trestbps", "chol", "fbs"]
RANGES = {"age": (18, 100), "trestbps": (70, 250), "chol": (80, 700)}


def strict_map(s, mapping, name):
    bad = s.notna() & ~s.isin(list(mapping))
    if bad.any():
        raise ValueError(f"{name}: unexpected codes {sorted(s[bad].unique())}")
    return s.map(mapping)


def uci():
    return _normalise(pd.read_csv("data/uci_heart_disease.csv"), "uci")


def z_alizadeh():
    return _normalise(pd.read_csv("data/z_alizadeh_sani.csv"), "z_alizadeh")


def duke():
    d = pd.read_excel(f"{RAW}/duke_acath/acath.xls", engine="xlrd")
    return pd.DataFrame({
        "age": d["age"],
        "sex": strict_map(d["sex"], {0: 1, 1: 0}, "duke sex"),
        "chol": d["choleste"],
        "target": strict_map(d["sigdz"], {0: 0, 1: 1}, "duke sigdz"),
    })


def soochow():
    d = pd.read_excel(f"{RAW}/soochow_ccssscc_fibrinogen/soochow_fibrinogen_raw.xls",
                      engine="xlrd").iloc[2:].dropna(how="all")
    for c in ["age", "sex", "sbp", "tc", "glucose", "Group"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Group"])
    return pd.DataFrame({
        "age": d["age"],
        "sex": strict_map(d["sex"], {1: 1, 2: 0}, "soochow sex"),
        "trestbps": d["sbp"],
        "chol": (d["tc"] * 38.67).round(0),
        "fbs": (d["glucose"] > 6.67).astype(float).where(d["glucose"].notna()),
        "target": strict_map(d["Group"], {0: 0, 1: 1}, "soochow Group").astype(int),
    })


def stgallen():
    d = pd.read_excel(f"{RAW}/stgallen_ica_2015/pone.0222137.s001.xlsx", engine="openpyxl")
    cad = [c for c in d.columns if c.startswith("Coro_CAD")][0]
    return pd.DataFrame({
        "age": d["Age"],
        "sex": strict_map(d["Sex"].str.strip().str.lower(), {"m": 1, "w": 0}, "stgallen sex"),
        "target": strict_map(d[cad], {0: 0, 1: 1, 3: 0}, "stgallen Coro_CAD"),
    })


def build():
    parts = []
    for name, fn in [("uci", uci), ("z_alizadeh", z_alizadeh), ("duke", duke),
                     ("soochow", soochow), ("stgallen", stgallen)]:
        d = fn().reindex(columns=FEATURES + ["target"]).astype(float)
        for c, (lo, hi) in RANGES.items():
            d.loc[(d[c] < lo) | (d[c] > hi), c] = np.nan
        d["target"] = d["target"].astype(int)
        d["source"] = name
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    df = build()
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT}: {len(df)} patients")
    summary = df.groupby("source").agg(patients=("target", "size"), disease=("target", "mean"))
    summary = summary.join(df.groupby("source")[FEATURES].apply(lambda g: g.isna().mean()).add_suffix("_missing"))
    print(summary.round(2).to_string())
