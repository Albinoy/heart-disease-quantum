"""Convert the Z-Alizadeh Sani dataset to CardiaQ's columns.

Source: UCI Machine Learning Repository, dataset 412 (CC BY 4.0).
  303 patients, Rajaie Cardiovascular Center, Tehran. CAD = coronary
  angiography showing >= 50% narrowing in at least one artery.
  Cite: Alizadehsani R. et al., "A data mining approach for diagnosis of
  coronary artery disease", Computer Methods and Programs in Biomedicine, 2013.

Mapping (only fields that mean the same thing are used):
  age       <- Age
  sex       <- Sex (Male = 1, Fmale = 0)
  cp        <- Typical Chest Pain -> 0, Atypical -> 1, Nonanginal -> 2, none -> 3
  trestbps  <- BP (mm Hg)
  chol      <- LDL + HDL + TG/5 (Friedewald equation rearranged); left missing when
               TG >= 400 mg/dl, where the equation is not valid
  fbs       <- FBS > 120 mg/dl
  restecg   <- LVH -> 2; ST elevation, ST depression or T inversion -> 1; else 0
  target    <- Cath (Cad = 1, Normal = 0)
  thalach, exang, oldpeak, slope: not recorded (no exercise test) -> missing

Usage: python convert_z_alizadeh_sani.py
  reads  data/raw/z_alizadeh_sani/Z-Alizadeh sani dataset.xlsx
  writes data/z_alizadeh_sani.csv
"""
import numpy as np
import pandas as pd

SRC = "data/raw/z_alizadeh_sani/Z-Alizadeh sani dataset.xlsx"
OUT = "data/z_alizadeh_sani.csv"


def yes(col):
    return col.astype(str).str.strip().str.upper().isin(["Y", "1"])


def convert(z: pd.DataFrame) -> pd.DataFrame:
    typical, atypical, nonanginal = (z["Typical Chest Pain"] == 1), yes(z["Atypical"]), yes(z["Nonanginal"])
    overlap = (typical.astype(int) + atypical.astype(int) + nonanginal.astype(int)) > 1
    if overlap.any():
        raise ValueError(f"{int(overlap.sum())} patients have more than one chest pain type")
    cp = np.select([typical, atypical, nonanginal], [0, 1, 2], default=3)

    st_t = (z["St Elevation"] == 1) | (z["St Depression"] == 1) | (z["Tinversion"] == 1)
    restecg = np.select([yes(z["LVH"]), st_t], [2, 1], default=0)

    chol = (z["LDL"] + z["HDL"] + z["TG"] / 5).where(z["TG"] < 400)

    return pd.DataFrame({
        "age": z["Age"],
        "sex": (z["Sex"].str.strip().str.lower() == "male").astype(int),
        "cp": cp,
        "trestbps": z["BP"],
        "chol": chol.round(0),
        "fbs": (z["FBS"] > 120).astype(int),
        "restecg": restecg,
        "thalach": np.nan,
        "exang": np.nan,
        "oldpeak": np.nan,
        "slope": np.nan,
        "target": (z["Cath"].str.strip().str.lower() == "cad").astype(int),
    })


if __name__ == "__main__":
    out = convert(pd.read_excel(SRC))
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT}: {len(out)} patients, {out['target'].mean():.0%} with CAD")
    print(out.describe().round(1).T[["count", "mean", "min", "max"]])
