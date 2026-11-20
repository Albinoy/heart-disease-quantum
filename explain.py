"""
CardiaQ - Phase 1: SHAP explainability for the Random Forest model.

What this module does
---------------------
1. Explains WHY the Random Forest predicted heart disease for one patient
   (per-feature SHAP contributions, in plain English).
2. Gives global feature importance across many patients.
3. Maps each feature to a heart region, so Phase 3 can make the
   matching part of the 3D heart blink.

Works with the standard UCI / Kaggle heart columns:
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
One-hot encoded columns such as "cp_2" or "thal_3" are handled too: they are
grouped back under their original feature name.

Install:  pip install shap scikit-learn pandas matplotlib
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Feature dictionary: plain-English name, heart region, explanation
#    Region ids are what the 3D viewer will use in Phase 3.
# ---------------------------------------------------------------------------
FEATURE_INFO: dict[str, dict[str, str]] = {
    "age": {
        "label": "Age",
        "region": "whole_heart",
        "why": "Risk of artery narrowing rises with age.",
    },
    "sex": {
        "label": "Sex",
        "region": "whole_heart",
        "why": "Heart disease rates differ between men and women.",
    },
    "cp": {
        "label": "Chest pain type",
        "region": "coronary_arteries",
        "why": "Angina-type chest pain usually comes from reduced blood flow in the coronary arteries.",
    },
    "trestbps": {
        "label": "Resting blood pressure",
        "region": "systemic_arteries",
        "why": "High blood pressure strains artery walls and the heart muscle.",
    },
    "chol": {
        "label": "Cholesterol",
        "region": "coronary_arteries",
        "why": "Cholesterol builds plaque that narrows the coronary arteries.",
    },
    "fbs": {
        "label": "Fasting blood sugar > 120 mg/dl",
        "region": "systemic_arteries",
        "why": "High blood sugar damages blood vessels over time.",
    },
    "restecg": {
        "label": "Resting ECG result",
        "region": "left_ventricle",
        "why": "ECG abnormalities can show a thickened left ventricle or strain.",
    },
    "thalach": {
        "label": "Maximum heart rate reached",
        "region": "myocardium",
        "why": "A low peak heart rate on exercise can mean the heart muscle is struggling.",
    },
    "exang": {
        "label": "Chest pain during exercise",
        "region": "myocardium",
        "why": "Pain on exertion suggests the heart muscle is not getting enough blood.",
    },
    "oldpeak": {
        "label": "ST depression (exercise ECG)",
        "region": "myocardium",
        "why": "ST depression is a classic sign of ischaemia (low blood flow to heart muscle).",
    },
    "slope": {
        "label": "ST segment slope",
        "region": "myocardium",
        "why": "A flat or downward slope on exercise ECG is linked to ischaemia.",
    },
    "ca": {
        "label": "Narrowed major vessels (0-3)",
        "region": "coronary_arteries",
        "why": "Counts major coronary vessels showing narrowing on fluoroscopy.",
    },
    "thal": {
        "label": "Thallium stress test",
        "region": "myocardium",
        "why": "Shows areas of heart muscle with poor blood supply (perfusion defects).",
    },
}

# Category codes -> names (same encoding as data_prep.py and the app's form).
VALUE_NAMES: dict[str, list[str]] = {
    "sex": ["Female", "Male"],
    "cp": ["Typical angina", "Atypical angina", "Non-anginal pain", "No chest pain (asymptomatic)"],
    "fbs": ["No", "Yes"],
    "restecg": ["Normal", "ST-T wave abnormality", "Left ventricle enlarged"],
    "exang": ["No", "Yes"],
    "slope": ["Upsloping", "Flat", "Downsloping"],
    "thal": ["Not recorded", "Fixed defect", "Normal", "Reversible defect"],
}

# Per-value explanations, used instead of FEATURE_INFO["why"] when present.
VALUE_WHY: dict[str, list[str]] = {
    "cp": [
        "Typical angina usually comes from reduced blood flow in the coronary arteries.",
        "Atypical angina can still point to reduced blood flow in the coronary arteries.",
        "Non-anginal chest pain is usually not caused by the heart.",
        "In this data, many patients with heart disease reported no chest pain, "
        "because coronary disease can be 'silent' and cause no symptoms.",
    ],
}


def _code(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return int(f) if f.is_integer() else None


def display_value(feature: str, value) -> str:
    """Readable value: category name for coded features, tidy number otherwise."""
    names, code = VALUE_NAMES.get(feature), _code(value)
    if names and code is not None and 0 <= code < len(names):
        return names[code]
    try:
        f = float(value)
        return f"{f:g}"
    except (TypeError, ValueError):
        return str(value)


# When a value counts as abnormal, so the medical "why" sentence (which describes an
# abnormal finding) applies. Standard clinical cut-offs; `p` is the whole patient row.
ABNORMAL = {
    "trestbps": lambda v, p: v >= 140,                      # hypertension threshold
    "chol": lambda v, p: v >= 240,                          # high total cholesterol
    "fbs": lambda v, p: v == 1,
    "restecg": lambda v, p: v > 0,
    "thalach": lambda v, p: v < 0.85 * (220 - p.get("age", 55)),   # below 85% of predicted max
    "exang": lambda v, p: v == 1,
    "oldpeak": lambda v, p: v >= 1.0,                       # >= 1 mm ST depression
    "slope": lambda v, p: v >= 1,                           # flat or downsloping
    "cp": lambda v, p: v != 2,                              # non-anginal pain is not a cardiac sign
}

NEUTRAL_WHY = ("This value is not abnormal on its own; the model links it to slightly higher risk "
               "together with this patient's other answers, based on patterns in its training data.")


def value_why(feature: str, value, patient: dict | None = None) -> str:
    """Medical reason for a risk-raising answer. If the patient's value is normal
    (e.g. blood pressure 130), the reason for an abnormal value would mislead, so a
    neutral sentence is returned instead."""
    rule = ABNORMAL.get(feature)
    if rule is not None:
        try:
            if not rule(float(value), patient or {}):
                return NEUTRAL_WHY
        except (TypeError, ValueError):
            pass
    whys, code = VALUE_WHY.get(feature), _code(value)
    if whys and code is not None and 0 <= code < len(whys):
        return whys[code]
    return FEATURE_INFO.get(feature, {}).get("why", "")


REGION_NAMES: dict[str, str] = {
    "coronary_arteries": "Coronary arteries (LAD, RCA, circumflex)",
    "left_ventricle": "Left ventricle",
    "myocardium": "Heart muscle (myocardium)",
    "systemic_arteries": "Arteries / blood pressure system",
    "whole_heart": "General risk (whole heart)",
}


def base_feature(column: str) -> str:
    """Map a (possibly one-hot) column name back to its original feature.

    "cp_2" -> "cp", "thal_3.0" -> "thal", "age" -> "age".
    """
    if column in FEATURE_INFO:
        return column
    prefix = column.split("_")[0]
    return prefix if prefix in FEATURE_INFO else column


# ---------------------------------------------------------------------------
# 2. SHAP explainer
# ---------------------------------------------------------------------------
def _split_model(model, X: pd.DataFrame):
    """If model is a sklearn Pipeline, return (final estimator, X after the earlier steps).

    SHAP's TreeExplainer needs the tree model itself, not a Pipeline.
    For a plain model this returns (model, X) unchanged.
    """
    steps = getattr(model, "steps", None)
    if not steps:
        return model, X
    est = steps[-1][1]
    Xt = model[:-1].transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    if isinstance(Xt, pd.DataFrame):
        return est, Xt
    try:
        cols = list(model[:-1].get_feature_names_out())
    except Exception:
        cols = list(X.columns) if Xt.shape[1] == X.shape[1] else [f"f{i}" for i in range(Xt.shape[1])]
    cols = [c.split("__", 1)[-1] for c in cols]  # "num__age" -> "age"
    return est, pd.DataFrame(Xt, columns=cols, index=X.index)


def build_explainer(model):
    """Create a fast, exact TreeExplainer for a fitted RandomForestClassifier
    (or a Pipeline ending in one)."""
    import shap  # imported here so the rest of the module works without shap

    steps = getattr(model, "steps", None)
    return shap.TreeExplainer(steps[-1][1] if steps else model)


def _positive_class_values(raw, n_features: int, class_index: int) -> np.ndarray:
    """Return SHAP values for the 'disease' class as shape (n_samples, n_features).

    Different SHAP versions return different shapes:
      - old: list of arrays, one per class           -> raw[class_index]
      - new: array (n_samples, n_features, n_classes) -> raw[:, :, class_index]
      - Explanation object                            -> raw.values
    """
    if hasattr(raw, "values") and not isinstance(raw, np.ndarray):
        raw = raw.values
    if isinstance(raw, list):
        return np.asarray(raw[class_index])
    raw = np.asarray(raw)
    if raw.ndim == 3:
        return raw[:, :, class_index]
    if raw.ndim == 2 and raw.shape[1] == n_features:
        return raw  # already single-output
    raise ValueError(f"Unexpected SHAP output shape: {raw.shape}")


def _base_value(explainer, class_index: int) -> float:
    ev = np.atleast_1d(np.asarray(explainer.expected_value, dtype=float))
    return float(ev[class_index] if ev.size > 1 else ev[0])


def _disease_class_index(model, positive_label=1) -> int:
    classes = list(getattr(model, "classes_", [0, 1]))
    if positive_label not in classes:
        raise ValueError(f"positive_label={positive_label!r} not in model.classes_={classes}")
    return classes.index(positive_label)


# ---------------------------------------------------------------------------
# 3. Explain one patient
# ---------------------------------------------------------------------------
def explain_patient(model, explainer, patient: pd.DataFrame, positive_label=1) -> dict:
    """Explain a single patient's prediction.

    patient: a one-row DataFrame with the same columns the model was trained on.

    Returns a dict with:
      probability   - predicted probability of heart disease
      base_value    - average model output (the starting point before features)
      contributions - DataFrame: feature, label, value, shap, region, why
                      sorted by absolute impact (one-hot columns grouped)
    """
    if len(patient) != 1:
        raise ValueError("explain_patient expects exactly one row")

    cls = _disease_class_index(model, positive_label)
    est, Xt = _split_model(model, patient)
    raw = explainer.shap_values(Xt)
    values = _positive_class_values(raw, Xt.shape[1], cls)[0]

    # Show the patient's original values where columns line up, else transformed ones.
    shown = patient.iloc[0].reindex(Xt.columns)
    shown = shown.where(shown.notna(), Xt.iloc[0])
    df = pd.DataFrame(
        {"column": Xt.columns, "value": shown.values, "shap": values}
    )
    df["feature"] = df["column"].map(base_feature)

    # Group one-hot columns back into their original feature.
    grouped = (
        df.groupby("feature", sort=False)
        .agg(
            shap=("shap", "sum"),
            value=("value", lambda v: v.iloc[0] if len(v) == 1 else _onehot_value(df, v)),
        )
        .reset_index()
    )
    grouped["label"] = grouped["feature"].map(lambda f: FEATURE_INFO.get(f, {}).get("label", f))
    grouped["region"] = grouped["feature"].map(lambda f: FEATURE_INFO.get(f, {}).get("region", "whole_heart"))
    row = {k: float(v) for k, v in patient.iloc[0].items() if pd.notna(v)}
    grouped["why"] = [value_why(f, v, row) for f, v in zip(grouped["feature"], grouped["value"])]
    grouped["value"] = [display_value(f, v) for f, v in zip(grouped["feature"], grouped["value"])]
    grouped["direction"] = np.where(grouped["shap"] > 0, "raises risk", "lowers risk")
    grouped = grouped.reindex(grouped["shap"].abs().sort_values(ascending=False).index).reset_index(drop=True)

    probability = float(model.predict_proba(patient)[0][cls])
    return {
        "probability": probability,
        "base_value": _base_value(explainer, cls),
        "contributions": grouped[["feature", "label", "value", "shap", "direction", "region", "why"]],
    }


def _onehot_value(df: pd.DataFrame, values: pd.Series) -> str:
    """For a one-hot group, show which category is active (e.g. 'cp_2')."""
    cols = df.loc[values.index, "column"]
    active = cols[values.astype(float) > 0]
    return ", ".join(active) if len(active) else "none"


def region_scores(contributions: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    """Sum the risk-RAISING SHAP values per heart region.

    This is the bridge to Phase 3: regions with the highest score are the
    ones the 3D heart should blink. Only positive contributions count,
    because a region should light up when it pushes risk UP.
    """
    pos = contributions[contributions["shap"] > 0]
    scores = (
        pos.groupby("region")
        .agg(score=("shap", "sum"), features=("label", lambda s: ", ".join(s)))
        .reset_index()
        .sort_values("score", ascending=False)
    )
    scores["region_name"] = scores["region"].map(REGION_NAMES).fillna(scores["region"])
    scores = scores.reset_index(drop=True)
    return scores.head(top_n) if top_n else scores


def explain_in_words(result: dict, top_k: int = 3) -> list[str]:
    """Plain-English sentences for the top risk-raising factors."""
    c = result["contributions"]
    risers = c[c["shap"] > 0].head(top_k)
    lines = []
    for _, row in risers.iterrows():
        pts = row["shap"] * 100
        lines.append(f"{row['label']} ({row['value']}) raised predicted risk by about {pts:.1f} points. {row['why']}")
    if not lines:
        lines.append("None of this patient's values pushed the risk above average.")
    return lines


def caveats(result: dict) -> list[str]:
    """Warnings for known quirks of the training data that affect this patient's estimate."""
    c = result["contributions"]
    out = []
    cp = c[c["feature"] == "cp"]
    if len(cp) and cp.iloc[0]["value"] in ("Typical angina", "Atypical angina") and cp.iloc[0]["shap"] < -0.01:
        out.append(
            f"{cp.iloc[0]['value']} lowered this estimate. That is a quirk of the UCI training data: most "
            "patients there were sent for angiography, and those without chest pain usually had other strong "
            "signs of disease, so 'no chest pain' carries more risk in the model than angina does. Clinically, "
            "angina is a warning sign, and anyone with chest pain should see a doctor."
        )
    return out


# ---------------------------------------------------------------------------
# 4. Global importance across many patients
# ---------------------------------------------------------------------------
def global_importance(model, explainer, X: pd.DataFrame, positive_label=1, max_rows: int = 300) -> pd.DataFrame:
    """Mean |SHAP| per feature over a sample of patients (one-hot grouped)."""
    cls = _disease_class_index(model, positive_label)
    sample = X.sample(min(max_rows, len(X)), random_state=42) if len(X) > max_rows else X
    _, Xt = _split_model(model, sample)
    values = _positive_class_values(explainer.shap_values(Xt), Xt.shape[1], cls)

    df = pd.DataFrame(np.abs(values), columns=Xt.columns)
    df.columns = [base_feature(c) for c in df.columns]
    imp = df.T.groupby(level=0).sum().T.mean().sort_values(ascending=False)
    out = imp.rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"})
    out["label"] = out["feature"].map(lambda f: FEATURE_INFO.get(f, {}).get("label", f))
    return out


# ---------------------------------------------------------------------------
# 5. Charts (plain matplotlib, so they render cleanly in Streamlit)
# ---------------------------------------------------------------------------
RISK_UP = "#d62839"
RISK_DOWN = "#2b7bba"


def plot_patient(result: dict, top_k: int = 8):
    """Horizontal bar chart: red bars raise risk, blue bars lower it."""
    c = result["contributions"].head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(c) + 1.2))
    colors = [RISK_UP if v > 0 else RISK_DOWN for v in c["shap"]]
    ax.barh(c["label"], c["shap"] * 100, color=colors)
    ax.axvline(0, color="#555", linewidth=0.8)
    ax.set_xlabel("Effect on predicted risk (percentage points)")
    ax.set_title(f"Why the model predicted {result['probability']:.0%} risk")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def plot_global(importance: pd.DataFrame, top_k: int = 13):
    imp = importance.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(imp) + 1.2))
    ax.barh(imp["label"], imp["mean_abs_shap"] * 100, color="#6c757d")
    ax.set_xlabel("Average impact on predicted risk (percentage points)")
    ax.set_title("Which features matter most overall")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig
