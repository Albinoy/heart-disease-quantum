"""Tests for explain.py that run WITHOUT shap installed (uses a stand-in explainer).

Run:  python test_explain.py
With shap installed, it also runs a real TreeExplainer check.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import explain as ex

COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"]


def make_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "age": rng.integers(29, 78, n), "sex": rng.integers(0, 2, n),
        "cp": rng.integers(0, 4, n), "trestbps": rng.integers(94, 200, n),
        "chol": rng.integers(126, 400, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "thalach": rng.integers(71, 202, n),
        "exang": rng.integers(0, 2, n), "oldpeak": rng.uniform(0, 6, n).round(1),
        "slope": rng.integers(0, 3, n), "ca": rng.integers(0, 4, n),
        "thal": rng.integers(1, 4, n),
    })
    logit = 0.9 * X["ca"] + 0.6 * X["oldpeak"] + 0.8 * X["exang"] - 0.02 * (X["thalach"] - 150)
    y = (logit + rng.normal(0, 1, n) > 2).astype(int)
    return X, y


class FakeExplainer:
    """Mimics shap.TreeExplainer output formats (old list / new 3D array)."""
    def __init__(self, n_features, fmt):
        self.fmt = fmt
        self.n = n_features
        self.expected_value = np.array([0.6, 0.4])

    def shap_values(self, X):
        rng = np.random.default_rng(1)
        pos = rng.normal(0, 0.05, (len(X), X.shape[1]))
        pos[:, list(X.columns).index("ca") if "ca" in X.columns else 0] = 0.2  # strong riser
        if self.fmt == "list":
            return [-pos, pos]
        return np.stack([-pos, pos], axis=2)


def test_formats_and_mapping():
    X, y = make_data()
    model = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)
    patient = X.iloc[[0]]
    for fmt in ("list", "array3d"):
        res = ex.explain_patient(model, FakeExplainer(X.shape[1], fmt), patient)
        c = res["contributions"]
        assert len(c) == 13 and c.iloc[0]["feature"] == "ca", fmt
        assert c.iloc[0]["region"] == "coronary_arteries"
        assert abs(res["base_value"] - 0.4) < 1e-9
        regions = ex.region_scores(c)
        assert regions.iloc[0]["region"] == "coronary_arteries"
        assert ex.explain_in_words(res)[0].startswith("Narrowed major vessels")
    imp = ex.global_importance(model, FakeExplainer(X.shape[1], "array3d"), X)
    assert imp.iloc[0]["feature"] == "ca"
    ex.plot_patient(res); ex.plot_global(imp)
    print("format + mapping tests passed")


def test_onehot_grouping():
    X, y = make_data()
    Xo = pd.get_dummies(X, columns=["cp", "thal"], dtype=int)
    model = RandomForestClassifier(n_estimators=50, random_state=0).fit(Xo, y)
    res = ex.explain_patient(model, FakeExplainer(Xo.shape[1], "list"), Xo.iloc[[0]])
    feats = set(res["contributions"]["feature"])
    assert feats == set(COLS), feats
    cp_val = res["contributions"].set_index("feature").loc["cp", "value"]
    assert str(cp_val).startswith("cp_"), cp_val
    print("one-hot grouping test passed")


def _pipeline(X, y):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    return make_pipeline(SimpleImputer(), RandomForestClassifier(n_estimators=50, random_state=0)).fit(X, y)


def test_pipeline():
    X, y = make_data()
    model = _pipeline(X, y)
    res = ex.explain_patient(model, FakeExplainer(X.shape[1], "array3d"), X.iloc[[0]].astype(float))
    c = res["contributions"]
    assert set(c["feature"]) == set(COLS) and c.iloc[0]["feature"] == "ca"
    assert float(c.set_index("feature").loc["age", "value"]) == float(X.iloc[0]["age"])
    ex.global_importance(model, FakeExplainer(X.shape[1], "list"), X.astype(float))
    print("pipeline test passed")


def test_real_shap():
    try:
        import shap  # noqa: F401
    except ImportError:
        print("shap not installed - skipped real TreeExplainer test")
        return
    X, y = make_data()
    model = RandomForestClassifier(n_estimators=100, random_state=0).fit(X, y)
    explainer = ex.build_explainer(model)
    patient = X.iloc[[5]]
    res = ex.explain_patient(model, explainer, patient)
    # SHAP additivity: base value + contributions == predicted probability
    total = res["base_value"] + res["contributions"]["shap"].sum()
    assert abs(total - res["probability"]) < 1e-3, (total, res["probability"])

    pipe = _pipeline(X, y)
    res = ex.explain_patient(pipe, ex.build_explainer(pipe), patient.astype(float))
    total = res["base_value"] + res["contributions"]["shap"].sum()
    assert abs(total - res["probability"]) < 1e-3, (total, res["probability"])
    print("real SHAP test passed (additivity holds)")


if __name__ == "__main__":
    test_formats_and_mapping()
    test_onehot_grouping()
    test_pipeline()
    test_real_shap()
