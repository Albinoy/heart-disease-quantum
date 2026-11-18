"""Streamlit app: classical vs quantum heart disease prediction.

Run:  streamlit run app.py   (after running train.py once)
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

st.set_page_config(page_title="Heart Disease: Classical vs Quantum", page_icon="❤️", layout="wide")


@st.cache_resource
def load_models():
    rf = joblib.load("rf_model.joblib")
    q = joblib.load("quantum_model.joblib")
    feature_map = ZZFeatureMap(feature_dimension=q["n_qubits"], reps=2)
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    with open("metrics.json") as f:
        metrics = json.load(f)
    return rf, q, feature_map, kernel, metrics


rf, q, feature_map, kernel, metrics = load_models()

st.title("❤️ Heart disease prediction: classical vs quantum")
st.warning(
    "Educational project only. Not medical advice and not a diagnostic tool. "
    "The model is trained on a small, old dataset and has not been clinically validated. "
    "If you are worried about your health, speak to a doctor or NHS 111."
)

tab_predict, tab_compare, tab_bench, tab_circuit, tab_about = st.tabs(
    ["Predict", "Model comparison", "Benchmark", "Quantum circuit", "About"]
)

# ---------------- Inputs ----------------
with st.sidebar:
    st.header("Patient values")
    age = st.slider("Age", 20, 90, 55)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda v: {0: "Female", 1: "Male"}[v])
    cp = st.selectbox(
        "Chest pain type", [0, 1, 2, 3],
        format_func=lambda v: ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"][v],
    )
    trestbps = st.slider("Resting blood pressure (mm Hg)", 80, 200, 130)
    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1], format_func=lambda v: "Yes" if v else "No")
    restecg = st.selectbox(
        "Resting ECG", [0, 1, 2],
        format_func=lambda v: ["Normal", "ST-T abnormality", "LV hypertrophy"][v],
    )
    thalach = st.slider("Max heart rate achieved", 60, 220, 150)
    exang = st.selectbox("Exercise-induced angina", [0, 1], format_func=lambda v: "Yes" if v else "No")
    oldpeak = st.slider("ST depression (oldpeak)", 0.0, 7.0, 1.0, 0.1)
    slope = st.selectbox(
        "Slope of peak exercise ST segment", [0, 1, 2],
        format_func=lambda v: ["Upsloping", "Flat", "Downsloping"][v],
    )
    ca = st.selectbox("Major vessels coloured by fluoroscopy", [0, 1, 2, 3, 4])
    thal = st.selectbox(
        "Thalassemia", [0, 1, 2, 3],
        format_func=lambda v: ["Unknown/0", "Fixed defect", "Normal", "Reversible defect"][v],
    )
    go = st.button("Predict", type="primary")

row = pd.DataFrame([{
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol, "fbs": fbs,
    "restecg": restecg, "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
    "slope": slope, "ca": ca, "thal": thal,
}])[metrics["features"]]

# ---------------- Predict ----------------
with tab_predict:
    if not go:
        st.info("Set the values in the sidebar and press **Predict**.")
    else:
        rf_prob = float(rf.predict_proba(row)[0][1])

        x_q = q["prep"].transform(row) * q["angle_scale"] * np.pi
        k_row = kernel.evaluate(x_vec=x_q, y_vec=q["Xq_train"])
        q_prob = float(q["svc"].predict_proba(k_row)[0][1])

        c1, c2 = st.columns(2)
        for col, name, p in [(c1, "Classical (Random Forest)", rf_prob),
                             (c2, "Quantum kernel SVM (simulated)", q_prob)]:
            with col:
                st.subheader(name)
                st.metric("Estimated probability of heart disease", f"{p:.0%}")
                st.progress(min(max(p, 0.0), 1.0))
                st.write("Flagged for check-up" if p >= 0.5 else "Not flagged")

        if (rf_prob >= 0.5) != (q_prob >= 0.5):
            st.info("The two models disagree here. That is normal for borderline cases and is worth discussing in your write-up.")

        st.subheader("What drives the classical model overall?")
        imp = pd.Series(rf.feature_importances_, index=metrics["features"]).sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        imp.plot.barh(ax=ax)
        ax.set_xlabel("Feature importance")
        st.pyplot(fig)

# ---------------- Compare ----------------
with tab_compare:
    st.subheader("Held-out test set results")
    table = pd.DataFrame({k: metrics[k] for k in ["Random Forest", "Quantum kernel SVM"]}).T
    st.dataframe(table.style.format("{:.3f}"))
    st.caption(
        f"Test patients: {metrics['n_test']}. Classical training rows: {metrics['n_train_classical']}. "
        f"Quantum training rows: {metrics['n_train_quantum']} (capped because kernel cost grows with n²). "
        f"Quantum angle scale: {metrics['angle_scale']}π, SVM C: {metrics['svm_C']} (chosen by cross-validation on training rows)."
    )
    st.write(
        "**Recall** matters most here: it measures how many truly sick patients the model catches. "
        "Training time for the quantum model is for a classical *simulator* of the circuit."
    )

# ---------------- Benchmark ----------------
with tab_bench:
    st.subheader("Cross-validated benchmark")
    if os.path.exists("results/cv_summary.csv"):
        s = pd.read_csv("results/cv_summary.csv", header=[0, 1], index_col=0).dropna(how="all")
        tbl = pd.DataFrame({
            m: s[m]["mean"].map("{:.3f}".format) + " ± " + s[m]["std"].map("{:.3f}".format)
            for m in ["accuracy", "recall", "roc_auc"]
        })
        tbl["fit time (s)"] = s["seconds"]["mean"].map("{:.3f}".format)
        st.table(tbl)
        st.caption("Mean ± standard deviation over repeated cross-validation folds. "
                   "Tuned models chose their settings using training rows only.")
    for path, caption in [
        ("results/cv_comparison.png", "Cross-validated comparison"),
        ("results/roc_curves.png", "ROC curves on one held-out split"),
        ("results/exp_angle_scale.png", "Quantum model: effect of angle scale"),
        ("results/exp_qubits_time.png", "Simulation time grows exponentially with qubits"),
    ]:
        if os.path.exists(path):
            st.image(path, caption=caption)
    st.markdown(
        """
**Findings**
- With default settings, the quantum kernel SVM performed far worse than the classical models.
- After tuning its angle scale and C, it matched the tuned RBF-SVM within the noise of the cross-validation, but did not beat it.
- Simulating the quantum kernel is orders of magnitude slower, and the cost grows exponentially with the number of qubits.
- On this small tabular dataset there is no evidence of a quantum advantage.
        """
    )

# ---------------- Circuit ----------------
with tab_circuit:
    st.subheader("The quantum feature map (ZZFeatureMap)")
    st.write(
        "Each patient is reduced to 4 numbers (PCA), scaled to angles, and encoded into a 4-qubit circuit. "
        "The kernel measures how similar two patients' quantum states are, and a standard SVM uses "
        "those similarity scores to classify."
    )
    fig = feature_map.decompose().draw("mpl")
    st.pyplot(fig)

# ---------------- About ----------------
with tab_about:
    st.markdown(
        """
**Limitations**
- Small dataset from specific hospitals; it will not generalise to the wider population.
- Quantum models show no proven advantage on small tabular data like this; results are for comparison.
- Runs on a simulator, not quantum hardware.
- Not clinically validated, and not approved as a medical device (in the UK, medical software is regulated by the MHRA).
        """
    )
