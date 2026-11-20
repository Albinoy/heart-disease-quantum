"""CardiaQ: heart-disease risk from routine tests, classical vs quantum.

Run:  streamlit run app.py   (after running train.py once)
"""
import json
import os
from explain_tab import render_explanation
from explain import region_scores
from heart3d import render_heart

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

import ui

st.set_page_config(page_title="CardiaQ", page_icon="🫀", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)


def chart(html_and_height):
    html, height = html_and_height
    components.html(html, height=height)


@st.cache_resource
def load_models():
    rf = joblib.load("rf_model.joblib")
    q = joblib.load("quantum_model.joblib")
    kernel = FidelityQuantumKernel(feature_map=ZZFeatureMap(feature_dimension=q["n_qubits"], reps=2))
    with open("metrics.json") as f:
        metrics = json.load(f)
    return rf, q, kernel, metrics


@st.cache_resource
def load_routine_model():
    """Second model (routine checks only), trained by train_routine.py. Optional."""
    if not (os.path.exists("routine_model.joblib") and os.path.exists("routine_metrics.json")):
        return None, None
    with open("routine_metrics.json") as f:
        return joblib.load("routine_model.joblib"), json.load(f)


ROUTINE_NAMES = {"uci": "UCI hospitals", "z_alizadeh": "Tehran", "duke": "Duke (USA)",
                 "soochow": "Soochow (China)", "stgallen": "St. Gallen (Switzerland)"}


@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        return None
    if path.endswith("cv_summary.csv"):
        return pd.read_csv(path, header=[0, 1], index_col=0).dropna(how="all")
    return pd.read_csv(path)


rf, q, kernel, metrics = load_models()
FEATURES = metrics["features"]
# data file -> (name in the data section, short name in the score text)
SOURCE_NAMES = {
    "uci_heart_disease.csv": ("UCI Heart Disease: Cleveland, Hungary, Switzerland and Long Beach VA", "UCI hospitals"),
    "z_alizadeh_sani.csv": ("Z-Alizadeh Sani: Rajaie Cardiovascular Center, Tehran, angiography-confirmed",
                            "Iranian patients"),
}
REF = metrics.get("reference", {})
M_RF, M_Q = metrics["Random Forest"], metrics["Quantum kernel SVM"]

# ------------------------------------------------------------------ inputs
CP = ["Typical angina", "Atypical angina", "Non-anginal pain", "No chest pain (asymptomatic)"]
ECG = ["Normal", "ST-T wave abnormality", "Left ventricle enlarged"]
SLOPE = ["Upsloping", "Flat", "Downsloping"]
THAL = ["Not recorded", "Fixed defect", "Normal", "Reversible defect"]
YESNO = ["No", "Yes"]

# feature: (label, kind, default, extra, help)
SPEC = {
    "age": ("Age", "num", 55, (20, 90, 1), "Years"),
    "sex": ("Sex", "sel", 1, ["Female", "Male"], None),
    "cp": ("Chest pain type", "sel", 3, CP, "The kind of chest pain the patient reports"),
    "trestbps": ("Resting blood pressure", "num", 130, (80, 200, 1), "Systolic, in mm Hg"),
    "chol": ("Cholesterol", "num", 240, (100, 600, 1), "Serum cholesterol, in mg/dl"),
    "fbs": ("Fasting blood sugar above 120 mg/dl", "sel", 0, YESNO, None),
    "restecg": ("Resting ECG", "sel", 0, ECG, "Electrocardiogram result at rest"),
    "thalach": ("Maximum heart rate", "num", 150, (60, 220, 1), "Highest rate reached in an exercise test"),
    "exang": ("Chest pain during exercise", "sel", 0, YESNO, "Exercise-induced angina"),
    "oldpeak": ("ST depression", "num", 1.0, (0.0, 7.0, 0.1), "Drop in the ECG ST segment during exercise, in mm"),
    "slope": ("ST segment slope", "sel", 1, SLOPE, "Shape of the ST segment at peak exercise"),
    "ca": ("Major vessels seen on fluoroscopy", "sel", 0, ["0", "1", "2", "3", "4"], None),
    "thal": ("Thallium stress test", "sel", 2, THAL, None),
}
SHORT = {"age": "Age", "sex": "Sex", "cp": "Chest pain type", "trestbps": "Blood pressure",
         "chol": "Cholesterol", "fbs": "Fasting blood sugar", "restecg": "Resting ECG",
         "thalach": "Max heart rate", "exang": "Exercise chest pain", "oldpeak": "ST depression",
         "slope": "ST slope", "ca": "Vessels on fluoroscopy", "thal": "Thallium test"}
UNITS = {"age": " yrs", "trestbps": " mm Hg", "chol": " mg/dl", "thalach": " bpm", "oldpeak": " mm"}


def describe(f, v):
    kind, extra = SPEC[f][1], SPEC[f][3]
    if kind == "sel":
        return extra[int(v)] if 0 <= int(v) < len(extra) else str(v)
    return f"{v:g}{UNITS.get(f, '')}"


def fmt(f, v):
    return f"{v:.1f}{UNITS.get(f, '')}" if f == "oldpeak" else f"{v:.0f}{UNITS.get(f, '')}"


def predict(row):
    p_rf = float(rf.predict_proba(row)[0][1])
    x = q["prep"].transform(row) * q["angle_scale"] * np.pi
    score = q["svc"].decision_function(kernel.evaluate(x_vec=x, y_vec=q["Xq_train"]))
    if "calib" in q:
        p_q = float(q["calib"].predict_proba(np.asarray(score).reshape(-1, 1))[0][1])
    else:
        p_q = float(1 / (1 + np.exp(-score[0])))
    return p_rf, p_q


def effects(row, p_rf):
    """How much each answer moves the classical estimate, compared with a typical value."""
    out = []
    for f in FEATURES:
        if f not in REF:
            continue
        alt = row.copy()
        alt[f] = REF[f]["typical"]
        if float(alt[f].iloc[0]) == float(row[f].iloc[0]):
            continue
        d = p_rf - float(rf.predict_proba(alt)[0][1])
        if abs(d) >= 0.005:          # hide changes under half a percentage point
            out.append((f, d))
    return sorted(out, key=lambda t: -abs(t[1]))


def benchmark_summary(cv):
    """Plain-language reading of the cross-validated results."""
    if ("roc_auc", "mean") not in cv.columns:
        return ""
    auc = lambda m: float(cv.loc[m, ("roc_auc", "mean")])
    sd = lambda m: float(cv.loc[m, ("roc_auc", "std")])
    qn = [m for m in cv.index if "Quantum" in m]
    cn = [m for m in cv.index if "Quantum" not in m and "default" not in m]
    if not qn or not cn:
        return ""
    qb, cb = max(qn, key=auc), max(cn, key=auc)
    diff, noise = auc(cb) - auc(qb), max(sd(qb), sd(cb))
    text = (f"The best quantum model, {qb}, scored an ROC AUC of {auc(qb):.2f}. The best classical model, "
            f"{cb}, scored {auc(cb):.2f}. ")
    if abs(diff) <= noise:
        text += "That gap is within the spread of the results, so the two are effectively tied. "
    elif diff > 0:
        text += "The classical model is ahead by more than the spread of the results. "
    else:
        text += ("The quantum model is ahead by more than the spread. Check this holds on new data "
                 "before calling it an advantage. ")
    # Like-for-like: classical models that see the same 4 compressed features as the quantum model.
    same = [m for m in cn if "PCA" in m or "SVM" in m]
    if same and cb not in same:
        sb = max(same, key=auc)
        sdiff, snoise = auc(sb) - auc(qb), max(sd(qb), sd(sb))
        text += (f"On the same 4 compressed features, the best classical model, {sb}, scored {auc(sb):.2f}, "
                 + ("so there the two are effectively tied. " if abs(sdiff) <= snoise else
                    "ahead by more than the spread. " if sdiff > 0 else "behind the quantum model. "))
    untuned = [m for m in qn if "default" in m]
    if untuned and auc(qb) - auc(untuned[0]) > noise:
        text += (f"Untuned, the quantum model reached only {auc(untuned[0]):.2f}, so tuning its angle "
                 "scale made most of the difference.")
    return text


def level_of(p):
    if p < 0.3:
        return "Low", ui.GREEN
    if p < 0.6:
        return "Moderate", ui.AMBER
    return "High", ui.RED


# ------------------------------------------------------------------ hero
st.markdown(ui.hero_html(), unsafe_allow_html=True)
st.markdown(
    '<div class="cq-note"><b>For learning and research only.</b> CardiaQ is not a medical device and '
    "cannot diagnose anyone. If you are worried about your heart, contact a doctor, or call NHS 111. "
    "In an emergency, call 999.</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ form
st.markdown(ui.section("Check a patient",
                       "Enter the results from a routine heart check-up. Both models give an estimate, "
                       "and CardiaQ explains what drove it.", anchor="check-a-patient"),
            unsafe_allow_html=True)

with st.form("patient"):
    cols = st.columns(3)
    values = {}
    for i, f in enumerate(FEATURES):
        label, kind, default, extra, help_ = SPEC[f]
        with cols[i % 3]:
            if kind == "num":
                lo, hi, step = extra
                values[f] = st.number_input(label, min_value=lo, max_value=hi, value=default,
                                            step=step, help=help_)
            else:
                values[f] = st.selectbox(label, list(range(len(extra))), index=default,
                                         format_func=lambda v, e=extra: e[v], help=help_)
    submitted = st.form_submit_button("Check risk")

if submitted:
    row = pd.DataFrame([values])[FEATURES].astype(float)
    with st.spinner("Running the quantum circuit simulation..."):
        p_rf, p_q = predict(row)
        st.session_state["result"] = {"values": values, "p_rf": p_rf, "p_q": p_q,
                                      "effects": effects(row, p_rf)}

# ------------------------------------------------------------------ result
res = st.session_state.get("result")
if res:
    p_rf, p_q, vals = res["p_rf"], res["p_q"], res["values"]
    avg = (p_rf + p_q) / 2
    level, color = level_of(avg)
    st.markdown(ui.section("Result", "The classical estimate closely matches the share of similar patients "
                                     "in the training data who had heart disease. The quantum estimate is "
                                     "rougher, and tends to be too low for patients like the Iranian group, "
                                     "who had no exercise test."), unsafe_allow_html=True)
    chart(ui.ruler_html(p_rf, p_q, level, color))

    agree = (p_rf >= 0.5) == (p_q >= 0.5)
    verdict = {
        "Low": "Both models see a pattern that mostly matched patients without heart disease.",
        "Moderate": "The pattern sits between the groups, so the models are less certain here.",
        "High": "The pattern closely matched patients who had heart disease in the training data.",
    }[level]
    st.markdown(
        f'<p class="cq-read">The classical model estimates <b>{p_rf:.0%}</b> and the quantum model '
        f"<b>{p_q:.0%}</b>. {verdict} "
        + ("The two models agree on which side of 50% this patient falls."
           if agree else "<b>The two models disagree</b>, which is common for borderline cases.")
        + "</p>", unsafe_allow_html=True)

    routine, rm = load_routine_model()
    if routine is not None:
        rrow = pd.DataFrame([{f: vals.get(f) for f in rm["features"]}]).astype(float)
        p_routine = float(routine.predict_proba(rrow)[0][1])
        srcs = rm["sources"]
        st.markdown(
            '<div class="cq-sub">Second opinion: routine checks only</div>'
            f'<p class="cq-read">A separate model that sees only age, sex, blood pressure, cholesterol and '
            f"fasting blood sugar, but was trained on <b>{rm['n_patients']:,} real patients</b> from five "
            f"hospital groups ({', '.join(ROUTINE_NAMES.get(s, s) for s in srcs)}), gives this patient a score "
            f"of <b>{p_routine:.0%}</b>.</p>"
            '<p class="cq-small">How to read it: the score is balanced so that 50% is the average patient '
            "across the five groups; it is not the chance of disease in any real population. It cannot use chest "
            "pain or exercise-test results, so it is much less accurate than the models above (ROC AUC "
            f"{min(v['roc_auc'] for v in rm['by_source'].values()):.2f}-"
            f"{max(v['roc_auc'] for v in rm['by_source'].values()):.2f} per hospital group, against "
            f"{M_RF.get('roc_auc', 0):.2f} for the classical model). It shows how much routine checks alone can "
            "tell, and the five groups define disease differently (angiography in four, heart attack in "
            "Soochow).</p>", unsafe_allow_html=True)

    eff = res["effects"]
    if eff:
        st.markdown('<div class="cq-sub">What moved the estimate</div>'
                    '<p class="cq-small">For each answer, CardiaQ swaps in a typical value and measures '
                    "how much the classical estimate changes. Red bars push the risk up; green bars pull "
                    "it down. Answers that match the typical value, or barely change it, are not shown.</p>",
                    unsafe_allow_html=True)
        items = [{"label": SHORT[f], "sub": describe(f, vals[f]), "value": d,
                  "display": f"{d * 100:+.0f}", "color": ui.RED if d > 0 else ui.GREEN}
                 for f, d in eff[:8]]
        chart(ui.bars_html(items, diverging=True, axis=("Lowers risk", "Raises risk")))
        st.markdown('<p class="cq-small">Numbers are percentage points. This is an approximation for one '
                    "model; it shows association in the data, not cause.</p>", unsafe_allow_html=True)

    # ---- SHAP explanation (Phase 1) + 3D heart (Phase 2)
    shap_row = pd.DataFrame([vals])[FEATURES].astype(float)
    result = render_explanation(rf, None, shap_row)

    st.markdown(ui.section("Where in the heart",
                           "Parts linked to this patient's higher risk pulse red. Click any part to learn about it."),
                unsafe_allow_html=True)
    render_heart(region_scores(result["contributions"]))

    rows = []
    for f in [f for f in ["age", "trestbps", "chol", "thalach", "oldpeak"] if f in FEATURES and "quantiles" in REF.get(f, {})]:
        qs, v = REF[f]["quantiles"], float(vals[f])
        pct = float(np.interp(v, qs, np.linspace(0, 100, len(qs))))
        rows.append({"label": SHORT[f], "text": describe(f, v), "pct": pct,
                     "note": f"Higher than {pct:.0f}% of patients. Typical: {fmt(f, REF[f]['median_healthy'])} "
                             f"without heart disease, {fmt(f, REF[f]['median_disease'])} with it."})
    if rows:
        st.markdown('<div class="cq-sub">How this patient compares</div>', unsafe_allow_html=True)
        chart(ui.percentile_html(rows))

    report = "\n".join(
        ["# CardiaQ result", "", "For learning and research only. Not a diagnosis.", "",
         f"- Classical model (Random Forest): {p_rf:.0%}",
         f"- Quantum model (quantum kernel SVM, simulated): {p_q:.0%}",
         f"- Combined signal: {level}", "", "## Inputs"]
        + [f"- {SPEC[f][0]}: {describe(f, vals[f])}" for f in FEATURES]
        + ["", "## What moved the classical estimate (percentage points)"]
        + [f"- {SHORT[f]}: {d * 100:+.1f}" for f, d in eff])
    st.download_button("Download this result", report, file_name="cardiaq-result.md", mime="text/markdown")

# ------------------------------------------------------------------ test results
st.markdown(ui.section("How well the models work",
                       f"Both models were tested on {metrics['n_test']} patients they never saw during "
                       "training. Here is what happened, in plain terms."), unsafe_allow_html=True)

st.markdown(
    '<div class="cq-def">'
    "<div><b>Recall</b><p>Of the people who had heart disease, the share the model caught. "
    "The most important number for screening.</p></div>"
    "<div><b>Precision</b><p>Of the people the model flagged, the share who really had heart disease.</p></div>"
    "<div><b>Accuracy</b><p>The share of all predictions that were right.</p></div>"
    "<div><b>F1</b><p>A balance of recall and precision in one number.</p></div>"
    "<div><b>ROC AUC</b><p>How well the model ranks sick patients above healthy ones. "
    "0.5 is guessing, 1.0 is perfect.</p></div></div>", unsafe_allow_html=True)

if "confusion" in M_RF and "confusion" in M_Q:
    chart(ui.waffle_html([{"name": "Classical", "color": ui.INK, "confusion": M_RF["confusion"]},
                          {"name": "Quantum", "color": ui.TEAL, "confusion": M_Q["confusion"]}]))

items = []
for key, name in [("recall", "Recall"), ("precision", "Precision"), ("accuracy", "Accuracy"),
                  ("f1", "F1"), ("roc_auc", "ROC AUC")]:
    if key in M_RF and key in M_Q:
        items.append({"label": name, "sub": "Classical", "value": M_RF[key],
                      "display": f"{M_RF[key]:.2f}", "color": ui.INK})
        items.append({"label": "", "sub": "Quantum", "value": M_Q[key],
                      "display": f"{M_Q[key]:.2f}", "color": ui.TEAL})
st.markdown('<div class="cq-sub">Scores on the test patients</div>', unsafe_allow_html=True)
chart(ui.bars_html(items, vmax=1.0))

gap = M_RF.get("roc_auc", 0) - M_Q.get("roc_auc", 0)
st.markdown(
    f'<p class="cq-read">On this test set the classical model scored an ROC AUC of {M_RF.get("roc_auc", 0):.2f} '
    f'and the quantum model {M_Q.get("roc_auc", 0):.2f}, a difference of {abs(gap):.2f}. With '
    f"{metrics['n_test']} test patients, differences of a few hundredths can come from which patients "
    "happened to be in the test set, so the cross-validated benchmark below is the fairer comparison.</p>",
    unsafe_allow_html=True)

rf_src, q_src = M_RF.get("by_source", {}), M_Q.get("by_source", {})
if len(rf_src) > 1:
    parts = [f"{SOURCE_NAMES.get(s, (s, s))[1]} ({r['n_test']} patients): classical {r['roc_auc']:.2f}, "
             f"quantum {q_src.get(s, {}).get('roc_auc', float('nan')):.2f}" for s, r in rf_src.items()]
    st.markdown('<p class="cq-read">ROC AUC by data source. ' + "; ".join(parts) + ". The Iranian "
                "patients never had an exercise test, so both models see less about them.</p>",
                unsafe_allow_html=True)

# ------------------------------------------------------------------ benchmark
cv = load_csv("results/cv_summary.csv")
exp = load_csv("results/quantum_experiments.csv")
if cv is not None or exp is not None:
    st.markdown(ui.section("Classical vs quantum, tested fairly",
                           "Repeated cross-validation: every model is trained and tested many times on "
                           "different splits, and tuned settings are chosen without looking at the test "
                           "patients. Whiskers show how much the score varied."), unsafe_allow_html=True)
    st.markdown('<p class="cq-small">Run on all patients from both datasets, 5 folds repeated twice. To keep '
                "the comparison fair, every 4-feature model gets the same 120 training patients per fold, so "
                "the quantum score here is lower than the app's quantum model, which trains on 240.</p>",
                unsafe_allow_html=True)
if cv is not None:
    models = list(cv.index)
    labels = [m.replace(" (", "<br>(") for m in models]
    colors = [ui.TEAL if "Quantum" in m else ("#8A93A6" if "default" in m else ui.INK) for m in models]
    tabs = {}
    for key, name in [("roc_auc", "ROC AUC"), ("recall", "Recall"), ("accuracy", "Accuracy")]:
        if key in cv.columns.get_level_values(0):
            tabs[name] = [(float(cv.loc[m, (key, "mean")]), float(cv.loc[m, (key, "std")])) for m in models]
    chart(ui.vbars_html(labels, tabs, colors))
    st.markdown(f'<p class="cq-read">{benchmark_summary(cv)}</p>', unsafe_allow_html=True)

if exp is not None:
    qb = exp[exp["experiment"] == "qubits"]
    if len(qb) >= 2:
        st.markdown('<div class="cq-sub">The cost of simulating qubits</div>'
                    '<p class="cq-small">Time to train the quantum model on a laptop as the number of qubits '
                    "grows. Each extra qubit doubles the size of the state the computer must track.</p>",
                    unsafe_allow_html=True)
        chart(ui.line_html(list(qb["setting"].astype(str)), list(qb["seconds"].astype(float)),
                           "Number of qubits", "Training time"))
    sc = exp[exp["experiment"] == "angle_scale"]
    if len(sc) >= 2:
        st.markdown('<div class="cq-sub">Why tuning mattered</div>'
                    '<p class="cq-small">The angle scale sets how strongly patient values rotate the qubits. '
                    "Too large and every patient looks unrelated to every other, so the model cannot learn. "
                    "Scores are ROC AUC on validation patients.</p>", unsafe_allow_html=True)
        chart(ui.bars_html([{"label": s.replace("*pi", " × π"), "value": float(a), "display": f"{a:.2f}",
                             "color": ui.TEAL} for s, a in zip(sc["setting"], sc["roc_auc"])], vmax=1.0))

# ------------------------------------------------------------------ how it works
st.markdown(ui.section("How the quantum model works"), unsafe_allow_html=True)
st.markdown(
    '<ol class="cq-steps">'
    f"<li>Each patient's results are compressed into {q['n_qubits']} numbers and turned into rotation "
    f"angles for {q['n_qubits']} simulated qubits.</li>"
    "<li>A quantum circuit entangles the qubits. Comparing two patients' circuits gives a similarity "
    "score between 0 and 1.</li>"
    "<li>A standard support vector machine uses those similarity scores to separate patients with and "
    "without heart disease.</li></ol>", unsafe_allow_html=True)
st.markdown(f'<p class="cq-small">Chosen by tuning: angle scale {metrics.get("angle_scale", "?")} × π, '
            f'C = {metrics.get("svm_C", "?")}. Trained on {metrics.get("n_train_quantum", "?")} patients, '
            "because the simulation cost grows quickly with data size.</p>", unsafe_allow_html=True)
with st.expander("Show the quantum circuit"):
    fig = ZZFeatureMap(feature_dimension=q["n_qubits"], reps=2).decompose().draw("mpl")
    st.pyplot(fig)

# ------------------------------------------------------------------ data and limits
info = metrics.get("data", {})
st.markdown(ui.section("Data and limits"), unsafe_allow_html=True)
if info:
    files = ", ".join(f"{SOURCE_NAMES.get(k, (k, k))[0]} ({v} patients)" for k, v in info.get("files", {}).items())
    dropped = info.get("dropped_features", {})
    st.markdown(
        f'<p class="cq-read"><b>Classical and quantum models:</b> trained on {info.get("rows", "?")} patients '
        f'from {files}, the only open real-patient data with exercise-test results. '
        f'{info.get("duplicates_removed", 0)} duplicate records were removed. '
        f'{info.get("positive_rate", 0):.0%} of patients had heart disease.'
        + (f" Not used because most records lack them: {', '.join(SHORT.get(c, c) for c in dropped)}." if dropped else "")
        + "</p>", unsafe_allow_html=True)
_, routine_info = load_routine_model()
if routine_info:
    parts = ", ".join(f"{ROUTINE_NAMES.get(s, s)} ({v['patients']:,})" for s, v in routine_info["sources"].items())
    st.markdown(
        f'<p class="cq-read"><b>Routine-checks model:</b> trained on {routine_info["n_patients"]:,} patients '
        f"from five real datasets: {parts}. It uses only age, sex, blood pressure, cholesterol and fasting "
        "blood sugar, because those are what all five recorded.</p>", unsafe_allow_html=True)
st.markdown(
    '<ul class="cq-limits">'
    "<li>The public datasets are small: the UCI patients were seen at four hospitals in the 1980s and the "
    "Iranian patients at one centre in Tehran, so the models may not reflect your population.</li>"
    "<li>The Iranian patients had no exercise test, and many UCI patients lack some results. The classical "
    "model handles these gaps itself; the quantum model fills them with typical values, which hides real "
    "uncertainty.</li>"
    "<li>The quantum model is weaker for the Iranian patients: on its test patients its estimates averaged "
    "58% when 70% had heart disease, and for Iranian men it ranked patients no better than chance (based "
    "on only 33 test patients).</li>"
    "<li>Chest pain works backwards in the UCI data: typical angina gives a lower estimate than no chest "
    "pain, because UCI patients without chest pain were usually sent for angiography only when other tests "
    "already pointed to disease. The Iranian data shows the expected pattern (94% of patients with typical "
    "angina had disease), but the model cannot tell which group a new patient resembles. Clinically, angina "
    "is a warning sign.</li>"
    "<li>The quantum model runs on a simulator on an ordinary computer, not on quantum hardware.</li>"
    "<li>Neither model has been clinically validated, and medical software in the UK must be approved "
    "by the MHRA before clinical use.</li></ul>", unsafe_allow_html=True)
st.markdown('<div class="cq-foot">CardiaQ, a student research project. Built with scikit-learn, Qiskit '
            "and Streamlit.</div>", unsafe_allow_html=True)
