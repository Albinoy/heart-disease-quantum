# Heart Disease Prediction: Classical vs Quantum (Streamlit + Qiskit)

Benchmarks a Random Forest against a quantum-kernel SVM (simulated with Qiskit)
on the UCI Heart Disease dataset, with a Streamlit app to try both side by side.

> Educational project. Not medical advice. Not clinically validated.

## Setup
1. Download `heart.csv` (UCI Heart Disease, e.g. from Kaggle) into this folder.
   Expected columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach,
   exang, oldpeak, slope, ca, thal, target (1 = disease).
2. `pip install -r requirements.txt`
3. `python train.py`   (trains both models, saves them, writes metrics.json)
4. `streamlit run app.py`

## Method
- Classical: Random Forest, class-balanced.
- Quantum: standardise, PCA to 4 components, scale to [0, pi], ZZFeatureMap
  (4 qubits, 2 reps), fidelity quantum kernel, SVM on the precomputed kernel.
- Quantum training is capped at 200 rows because kernel cost grows with n^2.
- Metrics reported: accuracy, recall, precision, F1, training time.

## Limitations
- Small, old dataset from specific hospitals; will not generalise.
- No proven quantum advantage on small tabular data; runs on a simulator.
- Not a medical device (UK: MHRA regulated).

## Ideas to extend
- Compare ZZFeatureMap vs PauliFeatureMap, vary reps and qubits.
- Cross-validation and SHAP explanations.
- Check performance by sex and age group.
- Run one experiment on an IBM quantum backend and discuss noise.
