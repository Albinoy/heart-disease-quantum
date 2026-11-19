# CardiaQ: heart-disease risk, classical vs quantum (Streamlit + Qiskit)

Benchmarks a Random Forest against a quantum-kernel SVM (simulated with Qiskit)
on the UCI Heart Disease dataset, with a Streamlit app to try both side by side.

> Educational project. Not medical advice. Not clinically validated.

## Setup
1. Put your datasets in `data/` (any number of CSV files), or a single
   `heart.csv` in this folder. Supported formats: UCI Cleveland (numeric or
   text), the UCI multi-hospital file, Kaggle "Heart Failure Prediction"
   (fedesoriano), and the Kaggle statlog/Cleveland/Hungary combined file.
   Files are combined, duplicate patients removed, and features missing in
   more than half the rows are dropped (see `data_prep.py`).
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

## Benchmark (benchmark.py)
`python benchmark.py --quick` (check it works) or `python benchmark.py` (full run).
Results are written to `results/`.

- Six models in repeated cross-validation: RF on 13 features, RF on 4 PCA
  features, RBF-SVM (default and tuned), quantum kernel SVM (default and tuned).
- Tuned models choose their settings (quantum: angle scale and C; RBF-SVM:
  C and gamma) by inner CV on training rows only, never on the test fold.
- Quantum experiments (angle scale, feature map, qubits, training size, C) are
  scored on a validation split, not the test set.
- Report mean ± std across folds, and say when differences fall within noise.

## Further tuning (benchmark_v2.py)
`python benchmark_v2.py --quick`, then `python benchmark_v2.py`. Results go to `results_v2/`.
Tests, with nested cross-validation: PCA vs the 4 most important original features,
a finer angle-scale grid, and a hybrid kernel (quantum + RBF). Paired per-fold AUC
differences are reported so small gaps are not over-claimed.

## Deploying as a website
The app needs these files in the repository: `app.py`, `ui.py`, `explain.py`,
`explain_tab.py`, `heart3d.py`, `heart.glb`, `rf_model.joblib`,
`quantum_model.joblib`, `metrics.json`, `results/` and `.streamlit/config.toml`.
`requirements.txt` is pinned to the tested versions. Keep `qiskit==2.4.2`:
Qiskit 2.5 and later crash (segmentation fault) when run inside Streamlit.

### Option A: Streamlit Community Cloud (free, simplest)
1. Push the repository to GitHub.
2. Sign in at share.streamlit.io with GitHub and click **Create app**.
3. Pick the repository and branch, set the main file to `app.py`.
4. Under **Advanced settings**, choose Python 3.14 (the tested version), then **Deploy**.
You get a public `https://<name>.streamlit.app` link. Each push to the branch redeploys.

### Option B: any Docker host (Hugging Face Spaces, Render, Railway, Fly.io, Cloud Run)
```
docker build -t cardiaq .
docker run -p 8501:8501 cardiaq      # then open http://localhost:8501
```
The container listens on `$PORT` (default 8501), which most hosts set for you.
- Hugging Face Spaces: create a Space with the **Docker** SDK, push this repo to it,
  and add `app_port: 8501` to the Space's README front matter.
- Render / Railway: create a web service from the GitHub repo; they detect the Dockerfile.

The quantum model is simulated on the CPU, so the free tiers of these hosts are enough.
