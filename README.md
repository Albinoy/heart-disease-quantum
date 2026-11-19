# CardiaQ: heart-disease risk, classical vs quantum (Streamlit + Qiskit)

Benchmarks a Random Forest against a quantum-kernel SVM (simulated with Qiskit)
on real-patient heart disease data from five hospitals, with a Streamlit app to
try both side by side.

> Educational project. Not medical advice. Not clinically validated.

## Data (real patients only)
| File | Patients | Source | Licence |
|---|---|---|---|
| `data/uci_heart_disease.csv` | 920 | UCI Heart Disease: Cleveland Clinic, Hungarian Institute of Cardiology (Budapest), University Hospitals Zurich and Basel, Long Beach VA, 1980s. Janosi, Steinbrunn, Pfisterer, Detrano (1988), [UCI 45](https://archive.ics.uci.edu/dataset/45/heart+disease) | CC BY 4.0 |
| `data/z_alizadeh_sani.csv` | 303 | Z-Alizadeh Sani: Rajaie Cardiovascular Center, Tehran; CAD confirmed by angiography. Alizadehsani et al., *Computer Methods and Programs in Biomedicine* (2013), [UCI 412](https://archive.ics.uci.edu/dataset/412/z+alizadeh+sani) | CC BY 4.0 |

`data/z_alizadeh_sani.csv` is made from `data/raw/` by `python convert_z_alizadeh_sani.py`,
which documents every field mapping. These patients had no exercise test, so maximum heart
rate, exercise angina, ST depression and ST slope are missing for them.

Checked and **not** used: the Kaggle 918/1,190-row "heart" files and the 2026 Mendeley
918-row file (copies of the UCI patients), the Mendeley "Cardiovascular Disease Dataset"
from India (patterns inconsistent with real patients, e.g. exercise angina unrelated to
disease and a within-dataset AUC of 0.997), the Kaggle Framingham file (a statistically
altered teaching dataset) and the Kaggle "Cardio 70k" file (no documented origin).

## Setup
1. Datasets go in `data/` (every CSV directly in that folder is combined). Supported
   formats: CardiaQ's own columns, UCI Cleveland (numeric or text), the UCI multi-hospital
   file, Kaggle "Heart Failure Prediction" (fedesoriano), and the Kaggle
   statlog/Cleveland/Hungary combined file. Duplicate patients are removed, and features
   missing in more than 60% of rows are dropped (see `data_prep.py`).
2. `pip install -r requirements.txt`
3. `python train.py`   (trains both models, saves them, writes metrics.json)
4. `streamlit run app.py`

## Method
- Classical: Random Forest, class-balanced; handles missing values itself.
- Quantum: fill gaps with the median, standardise, PCA to 4 components, scale to
  [0, pi], ZZFeatureMap (4 qubits, 2 reps), fidelity quantum kernel, SVM on the
  precomputed kernel.
- Quantum training is capped at 240 rows because kernel cost grows with n^2.
- The test split is stratified by data source; test ROC AUC is also reported per source.
- Metrics reported: accuracy, recall, precision, F1, ROC AUC, training time.

## Limitations
- Small, old datasets from specific hospitals; will not generalise.
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
