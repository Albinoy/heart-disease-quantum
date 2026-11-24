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
- Classical: Random Forest, class-balanced; handles missing values itself; constrained so that higher blood pressure or cholesterol can never lower its estimate (unconstrained, the UCI data made them go the wrong way for 22% / 39% of patients on a test grid; the constraint also raised cross-validated AUC from 0.876 to 0.880).
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

- Six models in repeated cross-validation: RF on all 11 features, RF on 4 PCA
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

## Validation at hospitals the models have never seen (validate_hospitals.py)
`python validate_hospitals.py` (about 25 minutes) trains both models, with train.py's exact
settings, on four hospitals and tests them on the fifth, for each hospital in turn. This
measures how the models would do somewhere new; the headline test scores (random 20% split)
only measure new patients from hospitals the models already learned from.

| Held-out hospital | Patients | Had disease | Classical AUC (95% CI) | Quantum AUC (95% CI) |
|---|---|---|---|---|
| Cleveland (USA) | 303 | 46% | 0.85 (0.81-0.89) | 0.79 (0.73-0.84) |
| Budapest (Hungary) | 293 | 36% | 0.84 (0.79-0.89) | 0.86 (0.81-0.90) |
| Zurich/Basel (Switzerland) | 123 | 94% | 0.73 (0.47-0.95) | 0.65 (0.46-0.82) |
| Long Beach VA (USA) | 199 | 74% | 0.68 (0.59-0.77) | 0.69 (0.60-0.78) |
| Tehran (Iran) | 303 | 71% | 0.58 (0.51-0.66) | 0.60 (0.53-0.67) |
| **Average** | | | **0.74** | **0.72** |

Against 0.91 (classical) and 0.82 (quantum) on familiar hospitals. At Tehran, whose patients
had no exercise test, the classical model's average estimate was 22% although 71% had heart
disease, and it caught only 6% of them at the 50% cut-off. Switzerland's interval is very wide
because only 8 of its patients had no disease. Results: `results/hospital_validation.csv` / `.json`.

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

## Agents and checks
Five Claude Code subagents live in `.claude/agents/`. Ask Claude Code to use one by name
(for example "use the data-discovery-agent to look for datasets from Japan").

| Agent | Job | Can change files? |
|---|---|---|
| `research-agent` | Cited answers to medical, statistical and quantum-ML questions | No |
| `data-discovery-agent` | Finds datasets worldwide, judges real/duplicate/fake, updates `datasets_catalog.csv` | Catalog and `data/raw/` only |
| `data-engineering-agent` | Writes `convert_<name>.py`, runs the realism checks, adds `data/<name>.csv` | Data and converters |
| `ml-training-engine` | Backs up models, retrains, restores the backup if checks fail | Models, `train.py` settings |
| `evaluation-agent` | Model checks, app stress test, benchmark, writes `reports/` | Reports only |

None of them commits, pushes or touches the live site; that stays with you.

The checks they run are plain scripts you can run yourself:
- `python check_dataset.py <file.csv>`: does a dataset look like real patients? (flat
  distributions, risk factors pointing the wrong way, implausibly high accuracy, duplicates)
- `python check_model.py --baseline <old metrics.json>`: are new models sane and not worse? Includes risk-factor directions across a grid of 80 patients, not just one typical patient.
- `python stress_test_app.py`: submits a patient 15 times headless; catches the Qiskit crash.

`datasets_catalog.csv` lists every dataset assessed so far (282), marked Right or Wrong with the reason.

## Tried and removed: a "routine checks" model on 13,190 patients
To use more of the real data collected, a second Random Forest was trained on the five inputs
most datasets share (age, sex, blood pressure, cholesterol, fasting sugar) using five real
datasets: UCI, Tehran, Duke (USA, 3,504), Soochow (China, 7,647) and St. Gallen
(Switzerland, 816). It was shown briefly in the app as a "second opinion" and then removed
after `evaluation-agent` failed it (full report: `reports/evaluation-routine-2026-11-20.md`):

- **It learned a medically wrong pattern.** In the Soochow data, heart-attack patients had
  lower blood pressure than the other cardiology inpatients they were compared with (41%
  disease at BP <= 100 mmHg, 2% above 180). The model learned that, so raising blood pressure
  lowered its score for 38% of patients on a test grid (e.g. a 75-year-old man: 82% at BP 90,
  50% at BP 200).
- **The extra inputs added almost nothing beyond age and sex** (no significant gain in 4 of 5
  sources; on 422 South African men from SAheart, never trained on, age alone matched it).
- **It disagreed with the main model for about 1 in 4 patients**, and the main model was right
  in two-thirds of those cases.

Lessons: pooling datasets whose "disease" labels mean different things (angiography vs heart
attack) can teach a model artefacts of how patients were selected; and direction checks must
cover a grid of patients, not one typical patient. The code is in git history (commit 605495c).
