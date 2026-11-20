# CardiaQ model evaluation, 2026-11-20

Models evaluated: `rf_model.joblib` and `quantum_model.joblib` at commit 16cff7b (live on cardiaq.streamlit.app).
Data: 1,221 patients (UCI Heart Disease 920 + Z-Alizadeh Sani 303, 2 duplicates removed), 11 features (ca and thal dropped).
Split: the same as train.py (test_size 0.2, random_state 42, stratified by source + target). Training: 976 rows (RF) and 240 rows (quantum). Test: 245 rows (184 UCI, 61 Z-Alizadeh; 145 with disease).
Baseline: backups/2026-11-20-1353/metrics.json.
Nothing was retrained. The saved models were loaded as they are, and the test AUCs reproduce metrics.json exactly (0.9061 / 0.8163).
Scripts are in the session scratchpad, not in the project.

## Verdict: WARN

All standard checks pass. The warnings come from the extra analyses:

1. **The quantum model ranks no better than chance for Z-Alizadeh men** (AUC 0.51, n = 33, only 9 without disease). It is also weak for Z-Alizadeh patients under 55 (AUC 0.59, n = 23). Both samples are tiny, but the overall Z-Alizadeh AUC is also low (0.72, 95% CI 0.57-0.85).
2. **The quantum model's probabilities are too low for Z-Alizadeh patients.** Its mean prediction is 0.58 against an observed rate of 0.70, ECE is 0.13, and its Brier score of 0.202 barely beats predicting the prevalence for everyone (0.220). The app says estimates are "the share of similar patients ... who had heart disease", and that is not true for this model on this source.
3. **The "typical angina scores lower than no chest pain" quirk is still present** in the classical model for any patient who has exercise-test results, which includes every patient entered in the app.
4. **explain.py contradicts the model in the contribution table.** For the app's default patient with typical angina, the table shows "Typical angina ... lowers risk" next to the sentence "Typical angina usually comes from reduced blood flow in the coronary arteries."

None of these are regressions. The models are practically identical to the baseline.

## Standard checks

| Check | Result |
|---|---|
| check_model.py --baseline | **PASS** (exit 0). Both models predict both classes, every source AUC is at least 0.65, all 5 medical directions are correct (exercise angina 58->84%, ST depression 3.0 58->75%, max HR 110 58->79%, age 70 58->66%, female 58->34%), quantum typical patient 58%, no source dropped against the baseline (largest change -0.003) |
| stress_test_app.py --runs 15 | **PASS**, 15/15, exit 0, no segfault (Qiskit 2.4.2). Only a deprecation notice for `st.components.v1.html` |

## Metrics (test set, threshold 0.5)

| Model | Data | n (pos) | Recall | Specificity | Precision | Accuracy | F1 | ROC AUC [95% bootstrap CI] |
|---|---|---|---|---|---|---|---|---|
| Random Forest | overall | 245 (145) | 0.848 | 0.740 | 0.826 | 0.804 | 0.837 | 0.906 [0.866, 0.940] |
| Random Forest | UCI | 184 (102) | 0.833 | 0.756 | 0.810 | 0.799 | 0.821 | 0.913 [0.872, 0.948] |
| Random Forest | Z-Alizadeh | 61 (43) | 0.884 | 0.667 | 0.864 | 0.820 | 0.874 | 0.872 [0.758, 0.961] |
| Quantum SVM | overall | 245 (145) | 0.786 | 0.700 | 0.792 | 0.751 | 0.789 | 0.816 [0.760, 0.865] |
| Quantum SVM | UCI | 184 (102) | 0.833 | 0.707 | 0.780 | 0.777 | 0.806 | 0.846 [0.788, 0.897] |
| Quantum SVM | Z-Alizadeh | 61 (43) | 0.674 | 0.667 | 0.829 | 0.672 | 0.744 | 0.721 [0.573, 0.849] |

Baseline: RF accuracy 0.808, recall 0.848, precision 0.831, AUC 0.907. Quantum accuracy 0.747, recall 0.786, precision 0.786, AUC 0.816. All changes are 0.005 or less.

No source is below AUC 0.65 overall. The quantum model on Z-Alizadeh (0.72) is the weakest result: its lower CI bound of 0.57 includes near-guessing, and its recall there is only 0.67. The gap between sources is 0.04 AUC for RF and 0.13 AUC for the quantum model. The quantum model fills Z-Alizadeh's missing exercise test with medians before PCA, which the app already discloses. Quantum settings: angle scale 0.05π, C = 100. C = 100 is the edge of C_GRID, so the search may not have reached its best value.

## 1. Calibration (5 equal-width bins; "pred" is the mean predicted probability, "obs" is the observed disease rate)

Brier score (lower is better; the reference is predicting the training prevalence for everyone):

| Model | Overall | UCI | Z-Alizadeh |
|---|---|---|---|
| Random Forest | 0.122 (ref 0.242), ECE 0.043 | 0.120 (ref 0.249), ECE 0.039 | 0.125 (ref 0.220), ECE 0.056 |
| Quantum SVM | 0.172 (ref 0.242), ECE 0.035 | 0.162 (ref 0.249), ECE 0.049 | **0.202 (ref 0.220), ECE 0.128** |

Reliability, overall (n per bin / pred / obs):

| Bin | RF | Quantum |
|---|---|---|
| 0.0-0.2 | 51 / 0.09 / 0.04 | 13 / 0.16 / 0.00 |
| 0.2-0.4 | 29 / 0.30 / 0.31 | 58 / 0.29 / 0.34 |
| 0.4-0.6 | 40 / 0.52 / 0.53 | 55 / 0.50 / 0.51 |
| 0.6-0.8 | 38 / 0.72 / 0.84 | 42 / 0.71 / 0.67 |
| 0.8-1.0 | 87 / 0.90 / 0.93 | 77 / 0.91 / 0.90 |

Per source:

| Bin | RF UCI | RF Z-A | Quantum UCI | Quantum Z-A |
|---|---|---|---|---|
| 0.0-0.2 | 46 / 0.09 / 0.04 | 5 / 0.14 / 0.00 | 13 / 0.16 / 0.00 | 0 |
| 0.2-0.4 | 20 / 0.29 / 0.30 | 9 / 0.31 / 0.33 | 41 / 0.28 / 0.27 | **17 / 0.31 / 0.53** |
| 0.4-0.6 | 31 / 0.51 / 0.52 | 9 / 0.52 / 0.56 | 39 / 0.50 / 0.46 | 16 / 0.50 / 0.62 |
| 0.6-0.8 | 27 / 0.71 / 0.81 | 11 / 0.74 / 0.91 | 27 / 0.72 / 0.59 | 15 / 0.70 / 0.80 |
| 0.8-1.0 | 60 / 0.90 / 0.93 | 27 / 0.91 / 0.93 | 64 / 0.91 / 0.89 | 13 / 0.88 / 0.92 |

Reading:
- **Random Forest:** reasonably calibrated on both sources. It is slightly under-confident in the 0.6-0.8 band (predicts 0.72, observed 0.84) and slightly over-predicts at the low end.
- **Quantum SVM:** fine overall and on UCI, where it is mildly over-confident in 0.6-0.8. On Z-Alizadeh it under-predicts across the whole range: patients given about 31% had disease 53% of the time (n = 17). Its single calibration step was fitted on mostly-UCI training rows and does not transfer to this source.
- Every bin has 5-87 patients, so the per-bin rates are rough. The Z-Alizadeh bins in particular have 5-27 patients each.

## 2. Subgroups (test set)

| Subgroup | n (pos/neg) | RF AUC [CI] | RF recall | Quantum AUC [CI] | Quantum recall |
|---|---|---|---|---|---|
| Female | 63 (27/36) | 0.942 [0.88, 0.99] | 0.852 | 0.881 [0.78, 0.95] | 0.741 |
| Male | 182 (118/64) | 0.884 [0.83, 0.93] | 0.847 | 0.782 [0.71, 0.85] | 0.797 |
| Age < 55 | 119 (55/64) | 0.945 [0.90, 0.98] | 0.782 | 0.833 [0.76, 0.90] | **0.636** |
| Age >= 55 | 126 (90/36) | 0.848 [0.78, 0.91] | 0.889 (spec 0.47) | 0.733 [0.64, 0.82] | 0.878 (spec 0.42) |
| Female, UCI | 35 (8/27) | 0.907 | 0.750 | 0.824 | **0.500** |
| Female, Z-A | 28 (19/9) | 0.947 | 0.895 | 0.889 | 0.842 |
| Male, UCI | 149 (94/55) | 0.899 | 0.840 | 0.827 | 0.862 |
| **Male, Z-A** | 33 (24/9) | 0.792 | 0.875 | **0.514** | **0.542** |
| Age < 55, UCI | 96 (42/54) | 0.951 | 0.786 | 0.880 | 0.714 |
| **Age < 55, Z-A** | 23 (13/10) | 0.915 | 0.769 | **0.585** | **0.385** |
| Age >= 55, UCI | 88 (60/28) | 0.840 | 0.867 | 0.762 | 0.917 |
| Age >= 55, Z-A | 38 (30/8) | 0.846 | 0.933 | 0.738 | 0.800 |

Flags:
- **Quantum SVM, Z-Alizadeh men:** AUC 0.51 (chance level) and recall 0.54. This rests on 33 patients, of whom only 9 had no disease.
- **Quantum SVM, Z-Alizadeh under 55:** AUC 0.59 and recall 0.38, from 23 patients.
- **Quantum SVM, UCI women:** recall 0.50, but there are only 8 positive patients.
- **Quantum SVM, under 55 overall:** recall 0.64, so it misses about 1 in 3 younger patients with disease (n = 119).
- **Both models, age >= 55:** AUC is lower and specificity is under 0.5. Most older patients without disease are flagged, so the model leans on age.
- **RF:** no subgroup is below 0.79 AUC. Its weakest is Z-Alizadeh men (0.79, n = 33).
- Women score higher than men for both models. Part of this is the case mix: women in UCI are mostly without disease and easy to separate.
- All subgroup cells under about 40 patients have wide uncertainty (a bootstrap CI half-width of roughly ±0.1 or more). Treat them as signals, not measurements.

## 3. Threshold (descriptive only, test set n = 245; not a recommendation to change the app)

| Model | Data | At 0.5: sens / spec | Threshold for about 90% sensitivity | sens / spec / PPV there | Share flagged |
|---|---|---|---|---|---|
| RF | overall | 0.848 / 0.740 | 0.41 | 0.910 / 0.690 / 0.810 | 67% |
| RF | UCI | 0.833 / 0.756 | 0.41 | 0.912 / 0.707 | 64% |
| RF | Z-A | 0.884 / 0.667 | 0.40 | 0.930 / 0.611 | 77% |
| Quantum | overall | 0.786 / 0.700 | 0.36 | 0.903 / 0.470 / 0.712 | 75% |
| Quantum | UCI | 0.833 / 0.707 | 0.39 | 0.912 / 0.524 | 72% |
| Quantum | Z-A | 0.674 / 0.667 | 0.32 | 0.930 / 0.333 | 85% |

The overall 0.36 quantum threshold gives only 0.84 sensitivity on Z-Alizadeh, because its scores sit lower on that source. For screening at about 90% sensitivity, RF costs about 5 points of specificity, while the quantum model costs about 23. These thresholds were chosen and evaluated on the same 245 patients, so they are optimistic. They describe this test set and should not be read as a validated operating point. Disease prevalence in this data is 59%, far above a screening population, so the PPVs above do not transfer.

## 4. Chest pain quirk ("typical angina" lower than "no chest pain"): still present

Classical estimate for the app's default patient (55, male, BP 130, chol 240, max HR 150, no exercise angina, ST 1.0, flat slope):

| Chest pain | Typical angina | Atypical | Non-anginal | None (asymptomatic) |
|---|---|---|---|---|
| Default patient | 0.32 | 0.27 | 0.33 | **0.58** |
| Same but female | 0.15 | 0.10 | 0.14 | 0.34 |
| 65, exercise angina, ST 2.0 | 0.63 | 0.67 | 0.68 | 0.86 |

When cp is swapped for every test patient, typical angina gives a lower estimate than no chest pain for 70% of them (median -0.10). The exception is Z-Alizadeh rows with no exercise test, where typical angina is now the highest-risk category.

Observed disease rate by chest pain type:

| | Typical | Atypical | Non-anginal | None |
|---|---|---|---|---|
| UCI (n) | 0.43 (46) | 0.14 (173) | 0.35 (203) | 0.79 (496) |
| Z-Alizadeh (n) | 0.94 (164) | 0.43 (93) | 0.19 (16) | 0.63 (30) |

**Why it happens.** Everyone in UCI was referred for coronary angiography. Patients who were referred while having no chest pain were mostly sent because of other strong evidence, such as an abnormal ECG or stress test or known disease. As a result, "asymptomatic" is the highest-risk group in UCI (79%), and the 46 typical-angina patients have a lower rate (43%). This is a selection effect of the referral population, not medicine. Z-Alizadeh shows the clinically expected pattern (typical angina 94%). The forest has no source variable, so it can only tell the two populations apart by whether the exercise test is missing. The app always supplies exercise values, so app users always get the UCI pattern.

## 5. explain.py against the current data and model

SHAP was checked on all 245 test patients and on the default patient.

1. **Typical angina wording contradicts the model (main finding).** VALUE_WHY for cp = 0 says typical angina "usually comes from reduced blood flow in the coronary arteries". For the default patient with typical angina, SHAP gives cp -0.030 ("lowers risk"), and the full contribution table in the app shows that text beside "lowers risk". This happens for 8 of 46 typical-angina test patients. For the other 38 the sign is positive (mean +0.078), driven by Z-Alizadeh. For any app input, the "why" text and the effect disagree whenever the UCI pattern applies.
2. **Cholesterol is weaker than the text implies.** The text "Cholesterol builds plaque" is medically right, but in Z-Alizadeh the disease rate is flat across cholesterol quartiles (0.74 / 0.66 / 0.76 / 0.69). The model's SHAP is only weakly tied to the value (Spearman 0.27). In 3 test patients, a low cholesterol (for example 127 mg/dl) appears as a top-3 risk raiser with that sentence. In UCI, a missing cholesterol value (the Swiss zeros) marks an 81%-disease group, so "missing" carries signal. The app never sends a missing value, so this does not reach users.
3. **Missing values show as raising risk.** Missing exang, fbs or thalach push risk up because they mark the high-prevalence Z-Alizadeh rows. Displayed, this would read "nan raised predicted risk". It does not reach app users, because the form always fills every field, but explain_patient would show it for raw data rows.
4. **Consistent with the model:**
   - Non-anginal pain "usually not caused by the heart" (SHAP always negative).
   - Asymptomatic "silent disease" (SHAP positive 98% of the time).
   - Exercise angina, ST depression (Spearman +0.90), low max heart rate (-0.94), age (+0.89), blood pressure (+0.55).
   - Flat or downsloping slope (always positive).
   - Abnormal resting ECG (positive).
   - Fasting blood sugar > 120 (positive).
   - Sex text (neutral wording; male positive).
5. **Unused labels.** The thal and ca text and VALUE_NAMES still exist, but both features are dropped, so they are harmless.

## Benchmark (results/, run yesterday on the same data; not re-run)

10-fold CV, mean ± SD across folds:

| Model | ROC AUC | Accuracy | Recall |
|---|---|---|---|
| RF, all features | 0.863 ± 0.031 | 0.795 ± 0.028 | 0.836 ± 0.044 |
| RBF-SVM, default | 0.810 ± 0.033 | 0.733 ± 0.029 | 0.720 ± 0.080 |
| RBF-SVM, tuned | 0.803 ± 0.033 | 0.688 ± 0.103 | 0.635 ± 0.246 |
| RF, 4 PCA features | 0.787 ± 0.038 | 0.730 ± 0.033 | 0.771 ± 0.059 |
| Quantum SVM, tuned | 0.787 ± 0.042 | 0.699 ± 0.105 | 0.656 ± 0.243 |
| Quantum SVM, default | 0.544 ± 0.052 | 0.541 ± 0.046 | 0.599 ± 0.044 |

**Fair comparison (tuned quantum against tuned RBF-SVM).** The quantum model's AUC is 0.016 lower on average. Per fold, the paired difference has an SD of 0.021, and the quantum model wins only 2 of 10 folds. The gap is within the spread across folds (each model's SD is 0.033-0.042), so the two are statistically indistinguishable, with a consistent slight edge for the classical kernel. RF on all features beats the tuned quantum model in every fold, by 0.046 to about 0.13 (mean 0.076).

**Benchmark caveat.** In fold 5, both tuned SVMs predicted "no disease" for everyone (recall 0, accuracy 0.41), even though their AUCs were 0.69-0.76. This single fold causes the very large recall and accuracy SDs. The same thing happens in quantum_experiments.csv at C = 0.1 (AUC 0.845, recall 0). It is a thresholding failure of uncalibrated SVM decision values. train.py avoids it with its calibration step, but the benchmark's tuned SVM rows do not.

**quantum_experiments.csv** (single split, 120 training rows):
- **Angle scale:** small scales work best (0.05π: AUC 0.849; 0.1π: 0.809). At 0.5π or more, AUC is 0.48-0.54, which is chance (kernel concentration). This supports the chosen 0.05π.
- **Feature map:** ZZ reps 1, 2 and 3 and Pauli(Y,ZZ) are all 0.84-0.85, so the feature map makes no real difference.
- **Qubits:** 2 gives 0.83 and 4 gives 0.85, but 6 drops to 0.74 and 8 to 0.69, while time grows from 20 s to 302 s. 4 is right.
- **Training size:** 40 rows give 0.71, 80 give 0.82, 120 give 0.85, and 200 give 0.83, so it plateaus at about 120.
- **C:** 1 is best (0.849); 10 gives 0.821. The live model uses C = 100, the edge of train.py's grid.

## Recommendation: keep live, with fixes

Keep live. The models pass every standard check, the app is stable, and they do not differ from the previous live models, so there is nothing to roll back to. The issues are pre-existing and apply to an educational tool that already carries a "not clinically validated" notice. Suggested fixes, in priority order:

1. **explain.py, cp = 0 wording:** make the typical-angina text honest about the data. For example: "Clinically, typical angina points to reduced coronary blood flow; in the UCI hospital data, however, patients referred without chest pain had more disease, so the model may score it lower." Alternatively, suppress the "why" sentence whenever its direction contradicts the SHAP sign. This is the only place where the app currently states something the model contradicts.
2. **Disclose or fix the quantum model on Z-Alizadeh-type patients.** Options are calibrating per source or with a missing-exercise-test indicator, or saying in the app that the quantum estimate is less reliable for patients without exercise-test data. At minimum, soften the claim that "estimates are the share of similar patients who had heart disease" for the quantum model.
3. **Chest pain quirk:** add a short note in the app (Data and limits) explaining the UCI referral effect. A modelling fix, such as a source or exercise-test-missing feature or monotonic constraints, needs a human decision, because it changes what the model learns.
4. **train.py:** extend C_GRID beyond 100 (or check why 100 wins), since the chosen C sits on the grid edge.
5. **benchmark.py:** calibrate or threshold the tuned SVMs before reporting recall and accuracy, so that fold-5-style all-negative predictions stop inflating the SDs. AUC conclusions are unaffected.

Deployment: no push or reboot is needed for this evaluation, because the live site already serves these models. If fixes 1-3 are made later, pushing to main redeploys, but the running app keeps the old models in memory until it is rebooted from share.streamlit.io (app menu -> Reboot). Afterwards, verify the "Trained on 1221 patients" line. Pushing and rebooting both need the user's approval.
