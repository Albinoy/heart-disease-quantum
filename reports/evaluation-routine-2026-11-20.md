# CardiaQ evaluation: routine-checks model ("second opinion"), 2026-11-20

Commit evaluated: 605495c (live on https://cardiaq.streamlit.app). Saved `routine_model.joblib` loaded as-is,
no retraining. Scripts and logs: `/private/tmp/claude-501/-Users-albinbinoy0325gmail-com--claude-spares-spare-FD6Co6/12e2507a-db80-43f3-b816-11ec6fe8c0db/scratchpad/eval_routine/`.

## Verdict: FAIL (second-opinion feature). Main app: PASS.

The routine model is not a pass. Reasons:

1. **Blood pressure points the wrong way for many patients.** In the Soochow data, heart-attack patients
   have *lower* systolic BP than the comparison inpatients: disease rate is 41% at BP <=100 and 2% at BP >180.
   That comes from how the label was defined; it is not medicine. The model learned it. A 75-year-old man (chol 240,
   fbs 0) scores 82% at BP 90 and 60% at BP 200. Across a realistic grid (age 35-80, BP 110-180, chol 160-320),
   raising BP from 110 to 180 *lowers* the score in 37% of cases, and by more than 5 points in 27%. The README says
   "every risk factor points the right way". That is false. The training script's direction check tests only
   one patient (55-year-old man, BP 130 -> 160), and that patient happens to go up.
2. **The other three inputs add little.** An age+sex-only model trained the same way is statistically
   indistinguishable in 4 of 5 sources (UCI, Tehran, Duke, St. Gallen). The only clear gain is in Soochow (+0.065),
   which is the source behind the BP artefact. On the external SAheart test the model scores 0.685, and raw age
   alone also scores 0.686. Age and sex explain 57% of the variation in the score.
3. **Sex dominates.** In the test set, no Duke woman was scored >= 50% (recall 0.00, n=92 diseased women).
   St. Gallen: women recall 0.20, men specificity 0.04.
4. **Leave-one-source-out AUC is near guessing for Soochow (0.59) and Tehran (0.64), both below 0.65.**
5. **Some app wording overstates what the model does** (details in item 4 below).

The pipeline itself is correct. The app passes the form values to the model with the right units, coding and
feature order (item 8). The standard checks pass.

Recommendation: **remove the second opinion from the app until it is fixed, or hide it behind a clearly
labelled "experimental" toggle.** Keeping it means users see an independent-looking "second opinion" in which very
high blood pressure can lower the score. Removing it, or keeping it with wording fixes only, is a product decision
for a human. On the evidence, a wording-only fix is not enough while the BP reversal remains.

## Standard checks

| Check | Result |
|---|---|
| `check_model.py` | PASS (all 10 checks) |
| `stress_test_app.py --runs 15` | PASS, 15/15, no segfault (Qiskit 2.4.2, Streamlit 1.67.0). The stress test submits the default patient, so the routine section runs too. |
| Note | Each render logs "`st.components.v1.html` will be removed after 2026-06-01". That date has passed. The pinned 1.67.0 still works, but any Streamlit upgrade may break the charts. |

Main models (metrics.json, unchanged by this commit):

| Model | Recall | Precision | Accuracy | F1 | ROC AUC | AUC UCI (n=184) | AUC Tehran (n=61) |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.85 | 0.83 | 0.80 | 0.84 | 0.906 | 0.913 | 0.872 |
| Quantum kernel SVM | 0.79 | 0.79 | 0.75 | 0.79 | 0.816 | 0.846 | 0.721 |

## 1. Reproduced test metrics (saved model, same split), 95% stratified bootstrap CIs (2,000 resamples)

All saved AUCs reproduce exactly. Threshold 0.5.

| Source | n test | diseased | ROC AUC [95% CI] | Recall [CI] | Precision [CI] | Specificity | Accuracy | F1 |
|---|---|---|---|---|---|---|---|---|
| UCI | 184 | 102 | 0.678 [0.600-0.759] | 0.76 [0.68-0.84] | 0.64 [0.59-0.70] | 0.46 | 0.63 | 0.70 |
| Tehran (Z-Alizadeh) | 60 | 43 | 0.740 [0.581-0.878] | 0.63 [0.49-0.77] | 0.87 [0.77-0.97] | 0.76 | 0.67 | 0.73 |
| Duke | 701 | 467 | 0.745 [0.705-0.782] | 0.68 [0.63-0.72] | 0.81 [0.78-0.84] | 0.69 | 0.68 | 0.74 |
| Soochow | 1,530 | 305 | 0.734 [0.703-0.766] | 0.75 [0.70-0.80] | 0.31 [0.29-0.33] | 0.59 | 0.62 | 0.44 |
| St. Gallen | 163 | 117 | 0.729 [0.639-0.815] | 0.85 [0.79-0.91] | 0.82 [0.78-0.87] | 0.52 | 0.76 | 0.84 |
| Pooled | 2,638 | 1,034 | 0.719 [0.698-0.738] | 0.73 [0.70-0.75] | 0.54 [0.52-0.55] | 0.60 | 0.65 | 0.62 |

- Mean of per-source AUCs: 0.725, 95% CI [0.686-0.764].
- Leave-one-source-out (from routine_metrics.json): Duke 0.747, St. Gallen 0.710, UCI 0.656, **Tehran 0.639,
  Soochow 0.590**. The last two are below 0.65, close to guessing.
- External SAheart test (422 adult men, 159 CHD; age and BP only): AUC 0.685 [0.632-0.733]. Raw age alone
  scores 0.686, so this external result shows no value beyond age.
- The lower bound of the UCI CI is 0.60, and Tehran's CI is very wide (n=60).
- The model predicts both classes in every source (share predicted positive: 0.48-0.75).

## 2. Medicine or source?

The weighting does what it was meant to do across sources:

- A model given only the source label (same weights) has no information: every score is 0.50, and the
  within-source AUC is 0.5 everywhere. Unweighted, the same source-only model would reach a pooled AUC of 0.743.
  That is higher than the real model's 0.719, so the weights are essential.
- Which inputs are missing explains only 2.8% of the score's variation; the source explains 2.7%. Sex plus age
  decade explain 57%, and the true outcome explains 13%.
- A model given only the missing-value pattern is useless within Duke, St. Gallen, Soochow and Tehran
  (AUC 0.49-0.51).

But:

- **Leak inside UCI:** cholesterol is missing for every Swiss UCI patient (94% diseased). In the UCI test set,
  83% of patients with missing cholesterol are diseased, against 47% of the rest, and their mean score is 0.66
  against 0.52. A missing-pattern-only model scores AUC 0.645 on UCI. On UCI patients with all inputs present,
  the model's AUC is 0.685, so the headline UCI number is not materially inflated. Still, the model rewards
  "cholesterol missing".
- **Missing-value branches change individual scores a lot.** Take the same full-input patients and set BP and
  sugar to missing: the median score moves by 0.10-0.11, and for UCI patients the mean rises by +0.11.
  Fill in Duke patients' missing BP and sugar with typical values instead, and their mean score drops by 0.10;
  with imputation from similar patients, it drops by 0.05. So the model's behaviour for complete inputs and for
  gappy inputs differs, and the Duke and St. Gallen test AUCs say little about complete inputs.

**What this means for app users (who always give all 5 inputs).** For a full-input patient, the training
patients sharing the same tree leaves (weighted) come from Tehran 37.5%, Soochow 35.8%, UCI 23.4%,
Duke 1.8% and St. Gallen 1.4%. So the score an app user sees is in effect a model of three groups,
about 8,900 patients. Under the equal-source weighting, the 303 Tehran patients carry the most influence.
The Soochow heart-attack label, which reverses BP, carries about a third. The relevant validation is the
full-input test AUC: UCI 0.685 (n=132), Tehran 0.739 (n=59), Soochow 0.733 (n=1,507).

## 3. Added value over age + sex (same RF settings, same weights, same split; trained in scratch)

| Source | Full model AUC | Age+sex AUC [95% CI] | Difference [paired 95% CI] |
|---|---|---|---|
| UCI | 0.678 | 0.701 [0.623-0.779] | -0.022 [-0.084, +0.040] |
| Tehran | 0.740 | 0.643 [0.477-0.798] | +0.096 [-0.027, +0.213] |
| Duke | 0.745 | 0.726 [0.688-0.764] | +0.019 [-0.005, +0.041] |
| Soochow | 0.734 | 0.669 [0.639-0.700] | **+0.065 [+0.036, +0.094]** |
| St. Gallen | 0.729 | 0.724 [0.629-0.811] | +0.006 [-0.047, +0.064] |
| SAheart (external) | 0.685 | 0.666 [0.611-0.716] | +0.019 [-0.003, +0.042] |

Only Soochow shows a significant gain. (St. Gallen has only age and sex, so it cannot benefit.) Adding only BP
and sugar reaches 0.728 on Soochow, so most of that gain comes from BP and sugar, including the BP reversal.
The extra inputs give no measurable benefit on UCI.

## 4. Calibration and the app wording

- "50% = average patient across the five groups": numerically true under the training weights. The weighted
  mean score is 0.500 on train and 0.508 on test. Within single groups it does not hold: the class-balanced mean
  score per source runs from 0.43 (Tehran) to 0.56 (St. Gallen). The balanced calibration (weighted disease share
  by score bin) is reasonable: score 0.18 -> 0.23, 0.45 -> 0.48, 0.65 -> 0.61, 0.78 -> 0.74.
- Typical scores (test-set medians, healthy / diseased): UCI 0.51 / 0.64; Tehran 0.29 / 0.60; Duke 0.34 / 0.62;
  Soochow 0.44 / 0.65; St. Gallen 0.45 / 0.67. **54% of healthy UCI patients and 48% of healthy St. Gallen
  patients score >= 50%.** The groups overlap heavily: the IQRs of healthy and diseased patients overlap in
  every source.
- Wording that is inaccurate or likely to mislead:
  1. "trained on 13,190 real patients from five hospital groups". For anyone using the app (all inputs
     entered), Duke and St. Gallen contribute about 3% of the prediction. The large number suggests robustness
     the model does not have for app users.
  2. "Second opinion" suggests independent, comparable evidence. The model is weaker (AUC about 0.74 on unseen
     patients against 0.84-0.91), behaves mostly like an age+sex rule, and lands on the opposite side of 50% from
     the main model for about a quarter of patients (item 6).
  3. The text says the groups "define disease differently (angiography in four, heart attack in Soochow)", but
     does not say the result: low BP raises the score and very high BP can lower it. A user who enters
     BP 200 can be reassured.
  4. README: "every risk factor points the right way". This is false for BP, and partly for cholesterol and for
     age above 75 (item 7).
  5. The AUC range shown (0.68-0.74 "per hospital group") includes Duke and St. Gallen, whose test patients look
     nothing like app inputs. The comparison "against 0.91 for the classical model" mixes different patient
     sets. On the same unseen patients the comparison is 0.74 against 0.84 (item 6).
  6. Minor: the app section header still says the classical/quantum "Result" text, and the second opinion
     has no level (Low/Moderate/High) of its own. That is fine, but 37% of test scores fall in 0.30-0.60, where
     the score says little.

## 5. Subgroups (test set; AUC only where both classes have at least 10 patients; CI from 1,000 bootstraps)

| Source | Group | n | diseased | AUC [95% CI] | Recall | Specificity |
|---|---|---|---|---|---|---|
| UCI | female | 34 | 11 | 0.58 [0.38-0.77] | 0.09 | 0.78 |
| UCI | male | 150 | 91 | 0.66 [0.57-0.74] | 0.85 | 0.34 |
| UCI | <55 | 89 | 39 | 0.66 [0.54-0.76] | 0.56 | 0.60 |
| UCI | >=55 | 95 | 63 | 0.62 [0.49-0.74] | 0.89 | 0.25 |
| Tehran | female | 28 | 18 | 0.82 [0.63-0.97] | 0.44 | 0.90 |
| Tehran | male | 32 | 25 | n/a (7 healthy) | 0.76 | 0.57 |
| Tehran | <55 | 22 | 12 | 0.76 [0.54-0.94] | 0.50 | 0.90 |
| Tehran | >=55 | 38 | 31 | n/a (7 healthy) | 0.68 | 0.57 |
| Duke | female | 215 | 92 | 0.66 [0.58-0.73] | **0.00** | 0.99 |
| Duke | male | 486 | 375 | 0.67 [0.61-0.73] | 0.85 | 0.35 |
| Duke | <55 | 389 | 227 | 0.69 [0.64-0.74] | 0.57 | 0.67 |
| Duke | >=55 | 312 | 240 | 0.78 [0.71-0.84] | 0.78 | 0.72 |
| Soochow | female | 646 | 55 | 0.73 [0.65-0.80] | 0.58 | 0.80 |
| Soochow | male | 884 | 250 | 0.66 [0.62-0.70] | 0.78 | 0.39 |
| Soochow | <55 | 470 | 73 | 0.76 [0.69-0.81] | 0.47 | 0.84 |
| Soochow | >=55 | 1,060 | 232 | 0.73 [0.69-0.76] | 0.84 | 0.47 |
| St. Gallen | female | 43 | 20 | 0.60 [0.41-0.76] | 0.20 | 1.00 |
| St. Gallen | male | 120 | 97 | 0.63 [0.50-0.75] | 0.99 | **0.04** |
| St. Gallen | <55 | 15 | 8 | n/a | 0.62 | 0.29 |
| St. Gallen | >=55 | 148 | 109 | 0.76 [0.66-0.84] | 0.87 | 0.56 |
| Pooled | female | 966 | 196 | 0.61 [0.57-0.65] | 0.23 | 0.84 |
| Pooled | male | 1,672 | 838 | 0.68 [0.65-0.71] | 0.84 | 0.37 |
| Pooled | <55 | 985 | 359 | 0.75 [0.72-0.78] | 0.55 | 0.77 |
| Pooled | >=55 | 1,653 | 675 | 0.72 [0.69-0.74] | 0.82 | 0.49 |

Within each sex, ranking is weak (pooled AUC 0.61 for women, 0.68 for men). At the 50% line the model mostly
sorts by sex. Women with disease are usually scored below 50% (pooled recall 0.23); men without disease
are usually scored above it (specificity 0.37).

## 6. Agreement with the main classical model (train.py split: 245 test patients, UCI 184, Tehran 61)

**Caveat:** 198 of these 245 patients (UCI 80%, Tehran 84%) are in the routine model's *training* set,
because the two models use different splits. Any routine-model metric on the main test set is optimistic. Only
the 47 unseen patients give a fair comparison.

| Patients | n | Pearson | Spearman | Opposite sides of 50% | Main model right when they disagree | AUC main | AUC routine |
|---|---|---|---|---|---|---|---|
| All | 245 | 0.68 | 0.66 | 23% | 51% | 0.906 | 0.884 (leaked) |
| Not in routine training | 47 | 0.60 | 0.60 | 26% | 67% | 0.842 | 0.742 |
| UCI, unseen | 37 | 0.56 | 0.56 | 30% | 64% | 0.862 | 0.729 |
| Tehran, unseen | 10 | 0.79 | 0.82 | 10% | 100% | 0.708 | 0.792 |

- Strong contradictions are rare. Main >= 60% with routine < 40%: 4%. Main < 30% with routine >= 60%: 2%.
- On the 47 unseen patients, the blend 0.8*main + 0.2*routine scores 0.855 against 0.842. That difference is
  noise at this n.
- Verdict: the two models disagree on about 1 patient in 4, and when they do, the weaker model is usually the
  one that is wrong. As presented, a "second opinion" that contradicts the main result 25% of the time, with no
  guidance on which to trust, is more likely to confuse than to help. Its legitimate use is as a "what routine
  checks alone can tell" illustration.

## 7. Directions over a grid (age 25-90, sex, BP 90-200, chol 120-500, fbs: 13,440 patients)

| Input | Typical effect (median, low -> high end) | Reversed (high end lower than low end) | Single-step falls > 1 point |
|---|---|---|---|
| Age 25 -> 90 | +0.24 | 0.8% | 25.6% of steps; mean score peaks at 70 (0.67) and falls to 0.62 at 90 |
| BP 90 -> 200 | **-0.06** | **66.7%** | 33.7%; the mean score is highest at BP 90 (0.61) |
| BP 110 -> 180 (realistic region) | +0.04 | 37.2% (26.7% by > 5 points) | |
| Chol 120 -> 500 | +0.11 | 13.5% | 19.2% |
| Chol 160 -> 320 (realistic region) | +0.07 | 26.2% (18.4% by > 5 points) | |
| fbs 0 -> 1 | +0.21 | 3.7% lower by > 1 point (worst -0.27) | |
| female -> male | +0.10 | 11.8% lower by > 1 point (worst -0.19) | |

Examples (chol 240, fbs 0 unless stated):

- BP 90 / 120 / 160 / 200:
  - man 75: 0.82 / 0.73 / 0.69 / 0.60
  - man 65: 0.80 / 0.62 / 0.67 / 0.61
  - woman 55: 0.18 / 0.17 / 0.28 / 0.31
- Cholesterol, man 65, BP 130: 180 -> 0.78, 260 -> 0.74, 320 -> 0.46.
- Age, woman, BP 130: 55 -> 0.19, 60 -> 0.57 (one five-year step adds 38 points), 65 -> 0.48.
- A 38-year-old man with BP 95 and chol 150 scores 59% in the live app.

Root cause for BP: raw per-source logistic coefficients for BP are UCI +0.22, Tehran +0.41, Soochow **-0.51**.
In Soochow the disease rate falls from 41% (BP <= 100) to 2% (BP > 180). Acute heart-attack patients present with
low pressure, and the hypertensive inpatients are the controls. For age above 75 and for cholesterol, the
non-monotonic regions come from sparse data and forest step noise (min_samples_leaf=10 on up to 303-patient
sources carrying 20% of the weight each).

## 8. App wiring

Correct. I drove the real app with AppTest using four non-default patients (for example, woman 72, BP 175, chol 310,
fbs yes). The displayed score matched a direct `predict_proba` every time (82/81.6, 59/58.5, 77/76.9, 53/52.6).

- Feature order comes from `rm["features"]`, and the model also checks feature names.
- Sex: 1 = male, the same as all sources.
- fbs "Yes" = 1, meaning > 120 mg/dl. Soochow is converted at 6.67 mmol/L, which is 120 mg/dl.
- BP is systolic mm Hg.
- Cholesterol is mg/dl. Soochow mmol/L x 38.67 is correct.
- Form ranges are age 20-90, BP 80-200, chol 100-600, all inside the training ranges.

Data note, not a wiring issue: Tehran cholesterol is *derived* (LDL + HDL + TG/5), and Tehran (median 170) and
Soochow (median 164) cholesterol are far below UCI (240) and Duke (225), probably because of treatment or
admission effects. The model's cholesterol scale therefore mixes populations.

## Recommendation

**Remove the second opinion from the live app until it is fixed** (or keep it only behind an "experimental"
toggle). Needs the user's decision. Fixes in priority order:

1. **Fix the BP reversal.** Options:
   - drop BP for the Soochow rows, or drop Soochow's BP and sugar altogether;
   - or use a model with monotonic constraints that supports missing values, such as
     HistGradientBoostingClassifier(monotonic_cst=...).

   Then re-run a grid direction check like item 7, not a single-patient check. Make the grid check part of
   `check_model.py` for this model.
2. **Show the score only if it beats age+sex.** Re-run item 3 after the fix. If no significant gain on
   full-input sources remains, say plainly that the score is mostly age and sex, or drop the model.
3. **Correct the wording:**
   - change "trained on 13,190 patients from five groups" to say that complete inputs are scored mostly from
     UCI, Tehran and Soochow;
   - remove "every risk factor points the right way" from the README;
   - rename "Second opinion" to something like "What routine checks alone suggest";
   - state the 25% disagreement rate, and that the main model should be preferred when the two disagree;
   - quote full-input AUCs (UCI 0.69, Tehran 0.74, Soochow 0.73) and LOSO 0.59-0.75.
4. **Fix the evaluation leak for item-6-style comparisons:** align the routine split with train.py's split for the
   UCI/Tehran patients, so main-model test patients are never in routine training.
5. **Within-UCI "cholesterol missing" leak** (Swiss hospital). Harmless for app users, but it inflates UCI
   numbers slightly. Consider dropping UCI-Switzerland rows or setting their cholesterol to missing for all UCI rows.
6. Housekeeping: migrate `st.components.v1.html` to `st.iframe` before any Streamlit upgrade.

Deployment: any change needs a push to main *and* a reboot from share.streamlit.io (app menu -> Reboot), because the
live app keeps old models in memory. Verify afterwards via the "Trained on N patients" line. Pushing and
rebooting need the user's go-ahead. Nothing was committed, pushed or changed on the live site during this evaluation.
