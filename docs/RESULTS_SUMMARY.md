# MedTrust-AI — Current Experimental Results

## Experiment

**Experiment ID:** `EXP-DN121-BASE-S42`

**Task:** Binary chest X-ray classification  
**Positive class:** Effusion present  
**Negative class:** No Finding  
**Architecture:** DenseNet-121  
**Best checkpoint:** Epoch 2  
**Seed:** 42

## Frozen Dataset

Total images: **19,326**

- Train: **12,384**
- Validation: **3,078**
- Test: **3,864**

Patient-level splitting was preserved with no cross-split patient overlap.

The test set has **not yet been used for evaluation**.

## Clean Validation Performance

- AUROC: **0.9279**
- AUPRC: **0.9269**
- Accuracy: **0.8564**
- Sensitivity: **0.8902**
- Specificity: **0.8226**
- F1: **0.8611**
- Failure rate: **0.1436**

## Calibration

Temperature scaling fitted on validation data produced:

- Temperature: **1.0016**
- ECE before: **0.01948**
- ECE after: **0.01901**

The near-unit temperature indicates little global miscalibration on clean validation data.

## Failure Detection

Deterministic confidence-based failure detection:

- AUROC: **0.7781**
- AUPRC: **0.3739**
- Random AUPRC baseline: **0.1436**

## Selective Prediction

Using uncertainty to abstain from uncertain predictions reduced risk among accepted predictions.

For MC predictive entropy:

- Full coverage risk: **14.36%**
- 90% coverage risk: **10.76%**
- 80% coverage risk: **8.77%**
- 70% coverage risk: **7.43%**
- 50% coverage risk: **5.33%**

## Distribution Shift Protocol

Five shift families were evaluated at mild, moderate, and severe levels:

1. Brightness reduction
2. Contrast reduction
3. Gaussian noise
4. Gaussian blur
5. Resolution degradation

Total evaluation conditions: **16** including clean.

No model retraining, finetuning, or shift-specific recalibration was performed.

## Severe Shift Findings

### Gaussian Blur — Severe

- Accuracy: **0.7086**
- Failure rate: **0.2914**
- Failure-detection AUROC: **0.7315**
- AURC: **0.1443**

### Gaussian Noise — Severe

- Accuracy: **0.7427**
- Failure rate: **0.2573**
- Classification AUROC: **0.8604**
- Failure-detection AUROC: **0.7244**
- AURC: **0.1327**

### Resolution Degradation — Severe

- Accuracy: **0.7580**
- Failure rate: **0.2420**
- Failure-detection AUROC: **0.7220**
- AURC: **0.1182**

## MC Dropout

MC Dropout was compared against deterministic confidence-based uncertainty.

Across 16 conditions:

- MC entropy failure-detection AUROC better: **5/16**
- MC entropy failure-detection AUROC worse: **11/16**
- MC entropy AUPRC better: **6/16**
- MC entropy AUPRC worse: **10/16**

Current evidence therefore does **not** show consistent superiority of MC Dropout over deterministic confidence.

## Current Interpretation

The current experiments suggest that:

1. Classification performance deteriorates as distribution shift becomes more severe.
2. Gaussian blur, Gaussian noise, and resolution degradation produce the largest reliability deterioration.
3. The model's ability to identify its own failures also weakens under strong distribution shift.
4. Selective prediction can substantially reduce risk by abstaining on uncertain predictions.
5. MC Dropout provides only small and inconsistent gains over deterministic confidence.

## Reproducibility

Training revision:

`0daf10aa0d6a19bd53ecc2eb8d24e6b1b3d19db2`

The repository contains frozen configurations, cohort manifests, training history, calibration artifacts, shift protocol, deterministic and MC Dropout analyses, and summary tables.

Large model checkpoint binaries and the NIH image dataset are intentionally excluded from GitHub.
