# MedTrust-AI

## Failure Detection and Selective Prediction for Chest X-Ray Classification Under Progressive Distribution Shift

MedTrust-AI is a controlled research project investigating how the reliability of a chest X-ray classifier changes under progressively increasing distribution shift.

The project focuses on whether confidence- and uncertainty-based methods can identify prediction failures, and whether selective prediction can reduce risk by abstaining on uncertain cases.

## Primary Research Question

**How reliably do confidence- and uncertainty-based methods identify prediction failures as chest X-ray data progressively shifts away from the training distribution, and how does this affect selective-prediction performance?**

## Primary Dataset

**NIH ChestX-ray14**

Primary binary classification endpoint:

- **Positive class:** Effusion present
- **Negative class:** No Finding

Multi-label cases containing Effusion are included in the positive class.

The original NIH labels are report/NLP-derived and are treated as noisy labels rather than radiologist-confirmed clinical ground truth.

## Frozen Cohort

The final cohort has been frozen and versioned.

- Total images: **19,326**
- Train: **12,384**
- Validation: **3,078**
- Test: **3,864**
- Unique patients: **10,490**
- Maximum images per patient: **5**
- Cross-split patient overlap: **0**
- Duplicate image IDs: **0**
- Seed: **42**

Frozen cohort artifacts are stored in:

- `manifests/nih_effusion_vs_nofinding_v1.csv`
- `manifests/nih_effusion_vs_nofinding_v1_summary.json`
- `manifests/nih_effusion_vs_nofinding_v1_checksums.txt`

## Preprocessing

Frozen preprocessing pipeline:

- 224×224 chest X-ray images
- grayscale images replicated to 3 channels
- pixel scaling to [0,1]
- ImageNet normalization
- no CLAHE
- no histogram equalization
- no denoising
- no sharpening
- no contrast enhancement

Training-only augmentation:

- rotation ±7°
- translation up to 5%
- no horizontal flip
- no vertical flip

## Baseline Model

Architecture: **DenseNet-121**

Configuration:

- ImageNet pretrained weights
- full fine-tuning
- Dropout: 0.2
- optimizer: AdamW
- learning rate: 1e-4
- weight decay: 1e-4
- loss: BCEWithLogitsLoss
- maximum epochs: 20
- early stopping on validation AUROC
- seed: 42

Best checkpoint: **Epoch 2**

## Clean Validation Performance

- AUROC: **0.9279**
- AUPRC: **0.9269**
- Accuracy: **0.8564**
- Sensitivity: **0.8902**
- Specificity: **0.8226**
- F1: **0.8611**
- Failure rate: **0.1436**

The test set has not yet been used for primary evaluation.

## Calibration

Validation-only temperature scaling was performed.

- Temperature: **1.0016**
- ECE before calibration: **0.01948**
- ECE after calibration: **0.01901**

The near-unit temperature indicates that the clean validation predictions were already close to globally calibrated.

## Failure Detection

Deterministic confidence-based failure detection:

- Failure-detection AUROC: **0.7781**
- Failure-detection AUPRC: **0.3739**
- Random AUPRC baseline: **0.1436**

## MC Dropout

Monte Carlo Dropout was evaluated as an uncertainty estimator.

On clean validation:

- deterministic confidence FD AUROC: **0.7781**
- MC predictive entropy FD AUROC: **0.7785**

Across all 16 shift conditions:

- MC entropy AUROC better than deterministic: **5/16**
- MC entropy AUROC worse than deterministic: **11/16**
- MC entropy AUPRC better: **6/16**
- MC entropy AUPRC worse: **10/16**

Current evidence does **not** show consistent superiority of MC Dropout over deterministic confidence.

## Selective Prediction

For MC predictive entropy:

- Full coverage risk: **14.36%**
- 90% coverage risk: **10.76%**
- 80% coverage risk: **8.77%**
- 70% coverage risk: **7.43%**
- 50% coverage risk: **5.33%**

Selective prediction substantially reduced accepted-case risk as coverage decreased.

## Frozen Distribution-Shift Protocol

Five controlled shift families were evaluated:

1. Brightness reduction
2. Contrast reduction
3. Gaussian noise
4. Gaussian blur
5. Resolution degradation

Each family was evaluated at mild, moderate, and severe levels.

Together with clean validation, this produced **16 total conditions**.

No retraining, fine-tuning, or shift-specific recalibration was performed.

## Key Distribution-Shift Findings

### Severe Gaussian Blur

- Accuracy: **0.7086**
- Failure rate: **0.2914**
- Failure-detection AUROC: **0.7315**
- AURC: **0.1443**

### Severe Gaussian Noise

- Classification AUROC: **0.8604**
- Accuracy: **0.7427**
- Failure rate: **0.2573**
- Failure-detection AUROC: **0.7244**
- AURC: **0.1327**

### Severe Resolution Degradation

- Accuracy: **0.7580**
- Failure rate: **0.2420**
- Failure-detection AUROC: **0.7220**
- AURC: **0.1182**

The main current result is that as distribution shift becomes stronger, the classifier becomes less reliable and its ability to recognize its own failures also weakens.

## Current Experimental Interpretation

The current experiments suggest that:

1. Classification performance deteriorates under increasing distribution shift.
2. Gaussian blur, Gaussian noise, and resolution degradation produce the strongest deterioration.
3. Confidence-based failure detection also weakens under strong shifts.
4. Selective prediction can reduce risk by abstaining on uncertain predictions.
5. MC Dropout provides only small and inconsistent gains over deterministic confidence.

Negative results are retained and reported rather than discarded.

## Reproducibility Policy

All primary experiments use frozen:

- patient-level cohort assignments
- preprocessing
- augmentation
- model configuration
- calibration procedure
- shift definitions
- evaluation procedures

Any protocol modification must be versioned and documented before use.

Large model checkpoint binaries and the NIH image dataset are intentionally excluded from GitHub.

## Repository Structure

- `configs/` — frozen experiment configurations
- `docs/` — research and results documentation
- `experiments/` — validation and distribution-shift outputs
- `manifests/` — frozen cohort manifests
- `notebooks/` — Kaggle training notebooks
- `reports/` — QC and validation reports
- `scripts/` — reproducible experiment scripts
- `src/` — model and source code

## Current Status

Completed:

- cohort freezing
- image QC
- preprocessing freeze
- DenseNet-121 baseline training
- calibration
- deterministic failure detection
- MC Dropout evaluation
- selective prediction
- frozen distribution-shift protocol
- deterministic 16-condition shift evaluation
- MC Dropout 16-condition shift evaluation
- summary analysis

## Detailed Results

See:

- `docs/RESULTS_SUMMARY.md`
- `docs/CONTRIBUTIONS.md`
- `experiments/EXP-DN121-BASE-S42/`
