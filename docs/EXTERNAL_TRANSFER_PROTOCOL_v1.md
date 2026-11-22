# MedTrust-AI External Transfer Protocol v1

## Research question

Can controlled progressive synthetic stress tests on NIH ChestX-ray14
predict or explain reliability-failure behavior under natural
cross-dataset distribution shift?

## Development source

NIH ChestX-ray14
Task: Effusion vs No Finding

The existing NIH train/validation/test split remains frozen.

The NIH locked test set MUST NOT be accessed during external-protocol
development.

## External dataset 1

CheXpert

Primary external task:
Pleural Effusion vs No Finding

## Label harmonization

Positive:
- Pleural Effusion == 1
- Other positive findings may coexist.
- This mirrors the NIH positive definition, where Effusion may occur
  in a multilabel image.

Negative:
- No Finding == 1
- Pleural Effusion must not be positive or uncertain.

Excluded:
- Pleural Effusion == -1 (uncertain)
- ambiguous/unusable labels
- lateral images
- non-frontal images

No uncertainty relabeling is permitted for the primary analysis.

## View policy

Use frontal chest radiographs only.

Allowed frontal orientations:
- AP
- PA

Lateral views are excluded.

## Patient handling

Patient identity must remain available throughout cohort construction.

Maximum:
5 selected images per patient.

If deterministic subsampling is necessary:
seed = 42.

No patient may be duplicated through accidental metadata joins.

## External evaluation cohorts

### A. Native-prevalence cohort

Preserve the naturally occurring eligible external class distribution.

Purpose:
evaluate real external-domain reliability behavior.

### B. Balanced-comparison cohort

Create an equal-size Pleural Effusion / No Finding cohort.

Purpose:
permit cleaner comparison against the balanced NIH development cohort.

The balanced cohort is a secondary comparison cohort and does not
replace native-prevalence evaluation.

## Frozen model policy

The external dataset must NOT be used for:

- model training
- fine-tuning
- temperature fitting
- calibration fitting
- threshold optimization
- hyperparameter selection
- architecture selection
- shift-signature construction

Use the existing frozen NIH DenseNet-121 checkpoint.

Use the temperature obtained on NIH validation.

Use the classification threshold fixed from NIH validation.

## Frozen preprocessing

Use exactly the NIH inference preprocessing:

1. load as grayscale
2. convert to 3 channels
3. resize to 224 x 224 according to the frozen implementation
4. scale to [0,1]
5. ImageNet normalization

No external-dataset-specific enhancement is permitted.

No CLAHE.
No histogram equalization.
No external normalization fitting.

## External metrics

Calculate the same reliability quantities used in NIH:

### Classification
- AUROC
- AUPRC
- accuracy
- sensitivity
- specificity
- F1
- BCE/NLL

### Calibration
- Brier score
- ECE

Do not refit temperature on external data.

### Failure detection
- failure rate
- failure-detection AUROC
- failure-detection AUPRC

### Selective prediction
- risk-coverage curve
- AURC
- risks at frozen reporting coverages

### Failure behavior
- false positives
- false negatives
- high-confidence wrong predictions
- confidence distribution among errors

## Synthetic-to-natural transfer analysis

Construct an external reliability-failure vector using the same
components available for the controlled NIH shifts.

Compare the natural external vector against the controlled synthetic
shift trajectories.

The primary scientific question is NOT simply whether external AUROC
decreases.

The question is whether controlled synthetic stress tests explain or
predict HOW reliability fails under natural domain shift.

## Analysis boundary

Synthetic shift results remain development analyses.

External CheXpert results must not be used to redesign the signature
after inspection.

Any later change to the method requires a new protocol version and
must be documented.

## Future replication

If the CheXpert experiment is informative, replicate the frozen method
on MIMIC-CXR-JPG.

MIMIC replication must not be used to tune the CheXpert-derived
conclusion.

## Locked NIH test

NIH test access remains prohibited until:

1. methodology is frozen;
2. external-cohort construction is frozen;
3. statistical analysis is frozen;
4. collaborator experiments are integrated;
5. final evaluation is explicitly approved.


## CheXpert metadata source clarification

The primary external cohort metadata source is:

`train_cheXbert.csv`

Therefore, the external labels used in this experiment are the
CheXbert-generated report labels supplied with the accessed CheXpert
resource.

This label provenance must be reported explicitly in all methods and
limitations sections.

The analysis must not describe these labels as image-level radiologist
ground truth.

## Patients contributing to both classes

A patient may have different eligible studies with different labels
across time.

Patients with both Pleural Effusion and No Finding eligible studies are
not automatically excluded.

However:

- the maximum 5-image cap applies globally across all eligible studies
  from that patient, not separately within each class;
- patient identity must remain in every manifest;
- later uncertainty estimates / confidence intervals must account for
  within-patient dependence;
- no external patient may contribute more than 5 selected images in
  either the native-prevalence or balanced-comparison cohort.
