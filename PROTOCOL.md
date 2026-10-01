# MedTrust-AI Research Protocol

Version: 0.1  
Status: DEVELOPMENT

## Study objective

Investigate the reliability of chest X-ray classifiers under progressively
increasing distribution shift. Assess whether confidence and uncertainty
measures identify prediction failures and whether abstaining on unreliable
cases reduces prediction risk.

## Primary research question

How reliably do confidence- and uncertainty-based methods identify prediction
failures as chest X-ray data progressively shifts away from the training
distribution, and how does this affect selective-prediction performance?

## Primary dataset

RSNA Pneumonia Detection Challenge. Dataset/cohort version: 0.1 DEVELOPMENT.
Final primary label definitions and dataset inclusion/exclusion decisions: TBD.

## Cohort construction rule

Construct and freeze one shared master cohort before primary model development.
Final cohort construction, eligibility criteria, exclusions, and identifier
mapping: TBD. Do not modify raw data. Do not create member-specific primary
cohorts or independently change primary labels.

## Biological patient-grouping rule

All images/examinations belonging to the same biological patient must remain
in the same fold. Biological patient identification and grouping procedures:
TBD; dataset identifiers must not be assumed to establish biological identity
without resolution. Freeze one shared biological patient assignment source.

## Data-splitting rule

Use patient-grouped splitting with five planned folds. Patient assignments are
not yet frozen. Final fold construction procedure, fold roles, and split seeds:
TBD. No team member may independently regenerate patient folds. All primary
experiments must use the same frozen patient assignments and fold version.

## Shared preprocessing rule

All primary experiments must use one shared, versioned preprocessing pipeline.
Final preprocessing parameters are not yet frozen. DICOM processing, resizing,
intensity processing, channel handling, normalization, and CLAHE decisions:
TBD, as recorded in configs/preprocessing_v1.yaml. Member-specific primary
preprocessing is not permitted.

## Planned primary models

- ResNet-50
- DenseNet-121

Model configurations and training settings: TBD. No primary model training has
started.

## Planned reliability methods

- Maximum prediction confidence
- Predictive entropy
- Monte Carlo Dropout

Method configurations, including dropout settings and stochastic sample count:
TBD.

## Planned distribution shifts

- Gaussian blur
- Gaussian noise
- Contrast
- Illumination
- Resolution degradation

Final shift parameters are not yet frozen. Severity levels, implementations,
and shift random seeds: TBD. Primary comparisons must use the same frozen,
versioned distribution-shift configuration.

## Selective prediction and risk-coverage analysis

Evaluate classification performance, calibration, prediction-failure detection,
selective prediction, and risk-coverage behavior. Final metrics, failure
definitions, abstention rules, calibration procedures, threshold selection,
distribution-distance analysis, and statistical evaluation procedures: TBD.
Freeze the primary evaluation protocol before primary experiments.

## External validation

External dataset: TBD, not selected. Dataset selection, cohort alignment,
label mapping, and external evaluation procedures remain DEVELOPMENT and
must be documented and versioned before use.

## Reproducibility requirements

The frozen master cohort, biological patient assignments, patient folds,
preprocessing configuration, distribution-shift configuration, and primary
evaluation protocol will be the shared source of truth. Do not create separate
member-specific versions of these primary research files. Researchers may work
on separate Git branches, but primary experimental comparisons must use the
same frozen cohort, folds, preprocessing, and evaluation protocol.

For every primary experiment, preserve:

- Experiment ID
- Dataset/cohort version
- Fold version
- Preprocessing version
- Model configuration
- Random seed
- Git commit
- Checkpoint
- Training history
- Patient/image identifiers
- Raw logits
- Probabilities
- Calibration parameters
- Uncertainty outputs
- Software/environment information

Where applicable, also preserve:

- Individual MC-Dropout stochastic predictions
- Feature embeddings
- Distribution-shift parameters
- Distribution-shift random seeds

Record experiments in EXPERIMENT_REGISTRY.csv with their output locations.
Preservation does not require committing datasets, model weights, generated
results, or large binaries to this repository. Final artifact storage and
retention procedures: TBD.

## Protocol-change policy

Primary protocol changes must be documented and versioned before use.
Record changes in CHANGELOG.md and retain the protocol/configuration versions
associated with each experiment. No researcher may independently modify
patient assignments, primary labels, preprocessing, primary shift definitions,
or final evaluation procedures. Shared change-review procedure: TBD.

## Negative-result policy

Unexpected or negative results must not be removed solely because they do not
support a hypothesis. Preserve experiment records and outputs, and document
any scientifically justified correction or exclusion with its reason and
applicable protocol version.

## Current status

- Protocol version 0.1 is DEVELOPMENT.
- No primary model training has started.
- Cohort definition and biological patient grouping are TBD.
- Patient assignments are not yet frozen.
- Final preprocessing parameters are not yet frozen.
- Final shift parameters are not yet frozen.
- Primary evaluation procedures are TBD.
- External validation dataset has not been selected.
