# MedTrust-AI

## Failure Detection and Selective Prediction for Chest X-Ray Classification Under Distribution Shift

MedTrust-AI is a controlled research project investigating the reliability of
chest X-ray classifiers under progressively increasing distribution shift.

The project studies whether confidence and uncertainty measures can identify
prediction failures and whether selective prediction can reduce prediction
risk by abstaining on unreliable cases.

### Primary Research Question

How reliably do confidence- and uncertainty-based methods identify prediction
failures as chest X-ray data progressively shifts away from the training
distribution, and how does this affect selective-prediction performance?

### Primary Dataset

RSNA Pneumonia Detection Challenge.

The final cohort definition and biological patient grouping will be resolved
and frozen before primary model development.

### Planned Reliability Methods

- Maximum prediction confidence
- Predictive entropy
- Monte Carlo Dropout

### Planned Distribution Shifts

- Gaussian blur
- Gaussian noise
- Contrast shift
- Illumination shift
- Resolution degradation

Final shift parameters and severity levels are not yet frozen.

### Planned Evaluation

- Classification performance
- Calibration
- Prediction-failure detection
- Selective prediction
- Risk-coverage analysis
- External validation
- Distribution-distance analysis

### Reproducibility Policy

All primary experiments must use the frozen master cohort, biological
patient-group assignments, preprocessing pipeline, and evaluation protocol.

No team member may independently modify patient assignments, primary labels,
preprocessing, primary distribution-shift definitions, or final evaluation
procedures.

Any protocol modification must be versioned and documented before use.
