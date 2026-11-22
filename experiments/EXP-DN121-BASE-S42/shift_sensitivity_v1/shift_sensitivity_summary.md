# MedTrust-AI Shift Sensitivity Analysis v1

This analysis uses the existing saved validation results. No model training or inference was rerun.

## Clean reference

- Accuracy: 0.8564
- AUROC: 0.9279
- Failure rate: 0.1436
- Failure-detection AUROC: 0.7781
- AURC: 0.0543

## Severe-shift ranking

| Overall rank | Shift | Accuracy drop (pp) | AUROC drop | Failure increase (pp) | FD-AUROC drop | AURC increase |
|---:|---|---:|---:|---:|---:|---:|
| 1 | gaussian_blur | 14.78 | 0.0381 | 14.78 | 0.0466 | 0.0900 |
| 2 | gaussian_noise | 11.37 | 0.0675 | 11.37 | 0.0536 | 0.0784 |
| 3 | resolution_degradation | 9.84 | 0.0278 | 9.84 | 0.0560 | 0.0638 |
| 4 | brightness | 1.20 | 0.0070 | 1.20 | 0.0239 | 0.0098 |
| 5 | contrast | 0.84 | 0.0071 | 0.84 | -0.0016 | 0.0019 |

## Metric-specific findings

- Largest severe accuracy drop: **gaussian_blur** (14.78 pp)
- Largest severe AUROC drop: **gaussian_noise** (0.0675)
- Largest severe failure-rate increase: **gaussian_blur** (14.78 pp)
- Largest severe failure-detection AUROC drop: **resolution_degradation** (0.0560)
- Largest severe AURC increase: **gaussian_blur** (0.0900)

## Interpretation

The ranking is descriptive and uses only the already-completed validation experiments. The overall severe-damage rank is the mean ordinal rank across five degradation measures: accuracy drop, classification AUROC drop, failure-rate increase, failure-detection AUROC drop, and AURC increase. It should not be interpreted as a new statistical significance test.
