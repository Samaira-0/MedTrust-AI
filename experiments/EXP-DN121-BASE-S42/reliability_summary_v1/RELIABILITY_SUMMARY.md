# MedTrust-AI — Consolidated Reliability Summary

This report consolidates the completed validation-only shift sensitivity, error-transition, and confidence-transition analyses. No training or inference was rerun.

## Severe shift comparison

| Rank | Shift | Acc drop (pp) | AUROC drop | Failure +pp | FD-AUROC drop | AURC + | Net new errors | FP change | FN change | C→W more uncertain | Wrong conf≥0.90 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | gaussian_blur | 14.78 | 0.0381 | 14.78 | 0.0466 | 0.0900 | 455 | +557 | -102 | 81.2% | 14 |
| 2 | gaussian_noise | 11.37 | 0.0675 | 11.37 | 0.0536 | 0.0784 | 350 | +375 | -25 | 86.5% | 1 |
| 3 | resolution_degradation | 9.84 | 0.0278 | 9.84 | 0.0560 | 0.0638 | 303 | +394 | -91 | 79.1% | 16 |
| 4 | brightness | 1.20 | 0.0070 | 1.20 | 0.0239 | 0.0098 | 37 | +62 | -25 | 64.2% | 34 |
| 5 | contrast | 0.84 | 0.0071 | 0.84 | -0.0016 | 0.0019 | 26 | -25 | +51 | 62.9% | 39 |

## Main findings

- **Gaussian blur** was the most damaging overall severe shift. It caused the largest accuracy drop, failure-rate increase, AURC increase, and net increase in errors.
- **Gaussian noise** caused the largest classification AUROC drop.
- **Resolution degradation** caused the largest drop in failure-detection AUROC.
- Severe blur produced a strong rise in false positives, while severe contrast produced the largest false-negative increase.
- Under blur, noise, and resolution degradation, most newly wrong predictions became more uncertain. In contrast and brightness shifts, the uncertainty response was weaker.
- High-confidence wrong predictions remained especially visible under severe contrast and brightness shifts.

## Interpretation

The reliability impact of distribution shift is not uniform. Some shifts mainly reduce classification performance, others weaken failure detection, and some leave the model confidently wrong. Therefore reliability should be evaluated using multiple complementary metrics rather than accuracy alone.

All findings are validation-development results and should not be interpreted as independent test conclusions or statistical significance tests.
