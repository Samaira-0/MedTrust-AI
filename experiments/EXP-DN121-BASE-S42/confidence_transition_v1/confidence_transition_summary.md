# MedTrust-AI Confidence Transition Analysis v1

This analysis uses saved validation predictions only. No model training or inference was rerun.

High-confidence error thresholds are reported at confidence >= 0.80 and >= 0.90.

## Severe shifts

| Shift | C→W | C→W more uncertain | C→W conf≥0.90 | All wrong conf≥0.90 |
|---|---:|---:|---:|---:|
| contrast | 89 | 62.9% | 0 | 39 |
| brightness | 109 | 64.2% | 0 | 34 |
| resolution_degradation | 412 | 79.1% | 2 | 16 |
| gaussian_blur | 570 | 81.2% | 2 | 14 |
| gaussian_noise | 465 | 86.5% | 0 | 1 |

## Main findings

- Most severe high-confidence wrong predictions: **contrast** (39).
- Most newly-created high-confidence errors: **resolution_degradation** (2).
- Lowest proportion of correct→wrong cases that became more uncertain: **contrast** (62.9%).

Interpretation should remain descriptive. These are validation-development findings and not statistical significance tests.
