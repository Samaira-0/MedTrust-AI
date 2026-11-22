# MedTrust-AI — Reliability Failure Signature Analysis v1

This is an exploratory validation-only analysis. It characterizes how different distribution shifts fail, rather than measuring only whether performance decreases.

**Important:** the four signature scores are descriptive study-internal constructs, not validated clinical metrics or statistical significance tests.

## Signature dimensions

- **Performance collapse:** accuracy, classification AUROC, and failure-rate deterioration.
- **Failure-awareness collapse:** deterioration in failure-detection AUROC/AURC plus newly wrong predictions that do not become more uncertain.
- **Silent-confidence failure:** wrong predictions that remain high-confidence, especially newly created errors.
- **Directional bias:** asymmetric change between false positives and false negatives.

## Severe conditions

| Shift | Performance | Awareness | Silent confidence | Directional bias | Dominant signature | Secondary |
|---|---:|---:|---:|---:|---|---|
| brightness | 0.081 | 0.382 | 0.319 | 0.112 | failure_awareness_collapse | silent_confidence_failure |
| contrast | 0.065 | 0.222 | 0.352 | 0.095 | silent_confidence_failure | failure_awareness_collapse |
| gaussian_blur | 0.854 | 0.658 | 0.313 | 1.000 | directional_bias | performance_collapse |
| gaussian_noise | 0.844 | 0.610 | 0.000 | 0.598 | performance_collapse | failure_awareness_collapse |
| resolution_degradation | 0.578 | 0.637 | 0.434 | 0.730 | directional_bias | failure_awareness_collapse |

## Family-level patterns

- **brightness**: dominant = `silent_confidence_failure`, secondary = `failure_awareness_collapse`.
- **contrast**: dominant = `silent_confidence_failure`, secondary = `failure_awareness_collapse`.
- **gaussian_blur**: dominant = `failure_awareness_collapse`, secondary = `directional_bias`.
- **gaussian_noise**: dominant = `performance_collapse`, secondary = `failure_awareness_collapse`.
- **resolution_degradation**: dominant = `failure_awareness_collapse`, secondary = `silent_confidence_failure`.

## Interpretation boundary

The dominant label simply identifies the largest normalized descriptive score within this experiment. It does not imply that the shift belongs to a universally validated failure class. The purpose of this stage is hypothesis generation for a possible formal reliability-failure taxonomy or silent-failure metric.
