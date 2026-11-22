# MedTrust-AI — Severity Trajectory Analysis v1

This analysis evaluates whether reliability-failure signature dimensions evolve systematically from mild to moderate to severe shift.

**Important:** this remains an exploratory validation-only analysis. The signature scores are study-internal descriptive constructs, not validated clinical metrics.

## Dominant signature trajectory by shift family

| Shift family | Mild | Moderate | Severe | Signature switches |
|---|---|---|---|---:|
| brightness | silent_confidence_failure | silent_confidence_failure | failure_awareness_collapse | 1 |
| contrast | silent_confidence_failure | silent_confidence_failure | silent_confidence_failure | 0 |
| gaussian_blur | silent_confidence_failure | failure_awareness_collapse | directional_bias | 2 |
| gaussian_noise | silent_confidence_failure | performance_collapse | performance_collapse | 1 |
| resolution_degradation | silent_confidence_failure | failure_awareness_collapse | directional_bias | 2 |

## Signature-specific trajectories

| Shift | Signature | Mild | Moderate | Severe | Pattern |
|---|---|---:|---:|---:|---|
| brightness | performance_collapse | 0.002 | 0.036 | 0.081 | monotonic_increase |
| brightness | failure_awareness_collapse | 0.283 | 0.337 | 0.382 | monotonic_increase |
| brightness | silent_confidence_failure | 0.588 | 0.536 | 0.319 | monotonic_decrease |
| brightness | directional_bias | 0.000 | 0.079 | 0.112 | monotonic_increase |
| contrast | performance_collapse | 0.004 | 0.027 | 0.065 | monotonic_increase |
| contrast | failure_awareness_collapse | 0.333 | 0.245 | 0.222 | monotonic_decrease |
| contrast | silent_confidence_failure | 0.617 | 0.446 | 0.352 | monotonic_decrease |
| contrast | directional_bias | 0.000 | 0.025 | 0.095 | monotonic_increase |
| gaussian_blur | performance_collapse | 0.027 | 0.160 | 0.854 | monotonic_increase |
| gaussian_blur | failure_awareness_collapse | 0.346 | 0.398 | 0.658 | monotonic_increase |
| gaussian_blur | silent_confidence_failure | 0.490 | 0.224 | 0.313 | moderate_dip |
| gaussian_blur | directional_bias | 0.137 | 0.262 | 1.000 | monotonic_increase |
| gaussian_noise | performance_collapse | 0.102 | 0.341 | 0.844 | monotonic_increase |
| gaussian_noise | failure_awareness_collapse | 0.290 | 0.160 | 0.610 | moderate_dip |
| gaussian_noise | silent_confidence_failure | 0.314 | 0.063 | 0.000 | monotonic_decrease |
| gaussian_noise | directional_bias | 0.012 | 0.247 | 0.598 | monotonic_increase |
| resolution_degradation | performance_collapse | 0.017 | 0.120 | 0.578 | monotonic_increase |
| resolution_degradation | failure_awareness_collapse | 0.280 | 0.296 | 0.637 | monotonic_increase |
| resolution_degradation | silent_confidence_failure | 0.403 | 0.245 | 0.434 | moderate_dip |
| resolution_degradation | directional_bias | 0.065 | 0.203 | 0.730 | monotonic_increase |

## Strongest mild-to-severe increases

- **performance_collapse**: gaussian_blur (Δ=+0.828).
- **failure_awareness_collapse**: resolution_degradation (Δ=+0.357).
- **silent_confidence_failure**: resolution_degradation (Δ=+0.031).
- **directional_bias**: gaussian_blur (Δ=+0.863).

## Dominant-signature stability

Families with the same dominant signature at mild, moderate, and severe levels:

- contrast: `silent_confidence_failure`

Families whose dominant signature changed with severity:

- brightness: silent_confidence_failure → silent_confidence_failure → failure_awareness_collapse
- gaussian_blur: silent_confidence_failure → failure_awareness_collapse → directional_bias
- gaussian_noise: silent_confidence_failure → performance_collapse → performance_collapse
- resolution_degradation: silent_confidence_failure → failure_awareness_collapse → directional_bias

## Interpretation boundary

A monotonic trajectory would support the hypothesis that a failure dimension systematically strengthens as degradation increases. A signature switch suggests that the failure mechanism may qualitatively change with severity. These observations are hypothesis-generating and require later statistical and external validation before being presented as a general taxonomy.
