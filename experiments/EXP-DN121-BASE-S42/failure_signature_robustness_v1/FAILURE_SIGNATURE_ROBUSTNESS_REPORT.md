# MedTrust-AI — Failure Signature Robustness Analysis v1

This analysis tests whether the exploratory failure-signature conclusions depend strongly on arbitrary scoring choices.

Variants include alternative normalization, weighting, adverse clipping, and leave-one-component-out analyses.

**Important:** robustness of a descriptive signature does not make it a validated clinical metric or prove novelty.

Total scoring variants tested: **14**

## Condition-level dominant-signature consensus

| Condition | Consensus dominant | Agreement | Unique dominant labels |
|---|---|---:|---:|
| brightness__mild | silent_confidence_failure | 100.0% | 1 |
| brightness__moderate | silent_confidence_failure | 100.0% | 1 |
| brightness__severe | failure_awareness_collapse | 85.7% | 2 |
| contrast__mild | silent_confidence_failure | 100.0% | 1 |
| contrast__moderate | silent_confidence_failure | 100.0% | 1 |
| contrast__severe | silent_confidence_failure | 92.9% | 2 |
| gaussian_blur__mild | silent_confidence_failure | 85.7% | 2 |
| gaussian_blur__moderate | failure_awareness_collapse | 85.7% | 2 |
| gaussian_blur__severe | directional_bias | 92.9% | 2 |
| gaussian_noise__mild | silent_confidence_failure | 57.1% | 3 |
| gaussian_noise__moderate | performance_collapse | 100.0% | 1 |
| gaussian_noise__severe | performance_collapse | 92.9% | 2 |
| resolution_degradation__mild | silent_confidence_failure | 85.7% | 2 |
| resolution_degradation__moderate | failure_awareness_collapse | 71.4% | 3 |
| resolution_degradation__severe | directional_bias | 92.9% | 2 |

## Family trajectory robustness

| Shift family | Consensus trajectory | Agreement | Unique trajectories |
|---|---|---:|---:|
| brightness | silent_confidence_failure -> silent_confidence_failure -> failure_awareness_collapse | 85.7% | 2 |
| contrast | silent_confidence_failure -> silent_confidence_failure -> silent_confidence_failure | 92.9% | 2 |
| gaussian_blur | silent_confidence_failure -> failure_awareness_collapse -> directional_bias | 64.3% | 4 |
| gaussian_noise | silent_confidence_failure -> performance_collapse -> performance_collapse | 50.0% | 4 |
| resolution_degradation | silent_confidence_failure -> failure_awareness_collapse -> directional_bias | 57.1% | 5 |

## Interpretation

High agreement across reasonable scoring variants suggests that the observed failure pattern is less likely to be an artifact of one normalization or weighting choice. Low agreement indicates that the descriptive taxonomy is unstable and should not be formalized without further methodological work.
