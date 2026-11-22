# MedTrust-AI Error Transition Analysis v1

This analysis compares each shifted prediction against the same image's clean prediction. No model training or inference was rerun.

## Clean confusion counts

- True positives: 1370
- True negatives: 1266
- False positives: 273
- False negatives: 169

## Severe-shift transitions

| Rank | Shift | Correct→Wrong | Wrong→Correct | Net new errors | FP change | FN change |
|---:|---|---:|---:|---:|---:|---:|
| 1 | gaussian_blur | 570 | 115 | 455 | +557 | -102 |
| 2 | gaussian_noise | 465 | 115 | 350 | +375 | -25 |
| 3 | resolution_degradation | 412 | 109 | 303 | +394 | -91 |
| 4 | brightness | 109 | 72 | 37 | +62 | -25 |
| 5 | contrast | 89 | 63 | 26 | -25 | +51 |

## Main findings

- Largest net increase in errors: **gaussian_blur** (455 additional errors).
- Largest false-positive increase: **gaussian_blur** (+557).
- Largest false-negative increase: **contrast** (+51).

These results are descriptive validation analyses and are not statistical significance tests.
