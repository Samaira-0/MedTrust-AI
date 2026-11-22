#!/usr/bin/env python3

import csv
import json
import hashlib
from pathlib import Path
from statistics import mean
from datetime import datetime, timezone

BASE = Path("experiments/EXP-DN121-BASE-S42")

S1 = BASE / "shift_sensitivity_v1" / "shift_sensitivity_condition_deltas.csv"
S2 = BASE / "error_transition_v1" / "error_transition_summary_all_conditions.csv"
S3 = BASE / "confidence_transition_v1" / "confidence_transition_summary_all_conditions.csv"

OUT = BASE / "failure_signature_v1"
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_CONDITIONS = 15
VALIDATION_N = 3078


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def harm(x):
    """Adverse direction only. Improvements are clipped to zero."""
    return max(float(x), 0.0)


def minmax(values):
    lo = min(values)
    hi = max(values)

    if hi == lo:
        return [0.0 for _ in values]

    return [(x - lo) / (hi - lo) for x in values]


# --------------------------------------------------
# 1. Load and align all three completed analyses
# --------------------------------------------------

s1_rows = read_csv(S1)
s2_rows = read_csv(S2)
s3_rows = read_csv(S3)

s1 = {r["condition_id"]: r for r in s1_rows}
s2 = {r["condition_id"]: r for r in s2_rows}
s3 = {r["condition_id"]: r for r in s3_rows}

conditions = sorted(set(s1) & set(s2) & set(s3))

if len(conditions) != EXPECTED_CONDITIONS:
    raise RuntimeError(
        f"Expected {EXPECTED_CONDITIONS} aligned shift conditions, "
        f"found {len(conditions)}"
    )

if set(s1) != set(s2) or set(s1) != set(s3):
    raise RuntimeError(
        "S1/S2/S3 condition IDs are not perfectly aligned."
    )


# --------------------------------------------------
# 2. Build raw failure-signature features
# --------------------------------------------------

rows = []

for condition_id in conditions:

    a = s1[condition_id]
    b = s2[condition_id]
    c = s3[condition_id]

    correct_to_wrong = int(b["correct_to_wrong"])

    c2w_high_conf_90 = int(
        c["c2w_high_confidence_90_count"]
    )

    new_high_conf_error_rate = (
        c2w_high_conf_90 / correct_to_wrong
        if correct_to_wrong > 0
        else 0.0
    )

    c2w_more_uncertain_rate = float(
        c["c2w_more_uncertain_rate"]
    )

    unaware_new_error_rate = (
        1.0 - c2w_more_uncertain_rate
    )

    fp_change = int(b["fp_change_vs_clean"])
    fn_change = int(b["fn_change_vs_clean"])

    directional_bias_rate = (
        abs(fp_change - fn_change) / VALIDATION_N
    )

    rows.append({
        "condition_id": condition_id,
        "shift_family": a["shift_family"],
        "severity": a["severity"],

        # Performance damage
        "accuracy_drop_pp":
            harm(a["accuracy_drop_percentage_points"]),

        "classification_auroc_drop":
            harm(a["auroc_drop"]),

        "failure_rate_increase_pp":
            harm(a["failure_rate_increase_percentage_points"]),

        # Failure-awareness damage
        "fd_auroc_drop":
            harm(a["fd_auroc_drop"]),

        "aurc_increase":
            harm(a["aurc_increase"]),

        "new_error_unaware_rate":
            unaware_new_error_rate,

        # Silent confidence
        "all_wrong_high_conf_90_rate":
            float(c["all_wrong_high_confidence_90_rate"]),

        "new_wrong_high_conf_90_rate":
            new_high_conf_error_rate,

        # Directionality
        "fp_change_vs_clean": fp_change,
        "fn_change_vs_clean": fn_change,

        "directional_bias_rate":
            directional_bias_rate,

        # Supporting counts
        "correct_to_wrong":
            correct_to_wrong,

        "net_new_errors":
            int(b["net_new_errors"]),
    })


# --------------------------------------------------
# 3. Normalize each harmful dimension across
#    the 15 development conditions
# --------------------------------------------------

feature_names = [
    "accuracy_drop_pp",
    "classification_auroc_drop",
    "failure_rate_increase_pp",
    "fd_auroc_drop",
    "aurc_increase",
    "new_error_unaware_rate",
    "all_wrong_high_conf_90_rate",
    "new_wrong_high_conf_90_rate",
    "directional_bias_rate",
]

for feature in feature_names:

    normalized = minmax(
        [float(r[feature]) for r in rows]
    )

    for r, value in zip(rows, normalized):
        r[f"norm_{feature}"] = value


# --------------------------------------------------
# 4. Construct four exploratory signature scores
# --------------------------------------------------

for r in rows:

    performance_score = mean([
        r["norm_accuracy_drop_pp"],
        r["norm_classification_auroc_drop"],
        r["norm_failure_rate_increase_pp"],
    ])

    awareness_score = mean([
        r["norm_fd_auroc_drop"],
        r["norm_aurc_increase"],
        r["norm_new_error_unaware_rate"],
    ])

    silent_score = mean([
        r["norm_all_wrong_high_conf_90_rate"],
        r["norm_new_wrong_high_conf_90_rate"],
        r["norm_new_error_unaware_rate"],
    ])

    directional_score = (
        r["norm_directional_bias_rate"]
    )

    signature_scores = {
        "performance_collapse":
            performance_score,

        "failure_awareness_collapse":
            awareness_score,

        "silent_confidence_failure":
            silent_score,

        "directional_bias":
            directional_score,
    }

    ordered = sorted(
        signature_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    r["performance_collapse_score"] = performance_score
    r["failure_awareness_collapse_score"] = awareness_score
    r["silent_confidence_failure_score"] = silent_score
    r["directional_bias_score"] = directional_score

    r["dominant_signature"] = ordered[0][0]
    r["dominant_signature_score"] = ordered[0][1]

    r["secondary_signature"] = ordered[1][0]
    r["secondary_signature_score"] = ordered[1][1]

    r["dominance_margin"] = (
        ordered[0][1] - ordered[1][1]
    )


# --------------------------------------------------
# 5. Sort for readable output
# --------------------------------------------------

severity_order = {
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}

rows.sort(
    key=lambda r: (
        r["shift_family"],
        severity_order[r["severity"]]
    )
)


# --------------------------------------------------
# 6. Save full condition-level signature table
# --------------------------------------------------

condition_csv = (
    OUT /
    "failure_signature_all_conditions.csv"
)

fields = [
    "condition_id",
    "shift_family",
    "severity",

    "accuracy_drop_pp",
    "classification_auroc_drop",
    "failure_rate_increase_pp",

    "fd_auroc_drop",
    "aurc_increase",
    "new_error_unaware_rate",

    "all_wrong_high_conf_90_rate",
    "new_wrong_high_conf_90_rate",

    "fp_change_vs_clean",
    "fn_change_vs_clean",
    "directional_bias_rate",

    "correct_to_wrong",
    "net_new_errors",

    "performance_collapse_score",
    "failure_awareness_collapse_score",
    "silent_confidence_failure_score",
    "directional_bias_score",

    "dominant_signature",
    "dominant_signature_score",

    "secondary_signature",
    "secondary_signature_score",

    "dominance_margin",
]

with open(
    condition_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
        extrasaction="ignore"
    )

    writer.writeheader()
    writer.writerows(rows)


# --------------------------------------------------
# 7. Family-level mean signatures
# --------------------------------------------------

families = sorted(
    set(r["shift_family"] for r in rows)
)

family_rows = []

for family in families:

    fam = [
        r for r in rows
        if r["shift_family"] == family
    ]

    scores = {
        "performance_collapse":
            mean(
                r["performance_collapse_score"]
                for r in fam
            ),

        "failure_awareness_collapse":
            mean(
                r["failure_awareness_collapse_score"]
                for r in fam
            ),

        "silent_confidence_failure":
            mean(
                r["silent_confidence_failure_score"]
                for r in fam
            ),

        "directional_bias":
            mean(
                r["directional_bias_score"]
                for r in fam
            ),
    }

    ordered = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    family_rows.append({
        "shift_family": family,

        "mean_performance_collapse_score":
            scores["performance_collapse"],

        "mean_failure_awareness_collapse_score":
            scores["failure_awareness_collapse"],

        "mean_silent_confidence_failure_score":
            scores["silent_confidence_failure"],

        "mean_directional_bias_score":
            scores["directional_bias"],

        "dominant_family_signature":
            ordered[0][0],

        "dominant_family_signature_score":
            ordered[0][1],

        "secondary_family_signature":
            ordered[1][0],

        "secondary_family_signature_score":
            ordered[1][1],
    })


family_csv = (
    OUT /
    "failure_signature_family_summary.csv"
)

with open(
    family_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            family_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(family_rows)


# --------------------------------------------------
# 8. Severe-only signature comparison
# --------------------------------------------------

severe = [
    r for r in rows
    if r["severity"] == "severe"
]

severe_csv = (
    OUT /
    "failure_signature_severe_conditions.csv"
)

with open(
    severe_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
        extrasaction="ignore"
    )

    writer.writeheader()
    writer.writerows(severe)


# --------------------------------------------------
# 9. Human-readable exploratory report
# --------------------------------------------------

report = (
    OUT /
    "FAILURE_SIGNATURE_REPORT.md"
)

with open(
    report,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "# MedTrust-AI — Reliability Failure Signature Analysis v1\n\n"
    )

    f.write(
        "This is an exploratory validation-only analysis. "
        "It characterizes how different distribution shifts fail, "
        "rather than measuring only whether performance decreases.\n\n"
    )

    f.write(
        "**Important:** the four signature scores are descriptive "
        "study-internal constructs, not validated clinical metrics "
        "or statistical significance tests.\n\n"
    )

    f.write("## Signature dimensions\n\n")

    f.write(
        "- **Performance collapse:** accuracy, classification AUROC, "
        "and failure-rate deterioration.\n"
    )

    f.write(
        "- **Failure-awareness collapse:** deterioration in failure-detection "
        "AUROC/AURC plus newly wrong predictions that do not become "
        "more uncertain.\n"
    )

    f.write(
        "- **Silent-confidence failure:** wrong predictions that remain "
        "high-confidence, especially newly created errors.\n"
    )

    f.write(
        "- **Directional bias:** asymmetric change between false positives "
        "and false negatives.\n\n"
    )

    f.write("## Severe conditions\n\n")

    f.write(
        "| Shift | Performance | Awareness | Silent confidence | "
        "Directional bias | Dominant signature | Secondary |\n"
    )

    f.write(
        "|---|---:|---:|---:|---:|---|---|\n"
    )

    for r in severe:

        f.write(
            f"| {r['shift_family']} "
            f"| {r['performance_collapse_score']:.3f} "
            f"| {r['failure_awareness_collapse_score']:.3f} "
            f"| {r['silent_confidence_failure_score']:.3f} "
            f"| {r['directional_bias_score']:.3f} "
            f"| {r['dominant_signature']} "
            f"| {r['secondary_signature']} |\n"
        )

    f.write("\n## Family-level patterns\n\n")

    for r in family_rows:

        f.write(
            f"- **{r['shift_family']}**: dominant = "
            f"`{r['dominant_family_signature']}`, "
            f"secondary = "
            f"`{r['secondary_family_signature']}`.\n"
        )

    f.write("\n## Interpretation boundary\n\n")

    f.write(
        "The dominant label simply identifies the largest normalized "
        "descriptive score within this experiment. It does not imply that "
        "the shift belongs to a universally validated failure class. "
        "The purpose of this stage is hypothesis generation for a possible "
        "formal reliability-failure taxonomy or silent-failure metric.\n"
    )


# --------------------------------------------------
# 10. Metadata and hashes
# --------------------------------------------------

metadata = {
    "analysis_name":
        "failure_signature_v1",

    "analysis_type":
        "exploratory_descriptive",

    "source_files": {
        str(S1): sha256_file(S1),
        str(S2): sha256_file(S2),
        str(S3): sha256_file(S3),
    },

    "conditions": 15,

    "signature_dimensions": [
        "performance_collapse",
        "failure_awareness_collapse",
        "silent_confidence_failure",
        "directional_bias",
    ],

    "normalization":
        "within-study min-max across 15 validation shift conditions",

    "adverse_clipping":
        "negative degradation values clipped to zero before normalization",

    "validated_metric":
        False,

    "statistical_significance_test":
        False,

    "validation_only":
        True,

    "test_set_accessed":
        False,

    "model_rerun":
        False,

    "inference_rerun":
        False,

    "generated_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),
}

metadata_path = (
    OUT /
    "analysis_metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


targets = [
    condition_csv,
    family_csv,
    severe_csv,
    report,
    metadata_path,
]

checksums = (
    OUT /
    "SHA256SUMS.txt"
)

with open(
    checksums,
    "w",
    encoding="utf-8"
) as f:

    for p in targets:
        f.write(
            f"{sha256_file(p)}  {p.name}\n"
        )


# --------------------------------------------------
# 11. Terminal report
# --------------------------------------------------

print("=" * 78)
print(
    "MEDTRUST-AI — RELIABILITY FAILURE SIGNATURE ANALYSIS V1"
)
print("=" * 78)

print("\nSevere-shift signatures:")

for r in severe:

    print(
        f"\n{r['shift_family']}"
    )

    print(
        f"  Performance collapse:       "
        f"{r['performance_collapse_score']:.3f}"
    )

    print(
        f"  Failure-awareness collapse: "
        f"{r['failure_awareness_collapse_score']:.3f}"
    )

    print(
        f"  Silent-confidence failure:  "
        f"{r['silent_confidence_failure_score']:.3f}"
    )

    print(
        f"  Directional bias:           "
        f"{r['directional_bias_score']:.3f}"
    )

    print(
        f"  DOMINANT: {r['dominant_signature']}"
    )

    print(
        f"  SECONDARY: {r['secondary_signature']}"
    )


print("\nFamily-level dominant patterns:")

for r in family_rows:

    print(
        f"  {r['shift_family']:<24} "
        f"{r['dominant_family_signature']} "
        f"(secondary: {r['secondary_family_signature']})"
    )


print("\nOutputs:")

for p in targets + [checksums]:
    print(f"  {p}")


print("\nExploratory analysis: YES")
print("Validated new metric: NO")
print("Test set accessed: NO")
print("Model/inference rerun: NO")

print("=" * 78)
