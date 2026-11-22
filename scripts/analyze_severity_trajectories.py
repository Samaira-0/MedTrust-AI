#!/usr/bin/env python3

import csv
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

BASE = Path("experiments/EXP-DN121-BASE-S42")

INPUT = (
    BASE /
    "failure_signature_v1" /
    "failure_signature_all_conditions.csv"
)

OUT = (
    BASE /
    "severity_trajectory_v1"
)

OUT.mkdir(parents=True, exist_ok=True)

SEVERITY_ORDER = {
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}

SIGNATURES = [
    "performance_collapse",
    "failure_awareness_collapse",
    "silent_confidence_failure",
    "directional_bias",
]

SCORE_COLUMNS = {
    "performance_collapse":
        "performance_collapse_score",

    "failure_awareness_collapse":
        "failure_awareness_collapse_score",

    "silent_confidence_failure":
        "silent_confidence_failure_score",

    "directional_bias":
        "directional_bias_score",
}


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def classify_trajectory(
    mild,
    moderate,
    severe,
    tol=1e-9
):
    """
    Simple descriptive pattern classifier.
    """

    if (
        mild <= moderate + tol
        and moderate <= severe + tol
    ):
        if abs(severe - mild) <= tol:
            return "flat"

        return "monotonic_increase"

    if (
        mild >= moderate - tol
        and moderate >= severe - tol
    ):
        if abs(severe - mild) <= tol:
            return "flat"

        return "monotonic_decrease"

    if (
        moderate > mild
        and moderate > severe
    ):
        return "moderate_peak"

    if (
        moderate < mild
        and moderate < severe
    ):
        return "moderate_dip"

    if (
        severe > moderate
        and severe > mild
    ):
        return "late_severe_surge"

    return "mixed_non_monotonic"


# --------------------------------------------------
# 1. Load
# --------------------------------------------------

with open(
    INPUT,
    newline="",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)
    rows = list(reader)


if len(rows) != 15:
    raise RuntimeError(
        f"Expected 15 shift conditions, found {len(rows)}"
    )


families = sorted(
    set(r["shift_family"] for r in rows)
)

if len(families) != 5:
    raise RuntimeError(
        f"Expected 5 shift families, found {len(families)}"
    )


# --------------------------------------------------
# 2. Validate 3 severity levels per family
# --------------------------------------------------

for family in families:

    fam = [
        r for r in rows
        if r["shift_family"] == family
    ]

    if len(fam) != 3:
        raise RuntimeError(
            f"{family}: expected 3 conditions, found {len(fam)}"
        )

    severities = {
        r["severity"]
        for r in fam
    }

    if severities != {
        "mild",
        "moderate",
        "severe"
    }:
        raise RuntimeError(
            f"{family}: unexpected severities {severities}"
        )


# --------------------------------------------------
# 3. Analyze every family × signature trajectory
# --------------------------------------------------

trajectory_rows = []

family_summary_rows = []

for family in families:

    fam = sorted(
        [
            r for r in rows
            if r["shift_family"] == family
        ],
        key=lambda r:
            SEVERITY_ORDER[r["severity"]]
    )

    by_severity = {
        r["severity"]: r
        for r in fam
    }

    dominant_sequence = [
        by_severity["mild"]["dominant_signature"],
        by_severity["moderate"]["dominant_signature"],
        by_severity["severe"]["dominant_signature"],
    ]

    dominant_switches = sum(
        dominant_sequence[i]
        != dominant_sequence[i - 1]
        for i in range(1, 3)
    )

    increasing_count = 0
    decreasing_count = 0
    nonmonotonic_count = 0

    for signature in SIGNATURES:

        col = SCORE_COLUMNS[signature]

        mild = float(
            by_severity["mild"][col]
        )

        moderate = float(
            by_severity["moderate"][col]
        )

        severe = float(
            by_severity["severe"][col]
        )

        pattern = classify_trajectory(
            mild,
            moderate,
            severe
        )

        if pattern == "monotonic_increase":
            increasing_count += 1

        elif pattern == "monotonic_decrease":
            decreasing_count += 1

        else:
            nonmonotonic_count += 1

        trajectory_rows.append({
            "shift_family": family,
            "signature": signature,

            "mild_score": mild,
            "moderate_score": moderate,
            "severe_score": severe,

            "moderate_minus_mild":
                moderate - mild,

            "severe_minus_moderate":
                severe - moderate,

            "severe_minus_mild":
                severe - mild,

            "trajectory_pattern":
                pattern,

            "monotonic_increase":
                int(
                    pattern == "monotonic_increase"
                ),

            "monotonic_decrease":
                int(
                    pattern == "monotonic_decrease"
                ),
        })


    family_summary_rows.append({
        "shift_family":
            family,

        "mild_dominant_signature":
            dominant_sequence[0],

        "moderate_dominant_signature":
            dominant_sequence[1],

        "severe_dominant_signature":
            dominant_sequence[2],

        "dominant_signature_switch_count":
            dominant_switches,

        "monotonic_increasing_dimensions":
            increasing_count,

        "monotonic_decreasing_dimensions":
            decreasing_count,

        "nonmonotonic_dimensions":
            nonmonotonic_count,

        "stable_dominant_signature":
            int(
                dominant_switches == 0
            ),
    })


# --------------------------------------------------
# 4. Save trajectory table
# --------------------------------------------------

trajectory_csv = (
    OUT /
    "severity_trajectory_by_signature.csv"
)

with open(
    trajectory_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            trajectory_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        trajectory_rows
    )


# --------------------------------------------------
# 5. Save family-level summary
# --------------------------------------------------

family_csv = (
    OUT /
    "severity_trajectory_family_summary.csv"
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
            family_summary_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        family_summary_rows
    )


# --------------------------------------------------
# 6. Detect strongest severity-linked changes
# --------------------------------------------------

largest_risers = {}

for signature in SIGNATURES:

    candidates = [
        r
        for r in trajectory_rows
        if r["signature"] == signature
    ]

    largest_risers[signature] = max(
        candidates,
        key=lambda r:
            r["severe_minus_mild"]
    )


stable_families = [
    r
    for r in family_summary_rows
    if r["stable_dominant_signature"] == 1
]

switching_families = [
    r
    for r in family_summary_rows
    if r["stable_dominant_signature"] == 0
]


# --------------------------------------------------
# 7. Human-readable report
# --------------------------------------------------

report = (
    OUT /
    "SEVERITY_TRAJECTORY_REPORT.md"
)

with open(
    report,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "# MedTrust-AI — Severity Trajectory Analysis v1\n\n"
    )

    f.write(
        "This analysis evaluates whether reliability-failure "
        "signature dimensions evolve systematically from "
        "mild to moderate to severe shift.\n\n"
    )

    f.write(
        "**Important:** this remains an exploratory validation-only "
        "analysis. The signature scores are study-internal descriptive "
        "constructs, not validated clinical metrics.\n\n"
    )

    f.write(
        "## Dominant signature trajectory by shift family\n\n"
    )

    f.write(
        "| Shift family | Mild | Moderate | Severe | "
        "Signature switches |\n"
    )

    f.write(
        "|---|---|---|---|---:|\n"
    )

    for r in family_summary_rows:

        f.write(
            f"| {r['shift_family']} "
            f"| {r['mild_dominant_signature']} "
            f"| {r['moderate_dominant_signature']} "
            f"| {r['severe_dominant_signature']} "
            f"| {r['dominant_signature_switch_count']} |\n"
        )


    f.write(
        "\n## Signature-specific trajectories\n\n"
    )

    f.write(
        "| Shift | Signature | Mild | Moderate | Severe | Pattern |\n"
    )

    f.write(
        "|---|---|---:|---:|---:|---|\n"
    )

    for r in trajectory_rows:

        f.write(
            f"| {r['shift_family']} "
            f"| {r['signature']} "
            f"| {r['mild_score']:.3f} "
            f"| {r['moderate_score']:.3f} "
            f"| {r['severe_score']:.3f} "
            f"| {r['trajectory_pattern']} |\n"
        )


    f.write(
        "\n## Strongest mild-to-severe increases\n\n"
    )

    for signature, r in largest_risers.items():

        f.write(
            f"- **{signature}**: "
            f"{r['shift_family']} "
            f"(Δ={r['severe_minus_mild']:+.3f}).\n"
        )


    f.write(
        "\n## Dominant-signature stability\n\n"
    )

    if stable_families:

        f.write(
            "Families with the same dominant signature "
            "at mild, moderate, and severe levels:\n\n"
        )

        for r in stable_families:

            f.write(
                f"- {r['shift_family']}: "
                f"`{r['mild_dominant_signature']}`\n"
            )

    else:

        f.write(
            "No shift family retained the same dominant signature "
            "across all three severity levels.\n"
        )


    f.write("\n")

    if switching_families:

        f.write(
            "Families whose dominant signature changed with severity:\n\n"
        )

        for r in switching_families:

            f.write(
                f"- {r['shift_family']}: "
                f"{r['mild_dominant_signature']} → "
                f"{r['moderate_dominant_signature']} → "
                f"{r['severe_dominant_signature']}\n"
            )


    f.write(
        "\n## Interpretation boundary\n\n"
    )

    f.write(
        "A monotonic trajectory would support the hypothesis that a "
        "failure dimension systematically strengthens as degradation "
        "increases. A signature switch suggests that the failure mechanism "
        "may qualitatively change with severity. These observations are "
        "hypothesis-generating and require later statistical and external "
        "validation before being presented as a general taxonomy.\n"
    )


# --------------------------------------------------
# 8. Metadata
# --------------------------------------------------

metadata = {
    "analysis_name":
        "severity_trajectory_v1",

    "source_file":
        str(INPUT),

    "source_sha256":
        sha256_file(INPUT),

    "families":
        families,

    "severity_order": [
        "mild",
        "moderate",
        "severe"
    ],

    "signature_dimensions":
        SIGNATURES,

    "validated_metric":
        False,

    "analysis_type":
        "exploratory_descriptive",

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


# --------------------------------------------------
# 9. Checksums
# --------------------------------------------------

targets = [
    trajectory_csv,
    family_csv,
    report,
    metadata_path,
]

checksum_path = (
    OUT /
    "SHA256SUMS.txt"
)

with open(
    checksum_path,
    "w",
    encoding="utf-8"
) as f:

    for p in targets:

        f.write(
            f"{sha256_file(p)}  {p.name}\n"
        )


# --------------------------------------------------
# 10. Terminal summary
# --------------------------------------------------

print("=" * 78)
print(
    "MEDTRUST-AI — SEVERITY TRAJECTORY ANALYSIS V1"
)
print("=" * 78)

print(
    "\nDominant signature trajectories:"
)

for r in family_summary_rows:

    print(
        f"  {r['shift_family']:<24} "
        f"{r['mild_dominant_signature']} -> "
        f"{r['moderate_dominant_signature']} -> "
        f"{r['severe_dominant_signature']} "
        f"(switches={r['dominant_signature_switch_count']})"
    )


print(
    "\nMonotonic increasing dimensions:"
)

for r in family_summary_rows:

    print(
        f"  {r['shift_family']:<24} "
        f"{r['monotonic_increasing_dimensions']}/4"
    )


print(
    "\nStrongest mild-to-severe increase by signature:"
)

for signature in SIGNATURES:

    r = largest_risers[signature]

    print(
        f"  {signature:<30} "
        f"{r['shift_family']} "
        f"Δ={r['severe_minus_mild']:+.3f}"
    )


print("\nOutputs:")

for p in targets + [checksum_path]:
    print(f"  {p}")


print("\nExploratory analysis: YES")
print("Validated taxonomy: NO")
print("Test set accessed: NO")
print("Model/inference rerun: NO")

print("=" * 78)
