#!/usr/bin/env python3

import csv
import json
import math
import hashlib
from pathlib import Path
from statistics import mean, median
from datetime import datetime, timezone

INPUT = Path(
    "experiments/EXP-DN121-BASE-S42/"
    "error_transition_v1/"
    "error_transition_per_image.csv"
)

OUTPUT_DIR = Path(
    "experiments/EXP-DN121-BASE-S42/"
    "confidence_transition_v1"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def confidence(p):
    return max(p, 1.0 - p)


def uncertainty(p):
    return min(p, 1.0 - p)


def entropy(p):
    eps = 1e-12
    p = min(max(p, eps), 1.0 - eps)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


# --------------------------------------------------
# 1. Load
# --------------------------------------------------

with open(INPUT, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if len(rows) != 46170:
    raise RuntimeError(
        f"Expected 46,170 rows, found {len(rows)}"
    )


# --------------------------------------------------
# 2. Derive confidence / uncertainty / entropy
# --------------------------------------------------

derived = []

for r in rows:

    cp = float(r["clean_probability"])
    sp = float(r["shifted_probability"])

    clean_conf = confidence(cp)
    shift_conf = confidence(sp)

    clean_unc = uncertainty(cp)
    shift_unc = uncertainty(sp)

    clean_ent = entropy(cp)
    shift_ent = entropy(sp)

    derived.append({
        **r,

        "clean_confidence": clean_conf,
        "shifted_confidence": shift_conf,
        "confidence_change": shift_conf - clean_conf,

        "clean_uncertainty": clean_unc,
        "shifted_uncertainty": shift_unc,
        "uncertainty_change": shift_unc - clean_unc,

        "clean_entropy": clean_ent,
        "shifted_entropy": shift_ent,
        "entropy_change": shift_ent - clean_ent,

        "shifted_wrong_high_conf_80":
            int(
                r["shifted_failure"] == "1"
                and shift_conf >= 0.80
            ),

        "shifted_wrong_high_conf_90":
            int(
                r["shifted_failure"] == "1"
                and shift_conf >= 0.90
            ),
    })


# --------------------------------------------------
# 3. Group by condition
# --------------------------------------------------

groups = {}

for r in derived:
    groups.setdefault(
        r["condition_id"], []
    ).append(r)


summary = []

for condition_id, group in groups.items():

    c2w = [
        r for r in group
        if r["transition"] == "correct_to_wrong"
    ]

    shifted_wrong = [
        r for r in group
        if r["shifted_failure"] == "1"
    ]

    if c2w:
        c2w_mean_conf_change = mean(
            float(r["confidence_change"])
            for r in c2w
        )

        c2w_mean_unc_change = mean(
            float(r["uncertainty_change"])
            for r in c2w
        )

        c2w_mean_entropy_change = mean(
            float(r["entropy_change"])
            for r in c2w
        )

        c2w_more_uncertain = sum(
            float(r["uncertainty_change"]) > 0
            for r in c2w
        )

        c2w_more_uncertain_rate = (
            c2w_more_uncertain / len(c2w)
        )

        c2w_high_conf_80 = sum(
            float(r["shifted_confidence"]) >= 0.80
            for r in c2w
        )

        c2w_high_conf_90 = sum(
            float(r["shifted_confidence"]) >= 0.90
            for r in c2w
        )

    else:
        c2w_mean_conf_change = 0.0
        c2w_mean_unc_change = 0.0
        c2w_mean_entropy_change = 0.0
        c2w_more_uncertain = 0
        c2w_more_uncertain_rate = 0.0
        c2w_high_conf_80 = 0
        c2w_high_conf_90 = 0

    wrong_high_conf_80 = sum(
        int(r["shifted_wrong_high_conf_80"])
        for r in shifted_wrong
    )

    wrong_high_conf_90 = sum(
        int(r["shifted_wrong_high_conf_90"])
        for r in shifted_wrong
    )

    summary.append({
        "condition_id": condition_id,
        "shift_family": group[0]["shift_family"],
        "severity": group[0]["severity"],

        "correct_to_wrong_count": len(c2w),

        "c2w_mean_confidence_change":
            c2w_mean_conf_change,

        "c2w_mean_uncertainty_change":
            c2w_mean_unc_change,

        "c2w_mean_entropy_change":
            c2w_mean_entropy_change,

        "c2w_more_uncertain_count":
            c2w_more_uncertain,

        "c2w_more_uncertain_rate":
            c2w_more_uncertain_rate,

        "c2w_high_confidence_80_count":
            c2w_high_conf_80,

        "c2w_high_confidence_90_count":
            c2w_high_conf_90,

        "shifted_wrong_count":
            len(shifted_wrong),

        "all_wrong_high_confidence_80_count":
            wrong_high_conf_80,

        "all_wrong_high_confidence_80_rate":
            (
                wrong_high_conf_80 / len(shifted_wrong)
                if shifted_wrong else 0.0
            ),

        "all_wrong_high_confidence_90_count":
            wrong_high_conf_90,

        "all_wrong_high_confidence_90_rate":
            (
                wrong_high_conf_90 / len(shifted_wrong)
                if shifted_wrong else 0.0
            ),
    })


severity_order = {
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}

summary.sort(
    key=lambda r: (
        r["shift_family"],
        severity_order[r["severity"]]
    )
)


# --------------------------------------------------
# 4. Save condition summary
# --------------------------------------------------

summary_csv = (
    OUTPUT_DIR /
    "confidence_transition_summary_all_conditions.csv"
)

with open(
    summary_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(summary[0].keys())
    )
    writer.writeheader()
    writer.writerows(summary)


# --------------------------------------------------
# 5. Severe-only table
# --------------------------------------------------

severe = [
    dict(r)
    for r in summary
    if r["severity"] == "severe"
]

severe.sort(
    key=lambda r:
        r["all_wrong_high_confidence_90_count"],
    reverse=True
)

for rank, r in enumerate(severe, start=1):
    r["high_conf_wrong_rank"] = rank


severe_csv = (
    OUTPUT_DIR /
    "confidence_transition_severe_summary.csv"
)

fields = [
    "high_conf_wrong_rank",
    "shift_family",
    "condition_id",

    "correct_to_wrong_count",
    "c2w_mean_confidence_change",
    "c2w_mean_uncertainty_change",
    "c2w_mean_entropy_change",
    "c2w_more_uncertain_count",
    "c2w_more_uncertain_rate",

    "c2w_high_confidence_80_count",
    "c2w_high_confidence_90_count",

    "shifted_wrong_count",

    "all_wrong_high_confidence_80_count",
    "all_wrong_high_confidence_80_rate",

    "all_wrong_high_confidence_90_count",
    "all_wrong_high_confidence_90_rate",
]

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
# 6. High-confidence error examples
# --------------------------------------------------

high_conf_examples = [
    r for r in derived
    if (
        r["severity"] == "severe"
        and r["shifted_failure"] == "1"
        and float(r["shifted_confidence"]) >= 0.90
    )
]

high_conf_examples.sort(
    key=lambda r: float(r["shifted_confidence"]),
    reverse=True
)

examples_csv = (
    OUTPUT_DIR /
    "severe_high_confidence_errors.csv"
)

example_fields = [
    "Image Index",
    "Patient ID",
    "binary_target",
    "condition_id",
    "shift_family",
    "severity",
    "clean_prediction",
    "shifted_prediction",
    "transition",
    "clean_probability",
    "shifted_probability",
    "clean_confidence",
    "shifted_confidence",
    "confidence_change",
    "clean_uncertainty",
    "shifted_uncertainty",
    "uncertainty_change",
    "clean_entropy",
    "shifted_entropy",
    "entropy_change",
]

with open(
    examples_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=example_fields,
        extrasaction="ignore"
    )
    writer.writeheader()
    writer.writerows(high_conf_examples)


# --------------------------------------------------
# 7. Human-readable report
# --------------------------------------------------

most_hc = max(
    severe,
    key=lambda r:
        r["all_wrong_high_confidence_90_count"]
)

most_c2w_hc = max(
    severe,
    key=lambda r:
        r["c2w_high_confidence_90_count"]
)

least_uncertainty_response = min(
    severe,
    key=lambda r:
        r["c2w_more_uncertain_rate"]
)


report = (
    OUTPUT_DIR /
    "confidence_transition_summary.md"
)

with open(
    report,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "# MedTrust-AI Confidence Transition Analysis v1\n\n"
    )

    f.write(
        "This analysis uses saved validation predictions only. "
        "No model training or inference was rerun.\n\n"
    )

    f.write(
        "High-confidence error thresholds are reported at "
        "confidence >= 0.80 and >= 0.90.\n\n"
    )

    f.write("## Severe shifts\n\n")

    f.write(
        "| Shift | C→W | C→W more uncertain | "
        "C→W conf≥0.90 | All wrong conf≥0.90 |\n"
    )

    f.write(
        "|---|---:|---:|---:|---:|\n"
    )

    for r in severe:
        f.write(
            f"| {r['shift_family']} "
            f"| {r['correct_to_wrong_count']} "
            f"| {r['c2w_more_uncertain_rate']:.1%} "
            f"| {r['c2w_high_confidence_90_count']} "
            f"| {r['all_wrong_high_confidence_90_count']} |\n"
        )

    f.write("\n## Main findings\n\n")

    f.write(
        f"- Most severe high-confidence wrong predictions: "
        f"**{most_hc['shift_family']}** "
        f"({most_hc['all_wrong_high_confidence_90_count']}).\n"
    )

    f.write(
        f"- Most newly-created high-confidence errors: "
        f"**{most_c2w_hc['shift_family']}** "
        f"({most_c2w_hc['c2w_high_confidence_90_count']}).\n"
    )

    f.write(
        f"- Lowest proportion of correct→wrong cases that "
        f"became more uncertain: "
        f"**{least_uncertainty_response['shift_family']}** "
        f"({least_uncertainty_response['c2w_more_uncertain_rate']:.1%}).\n"
    )

    f.write(
        "\nInterpretation should remain descriptive. "
        "These are validation-development findings and not "
        "statistical significance tests.\n"
    )


# --------------------------------------------------
# 8. Metadata + checksums
# --------------------------------------------------

metadata = {
    "analysis_name": "confidence_transition_v1",
    "source_file": str(INPUT),
    "source_sha256": sha256_file(INPUT),

    "high_confidence_thresholds": [
        0.80,
        0.90
    ],

    "validation_only": True,
    "test_set_accessed": False,
    "model_rerun": False,
    "inference_rerun": False,

    "analysis_generated_utc":
        datetime.now(timezone.utc).isoformat(),
}

metadata_path = (
    OUTPUT_DIR /
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
    summary_csv,
    severe_csv,
    examples_csv,
    report,
    metadata_path,
]

checksums_path = (
    OUTPUT_DIR /
    "SHA256SUMS.txt"
)

with open(
    checksums_path,
    "w",
    encoding="utf-8"
) as f:
    for p in targets:
        f.write(
            f"{sha256_file(p)}  {p.name}\n"
        )


# --------------------------------------------------
# 9. Terminal summary
# --------------------------------------------------

print("=" * 72)
print(
    "MEDTRUST-AI — CONFIDENCE TRANSITION ANALYSIS V1"
)
print("=" * 72)

print("\nSevere-shift confidence behavior:")

for r in severe:
    print(
        f"  {r['shift_family']:<24} "
        f"C→W={r['correct_to_wrong_count']:>4} | "
        f"more uncertain="
        f"{r['c2w_more_uncertain_rate']*100:>5.1f}% | "
        f"C→W conf>=0.90="
        f"{r['c2w_high_confidence_90_count']:>4} | "
        f"all wrong conf>=0.90="
        f"{r['all_wrong_high_confidence_90_count']:>4}"
    )

print("\nLargest severe effects:")

print(
    "  Most high-confidence wrong predictions: "
    f"{most_hc['shift_family']} "
    f"({most_hc['all_wrong_high_confidence_90_count']})"
)

print(
    "  Most new high-confidence errors:        "
    f"{most_c2w_hc['shift_family']} "
    f"({most_c2w_hc['c2w_high_confidence_90_count']})"
)

print(
    "  Weakest uncertainty response:           "
    f"{least_uncertainty_response['shift_family']} "
    f"({least_uncertainty_response['c2w_more_uncertain_rate']*100:.1f}% "
    f"of C→W became more uncertain)"
)

print("\nOutputs:")
for p in [
    summary_csv,
    severe_csv,
    examples_csv,
    report,
    metadata_path,
    checksums_path,
]:
    print(f"  {p}")

print("\nTest set accessed: NO")
print("Model/inference rerun: NO")
print("=" * 72)
