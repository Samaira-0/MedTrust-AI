#!/usr/bin/env python3

import csv
import json
import hashlib
from pathlib import Path
from statistics import mean
from datetime import datetime, timezone

INPUT = Path(
    "experiments/EXP-DN121-BASE-S42/"
    "shift_summary_v1/shift_summary_all_conditions.csv"
)

OUTPUT_DIR = Path(
    "experiments/EXP-DN121-BASE-S42/"
    "shift_sensitivity_v1"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_FAMILIES = {
    "brightness",
    "contrast",
    "gaussian_noise",
    "gaussian_blur",
    "resolution_degradation",
}

SEVERITY_ORDER = {
    "clean": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}


def f(row, key):
    return float(row[key])


def pct_points(x):
    return x * 100.0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------
# 1. Load and validate
# --------------------------------------------------

with open(INPUT, newline="", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    rows = list(reader)

if len(rows) != 16:
    raise RuntimeError(
        f"Expected 16 total conditions, found {len(rows)}"
    )

clean_rows = [r for r in rows if r["condition_id"] == "clean"]

if len(clean_rows) != 1:
    raise RuntimeError(
        f"Expected exactly one clean condition, found {len(clean_rows)}"
    )

clean = clean_rows[0]

shift_rows = [r for r in rows if r["condition_id"] != "clean"]

families = {r["shift_family"] for r in shift_rows}

if families != EXPECTED_FAMILIES:
    raise RuntimeError(
        f"Unexpected shift families.\n"
        f"Expected: {sorted(EXPECTED_FAMILIES)}\n"
        f"Found: {sorted(families)}"
    )

for family in EXPECTED_FAMILIES:
    family_rows = [r for r in shift_rows if r["shift_family"] == family]
    severities = {r["severity"] for r in family_rows}

    if len(family_rows) != 3:
        raise RuntimeError(
            f"{family}: expected 3 severities, found {len(family_rows)}"
        )

    if severities != {"mild", "moderate", "severe"}:
        raise RuntimeError(
            f"{family}: unexpected severities {severities}"
        )


# --------------------------------------------------
# 2. Build clean-relative condition table
# --------------------------------------------------

condition_results = []

for r in shift_rows:

    accuracy_drop = f(clean, "accuracy") - f(r, "accuracy")
    auroc_drop = f(clean, "auroc") - f(r, "auroc")

    failure_rate_increase = (
        f(r, "failure_rate") - f(clean, "failure_rate")
    )

    fd_auroc_drop = (
        f(clean, "failure_detection_auroc")
        - f(r, "failure_detection_auroc")
    )

    aurc_increase = f(r, "aurc") - f(clean, "aurc")

    condition_results.append({
        "condition_id": r["condition_id"],
        "shift_family": r["shift_family"],
        "severity": r["severity"],

        "accuracy": f(r, "accuracy"),
        "auroc": f(r, "auroc"),
        "failure_rate": f(r, "failure_rate"),
        "failure_detection_auroc": f(
            r, "failure_detection_auroc"
        ),
        "aurc": f(r, "aurc"),

        "accuracy_drop": accuracy_drop,
        "accuracy_drop_percentage_points":
            pct_points(accuracy_drop),

        "auroc_drop": auroc_drop,

        "failure_rate_increase":
            failure_rate_increase,
        "failure_rate_increase_percentage_points":
            pct_points(failure_rate_increase),

        "fd_auroc_drop": fd_auroc_drop,
        "aurc_increase": aurc_increase,
    })


condition_results.sort(
    key=lambda r: (
        r["shift_family"],
        SEVERITY_ORDER[r["severity"]]
    )
)


condition_csv = OUTPUT_DIR / "shift_sensitivity_condition_deltas.csv"

condition_fields = list(condition_results[0].keys())

with open(condition_csv, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=condition_fields)
    writer.writeheader()
    writer.writerows(condition_results)


# --------------------------------------------------
# 3. Severe-condition comparison
# --------------------------------------------------

severe = [
    dict(r)
    for r in condition_results
    if r["severity"] == "severe"
]


def assign_damage_rank(records, metric):
    """
    Higher metric value = more damaging.
    Rank 1 = most damaging.
    """

    ordered = sorted(
        records,
        key=lambda r: r[metric],
        reverse=True
    )

    for rank, r in enumerate(ordered, start=1):
        for original in records:
            if original["shift_family"] == r["shift_family"]:
                original[f"{metric}_rank"] = rank
                break


damage_metrics = [
    "accuracy_drop",
    "auroc_drop",
    "failure_rate_increase",
    "fd_auroc_drop",
    "aurc_increase",
]

for metric in damage_metrics:
    assign_damage_rank(severe, metric)

for r in severe:
    ranks = [
        r[f"{metric}_rank"]
        for metric in damage_metrics
    ]

    # Lower mean rank = overall more damaging
    r["mean_damage_rank"] = mean(ranks)


severe.sort(key=lambda r: r["mean_damage_rank"])

for overall_rank, r in enumerate(severe, start=1):
    r["overall_severe_damage_rank"] = overall_rank


severe_csv = OUTPUT_DIR / "shift_sensitivity_severe_ranking.csv"

severe_fields = [
    "overall_severe_damage_rank",
    "shift_family",

    "accuracy",
    "accuracy_drop",
    "accuracy_drop_percentage_points",
    "accuracy_drop_rank",

    "auroc",
    "auroc_drop",
    "auroc_drop_rank",

    "failure_rate",
    "failure_rate_increase",
    "failure_rate_increase_percentage_points",
    "failure_rate_increase_rank",

    "failure_detection_auroc",
    "fd_auroc_drop",
    "fd_auroc_drop_rank",

    "aurc",
    "aurc_increase",
    "aurc_increase_rank",

    "mean_damage_rank",
]

with open(severe_csv, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(
        fh,
        fieldnames=severe_fields,
        extrasaction="ignore"
    )
    writer.writeheader()
    writer.writerows(severe)


# --------------------------------------------------
# 4. Mean effect across mild/moderate/severe
# --------------------------------------------------

family_summary = []

for family in sorted(EXPECTED_FAMILIES):

    fam = [
        r for r in condition_results
        if r["shift_family"] == family
    ]

    family_summary.append({
        "shift_family": family,
        "mean_accuracy_drop":
            mean(r["accuracy_drop"] for r in fam),

        "mean_auroc_drop":
            mean(r["auroc_drop"] for r in fam),

        "mean_failure_rate_increase":
            mean(r["failure_rate_increase"] for r in fam),

        "mean_fd_auroc_drop":
            mean(r["fd_auroc_drop"] for r in fam),

        "mean_aurc_increase":
            mean(r["aurc_increase"] for r in fam),
    })


family_csv = OUTPUT_DIR / "shift_sensitivity_family_means.csv"

with open(family_csv, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(
        fh,
        fieldnames=list(family_summary[0].keys())
    )
    writer.writeheader()
    writer.writerows(family_summary)


# --------------------------------------------------
# 5. Human-readable summary
# --------------------------------------------------

most_accuracy = max(
    severe,
    key=lambda r: r["accuracy_drop"]
)

most_auroc = max(
    severe,
    key=lambda r: r["auroc_drop"]
)

most_failure = max(
    severe,
    key=lambda r: r["failure_rate_increase"]
)

most_fd_damage = max(
    severe,
    key=lambda r: r["fd_auroc_drop"]
)

most_aurc = max(
    severe,
    key=lambda r: r["aurc_increase"]
)


summary_md = OUTPUT_DIR / "shift_sensitivity_summary.md"

with open(summary_md, "w", encoding="utf-8") as fh:

    fh.write("# MedTrust-AI Shift Sensitivity Analysis v1\n\n")

    fh.write(
        "This analysis uses the existing saved validation results. "
        "No model training or inference was rerun.\n\n"
    )

    fh.write("## Clean reference\n\n")
    fh.write(
        f"- Accuracy: {f(clean, 'accuracy'):.4f}\n"
        f"- AUROC: {f(clean, 'auroc'):.4f}\n"
        f"- Failure rate: {f(clean, 'failure_rate'):.4f}\n"
        f"- Failure-detection AUROC: "
        f"{f(clean, 'failure_detection_auroc'):.4f}\n"
        f"- AURC: {f(clean, 'aurc'):.4f}\n\n"
    )

    fh.write("## Severe-shift ranking\n\n")

    fh.write(
        "| Overall rank | Shift | Accuracy drop (pp) | "
        "AUROC drop | Failure increase (pp) | "
        "FD-AUROC drop | AURC increase |\n"
    )

    fh.write(
        "|---:|---|---:|---:|---:|---:|---:|\n"
    )

    for r in severe:
        fh.write(
            f"| {r['overall_severe_damage_rank']} "
            f"| {r['shift_family']} "
            f"| {r['accuracy_drop_percentage_points']:.2f} "
            f"| {r['auroc_drop']:.4f} "
            f"| {r['failure_rate_increase_percentage_points']:.2f} "
            f"| {r['fd_auroc_drop']:.4f} "
            f"| {r['aurc_increase']:.4f} |\n"
        )

    fh.write("\n## Metric-specific findings\n\n")

    fh.write(
        f"- Largest severe accuracy drop: "
        f"**{most_accuracy['shift_family']}** "
        f"({most_accuracy['accuracy_drop_percentage_points']:.2f} pp)\n"
    )

    fh.write(
        f"- Largest severe AUROC drop: "
        f"**{most_auroc['shift_family']}** "
        f"({most_auroc['auroc_drop']:.4f})\n"
    )

    fh.write(
        f"- Largest severe failure-rate increase: "
        f"**{most_failure['shift_family']}** "
        f"({most_failure['failure_rate_increase_percentage_points']:.2f} pp)\n"
    )

    fh.write(
        f"- Largest severe failure-detection AUROC drop: "
        f"**{most_fd_damage['shift_family']}** "
        f"({most_fd_damage['fd_auroc_drop']:.4f})\n"
    )

    fh.write(
        f"- Largest severe AURC increase: "
        f"**{most_aurc['shift_family']}** "
        f"({most_aurc['aurc_increase']:.4f})\n"
    )

    fh.write("\n## Interpretation\n\n")

    fh.write(
        "The ranking is descriptive and uses only the already-completed "
        "validation experiments. The overall severe-damage rank is the "
        "mean ordinal rank across five degradation measures: accuracy drop, "
        "classification AUROC drop, failure-rate increase, "
        "failure-detection AUROC drop, and AURC increase. "
        "It should not be interpreted as a new statistical significance test.\n"
    )


# --------------------------------------------------
# 6. Metadata + checksums
# --------------------------------------------------

metadata = {
    "analysis_name": "shift_sensitivity_v1",
    "source_file": str(INPUT),
    "source_sha256": sha256_file(INPUT),
    "conditions_total": len(rows),
    "shift_conditions": len(shift_rows),
    "shift_families": sorted(EXPECTED_FAMILIES),
    "severities": ["mild", "moderate", "severe"],
    "test_set_accessed": False,
    "model_rerun": False,
    "inference_rerun": False,
    "analysis_generated_utc":
        datetime.now(timezone.utc).isoformat(),
}

metadata_path = OUTPUT_DIR / "analysis_metadata.json"

with open(metadata_path, "w", encoding="utf-8") as fh:
    json.dump(metadata, fh, indent=2)


checksum_files = [
    condition_csv,
    severe_csv,
    family_csv,
    summary_md,
    metadata_path,
]

checksums_path = OUTPUT_DIR / "SHA256SUMS.txt"

with open(checksums_path, "w", encoding="utf-8") as fh:
    for path in checksum_files:
        fh.write(
            f"{sha256_file(path)}  {path.name}\n"
        )


# --------------------------------------------------
# 7. Terminal output
# --------------------------------------------------

print("=" * 72)
print("MEDTRUST-AI — SHIFT SENSITIVITY ANALYSIS V1")
print("=" * 72)

print("\nClean reference:")
print(f"  Accuracy:       {f(clean, 'accuracy'):.4f}")
print(f"  AUROC:          {f(clean, 'auroc'):.4f}")
print(f"  Failure rate:   {f(clean, 'failure_rate'):.4f}")
print(
    "  FD AUROC:       "
    f"{f(clean, 'failure_detection_auroc'):.4f}"
)
print(f"  AURC:           {f(clean, 'aurc'):.4f}")

print("\nSevere-shift overall ranking:")
for r in severe:
    print(
        f"  {r['overall_severe_damage_rank']}. "
        f"{r['shift_family']:<24} "
        f"mean rank={r['mean_damage_rank']:.2f}"
    )

print("\nLargest severe degradation by metric:")
print(
    "  Accuracy drop:      "
    f"{most_accuracy['shift_family']} "
    f"({most_accuracy['accuracy_drop_percentage_points']:.2f} pp)"
)
print(
    "  AUROC drop:         "
    f"{most_auroc['shift_family']} "
    f"({most_auroc['auroc_drop']:.4f})"
)
print(
    "  Failure increase:   "
    f"{most_failure['shift_family']} "
    f"({most_failure['failure_rate_increase_percentage_points']:.2f} pp)"
)
print(
    "  FD-AUROC drop:      "
    f"{most_fd_damage['shift_family']} "
    f"({most_fd_damage['fd_auroc_drop']:.4f})"
)
print(
    "  AURC increase:      "
    f"{most_aurc['shift_family']} "
    f"({most_aurc['aurc_increase']:.4f})"
)

print("\nOutputs:")
for p in [
    condition_csv,
    severe_csv,
    family_csv,
    summary_md,
    metadata_path,
    checksums_path,
]:
    print(f"  {p}")

print("\nTest set accessed: NO")
print("Model/inference rerun: NO")
print("=" * 72)
