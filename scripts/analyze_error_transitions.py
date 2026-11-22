#!/usr/bin/env python3

import csv
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

INPUT = Path(
    "experiments/EXP-DN121-BASE-S42/"
    "shift_validation_deterministic_v1/"
    "shift_all_predictions.csv"
)

OUTPUT_DIR = Path(
    "experiments/EXP-DN121-BASE-S42/"
    "error_transition_v1"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def as_int(row, key):
    return int(float(row[key]))


# --------------------------------------------------
# 1. Load all predictions
# --------------------------------------------------

with open(INPUT, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if len(rows) != 49248:
    raise RuntimeError(
        f"Expected 49,248 rows, found {len(rows)}"
    )

conditions = sorted(set(r["condition_id"] for r in rows))

if len(conditions) != 16:
    raise RuntimeError(
        f"Expected 16 conditions, found {len(conditions)}"
    )


# --------------------------------------------------
# 2. Build clean reference
# --------------------------------------------------

clean_rows = [
    r for r in rows
    if r["condition_id"] == "clean"
]

if len(clean_rows) != 3078:
    raise RuntimeError(
        f"Expected 3,078 clean rows, found {len(clean_rows)}"
    )

clean_map = {}

for r in clean_rows:
    key = (
        r["Image Index"],
        r["Patient ID"],
    )

    if key in clean_map:
        raise RuntimeError(
            f"Duplicate clean key detected: {key}"
        )

    clean_map[key] = r


# --------------------------------------------------
# 3. Validate every shifted condition
# --------------------------------------------------

condition_groups = {}

for r in rows:
    if r["condition_id"] == "clean":
        continue

    condition_groups.setdefault(
        r["condition_id"], []
    ).append(r)


for condition_id, group in condition_groups.items():

    if len(group) != 3078:
        raise RuntimeError(
            f"{condition_id}: expected 3,078 rows, "
            f"found {len(group)}"
        )

    seen = set()

    for r in group:
        key = (
            r["Image Index"],
            r["Patient ID"],
        )

        if key in seen:
            raise RuntimeError(
                f"{condition_id}: duplicate key {key}"
            )

        seen.add(key)

        if key not in clean_map:
            raise RuntimeError(
                f"{condition_id}: key missing from clean: {key}"
            )

        clean = clean_map[key]

        if r["binary_target"] != clean["binary_target"]:
            raise RuntimeError(
                f"{condition_id}: target mismatch for {key}"
            )

    if seen != set(clean_map.keys()):
        raise RuntimeError(
            f"{condition_id}: image set does not match clean"
        )


# --------------------------------------------------
# 4. Clean confusion counts
# --------------------------------------------------

clean_tp = 0
clean_tn = 0
clean_fp = 0
clean_fn = 0

for r in clean_rows:

    y = as_int(r, "binary_target")
    pred = as_int(r, "predicted_class")

    if y == 1 and pred == 1:
        clean_tp += 1
    elif y == 0 and pred == 0:
        clean_tn += 1
    elif y == 0 and pred == 1:
        clean_fp += 1
    elif y == 1 and pred == 0:
        clean_fn += 1


# --------------------------------------------------
# 5. Per-condition transition analysis
# --------------------------------------------------

condition_summary = []
transition_rows = []

for condition_id, group in condition_groups.items():

    correct_to_wrong = 0
    wrong_to_correct = 0
    correct_to_correct = 0
    wrong_to_wrong = 0

    shifted_tp = 0
    shifted_tn = 0
    shifted_fp = 0
    shifted_fn = 0

    for shifted in group:

        key = (
            shifted["Image Index"],
            shifted["Patient ID"],
        )

        clean = clean_map[key]

        y = as_int(clean, "binary_target")

        clean_pred = as_int(
            clean,
            "predicted_class"
        )

        shift_pred = as_int(
            shifted,
            "predicted_class"
        )

        clean_correct = int(clean_pred == y)
        shift_correct = int(shift_pred == y)

        if clean_correct == 1 and shift_correct == 0:
            transition = "correct_to_wrong"
            correct_to_wrong += 1

        elif clean_correct == 0 and shift_correct == 1:
            transition = "wrong_to_correct"
            wrong_to_correct += 1

        elif clean_correct == 1 and shift_correct == 1:
            transition = "correct_to_correct"
            correct_to_correct += 1

        else:
            transition = "wrong_to_wrong"
            wrong_to_wrong += 1

        if y == 1 and shift_pred == 1:
            shifted_tp += 1
        elif y == 0 and shift_pred == 0:
            shifted_tn += 1
        elif y == 0 and shift_pred == 1:
            shifted_fp += 1
        elif y == 1 and shift_pred == 0:
            shifted_fn += 1

        transition_rows.append({
            "Image Index": shifted["Image Index"],
            "Patient ID": shifted["Patient ID"],
            "binary_target": y,

            "condition_id": condition_id,
            "shift_family": shifted["shift_family"],
            "severity": shifted["severity"],

            "clean_prediction": clean_pred,
            "shifted_prediction": shift_pred,

            "clean_failure": as_int(
                clean,
                "failure"
            ),

            "shifted_failure": as_int(
                shifted,
                "failure"
            ),

            "transition": transition,

            "clean_probability":
                float(clean["calibrated_probability"]),

            "shifted_probability":
                float(shifted["calibrated_probability"]),

            "probability_change":
                float(shifted["calibrated_probability"])
                - float(clean["calibrated_probability"]),
        })

    total = len(group)

    net_new_errors = (
        correct_to_wrong - wrong_to_correct
    )

    condition_summary.append({
        "condition_id": condition_id,
        "shift_family": group[0]["shift_family"],
        "severity": group[0]["severity"],
        "samples": total,

        "correct_to_wrong": correct_to_wrong,
        "wrong_to_correct": wrong_to_correct,
        "correct_to_correct": correct_to_correct,
        "wrong_to_wrong": wrong_to_wrong,

        "correct_to_wrong_rate":
            correct_to_wrong / total,

        "wrong_to_correct_rate":
            wrong_to_correct / total,

        "net_new_errors": net_new_errors,
        "net_new_error_rate":
            net_new_errors / total,

        "shifted_tp": shifted_tp,
        "shifted_tn": shifted_tn,
        "shifted_fp": shifted_fp,
        "shifted_fn": shifted_fn,

        "fp_change_vs_clean":
            shifted_fp - clean_fp,

        "fn_change_vs_clean":
            shifted_fn - clean_fn,
    })


severity_order = {
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}

condition_summary.sort(
    key=lambda r: (
        r["shift_family"],
        severity_order[r["severity"]]
    )
)


# --------------------------------------------------
# 6. Save per-condition summary
# --------------------------------------------------

summary_csv = (
    OUTPUT_DIR /
    "error_transition_summary_all_conditions.csv"
)

with open(
    summary_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            condition_summary[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(condition_summary)


# --------------------------------------------------
# 7. Save detailed transitions
# --------------------------------------------------

detail_csv = (
    OUTPUT_DIR /
    "error_transition_per_image.csv"
)

with open(
    detail_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            transition_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(transition_rows)


# --------------------------------------------------
# 8. Severe-only ranking
# --------------------------------------------------

severe_rows = [
    dict(r)
    for r in condition_summary
    if r["severity"] == "severe"
]

severe_rows.sort(
    key=lambda r: r["net_new_errors"],
    reverse=True
)

for rank, r in enumerate(
    severe_rows,
    start=1
):
    r["net_error_rank"] = rank


severe_csv = (
    OUTPUT_DIR /
    "error_transition_severe_ranking.csv"
)

fields = [
    "net_error_rank",
    "shift_family",
    "condition_id",
    "correct_to_wrong",
    "wrong_to_correct",
    "net_new_errors",
    "net_new_error_rate",
    "shifted_fp",
    "shifted_fn",
    "fp_change_vs_clean",
    "fn_change_vs_clean",
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
    writer.writerows(severe_rows)


# --------------------------------------------------
# 9. Human-readable report
# --------------------------------------------------

report = (
    OUTPUT_DIR /
    "error_transition_summary.md"
)

worst = severe_rows[0]

largest_fp = max(
    severe_rows,
    key=lambda r: r["fp_change_vs_clean"]
)

largest_fn = max(
    severe_rows,
    key=lambda r: r["fn_change_vs_clean"]
)


with open(
    report,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "# MedTrust-AI Error Transition Analysis v1\n\n"
    )

    f.write(
        "This analysis compares each shifted prediction "
        "against the same image's clean prediction. "
        "No model training or inference was rerun.\n\n"
    )

    f.write("## Clean confusion counts\n\n")

    f.write(
        f"- True positives: {clean_tp}\n"
        f"- True negatives: {clean_tn}\n"
        f"- False positives: {clean_fp}\n"
        f"- False negatives: {clean_fn}\n\n"
    )

    f.write("## Severe-shift transitions\n\n")

    f.write(
        "| Rank | Shift | Correct→Wrong | "
        "Wrong→Correct | Net new errors | "
        "FP change | FN change |\n"
    )

    f.write(
        "|---:|---|---:|---:|---:|---:|---:|\n"
    )

    for r in severe_rows:
        f.write(
            f"| {r['net_error_rank']} "
            f"| {r['shift_family']} "
            f"| {r['correct_to_wrong']} "
            f"| {r['wrong_to_correct']} "
            f"| {r['net_new_errors']} "
            f"| {r['fp_change_vs_clean']:+d} "
            f"| {r['fn_change_vs_clean']:+d} |\n"
        )

    f.write("\n## Main findings\n\n")

    f.write(
        f"- Largest net increase in errors: "
        f"**{worst['shift_family']}** "
        f"({worst['net_new_errors']} additional errors).\n"
    )

    f.write(
        f"- Largest false-positive increase: "
        f"**{largest_fp['shift_family']}** "
        f"({largest_fp['fp_change_vs_clean']:+d}).\n"
    )

    f.write(
        f"- Largest false-negative increase: "
        f"**{largest_fn['shift_family']}** "
        f"({largest_fn['fn_change_vs_clean']:+d}).\n"
    )

    f.write(
        "\nThese results are descriptive validation analyses "
        "and are not statistical significance tests.\n"
    )


# --------------------------------------------------
# 10. Metadata
# --------------------------------------------------

metadata = {
    "analysis_name": "error_transition_v1",
    "source_file": str(INPUT),
    "source_sha256": sha256_file(INPUT),

    "validation_images": 3078,
    "conditions_total": 16,
    "shift_conditions": 15,

    "clean_tp": clean_tp,
    "clean_tn": clean_tn,
    "clean_fp": clean_fp,
    "clean_fn": clean_fn,

    "test_set_accessed": False,
    "model_rerun": False,
    "inference_rerun": False,

    "analysis_generated_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),
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


# --------------------------------------------------
# 11. Checksums
# --------------------------------------------------

checksum_targets = [
    summary_csv,
    detail_csv,
    severe_csv,
    report,
    metadata_path,
]

checksums = (
    OUTPUT_DIR /
    "SHA256SUMS.txt"
)

with open(
    checksums,
    "w",
    encoding="utf-8"
) as f:

    for p in checksum_targets:
        f.write(
            f"{sha256_file(p)}  {p.name}\n"
        )


# --------------------------------------------------
# 12. Terminal summary
# --------------------------------------------------

print("=" * 72)
print(
    "MEDTRUST-AI — ERROR TRANSITION ANALYSIS V1"
)
print("=" * 72)

print("\nClean confusion matrix:")
print(f"  TP: {clean_tp}")
print(f"  TN: {clean_tn}")
print(f"  FP: {clean_fp}")
print(f"  FN: {clean_fn}")

print("\nSevere-shift transition ranking:")

for r in severe_rows:
    print(
        f"  {r['net_error_rank']}. "
        f"{r['shift_family']:<24} "
        f"C→W={r['correct_to_wrong']:>4}  "
        f"W→C={r['wrong_to_correct']:>4}  "
        f"net={r['net_new_errors']:>4}"
    )

print("\nLargest severe effects:")

print(
    "  Net new errors:      "
    f"{worst['shift_family']} "
    f"({worst['net_new_errors']:+d})"
)

print(
    "  FP increase:         "
    f"{largest_fp['shift_family']} "
    f"({largest_fp['fp_change_vs_clean']:+d})"
)

print(
    "  FN increase:         "
    f"{largest_fn['shift_family']} "
    f"({largest_fn['fn_change_vs_clean']:+d})"
)

print("\nOutputs:")
for p in [
    summary_csv,
    detail_csv,
    severe_csv,
    report,
    metadata_path,
    checksums,
]:
    print(f"  {p}")

print("\nTest set accessed: NO")
print("Model/inference rerun: NO")
print("=" * 72)
