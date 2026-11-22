#!/usr/bin/env python3

import csv
import json
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone

BASE = Path("experiments/EXP-DN121-BASE-S42")

INPUT = (
    BASE /
    "external_cohort_audit_v1" /
    "chexpert_eligible_candidates_v1.csv"
)

OUT = (
    BASE /
    "external_cohort_v1"
)

OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
MAX_IMAGES_PER_PATIENT = 5


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_key(*parts):
    text = "|".join(str(x) for x in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------
# 1. Load audited eligible candidates
# --------------------------------------------------

with open(INPUT, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise RuntimeError("Eligible candidate manifest is empty.")

required = {
    "Path",
    "patient_id",
    "orientation",
    "binary_target",
    "cohort_label",
}

missing = required - set(rows[0].keys())

if missing:
    raise RuntimeError(
        f"Missing required columns: {sorted(missing)}"
    )


# --------------------------------------------------
# 2. Safety checks
# --------------------------------------------------

paths = [r["Path"] for r in rows]

if len(paths) != len(set(paths)):
    raise RuntimeError(
        "Duplicate image paths detected in candidate manifest."
    )

if any(not r["patient_id"] for r in rows):
    raise RuntimeError(
        "Missing patient IDs detected."
    )

for r in rows:
    target = int(r["binary_target"])

    if target not in {0, 1}:
        raise RuntimeError(
            f"Unexpected binary target: {target}"
        )

    r["binary_target"] = target


# --------------------------------------------------
# 3. Apply GLOBAL patient cap <=5
#
# Important:
# Positive and negative images are pooled together
# before the cap is applied.
# --------------------------------------------------

by_patient = defaultdict(list)

for r in rows:
    by_patient[r["patient_id"]].append(r)

native_rows = []

for patient_id, patient_rows in by_patient.items():

    ordered = sorted(
        patient_rows,
        key=lambda r: stable_key(
            SEED,
            "global_patient_cap",
            patient_id,
            r["Path"]
        )
    )

    selected = ordered[:MAX_IMAGES_PER_PATIENT]

    for r in selected:
        rr = dict(r)
        rr["selection_seed"] = SEED
        rr["patient_cap"] = MAX_IMAGES_PER_PATIENT
        native_rows.append(rr)


native_rows = sorted(
    native_rows,
    key=lambda r: (
        r["patient_id"],
        stable_key(
            SEED,
            "native_output",
            r["Path"]
        )
    )
)


# --------------------------------------------------
# 4. Verify global patient cap
# --------------------------------------------------

native_pp = Counter(
    r["patient_id"]
    for r in native_rows
)

max_native_pp = max(native_pp.values())

if max_native_pp > MAX_IMAGES_PER_PATIENT:
    raise RuntimeError(
        f"Global patient cap violated: {max_native_pp}"
    )


# --------------------------------------------------
# 5. Native-prevalence cohort
# --------------------------------------------------

native_pos = [
    r for r in native_rows
    if r["binary_target"] == 1
]

native_neg = [
    r for r in native_rows
    if r["binary_target"] == 0
]


native_manifest = (
    OUT /
    "chexpert_native_prevalence_v1.csv"
)

native_fields = [
    "Path",
    "patient_id",
    "orientation",
    "binary_target",
    "cohort_label",
    "selection_seed",
    "patient_cap",
]

with open(
    native_manifest,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=native_fields
    )

    writer.writeheader()
    writer.writerows(native_rows)


# --------------------------------------------------
# 6. Balanced comparison cohort
#
# Built ONLY from globally capped native cohort.
# Therefore <=5 images/patient remains guaranteed.
# --------------------------------------------------

balanced_n_per_class = min(
    len(native_pos),
    len(native_neg)
)


def deterministic_class_sample(rows, target, label):

    ordered = sorted(
        rows,
        key=lambda r: stable_key(
            SEED,
            "balanced",
            label,
            r["patient_id"],
            r["Path"]
        )
    )

    return ordered[:target]


balanced_pos = deterministic_class_sample(
    native_pos,
    balanced_n_per_class,
    "positive"
)

balanced_neg = deterministic_class_sample(
    native_neg,
    balanced_n_per_class,
    "negative"
)

balanced_rows = balanced_pos + balanced_neg

balanced_rows = sorted(
    balanced_rows,
    key=lambda r: (
        r["patient_id"],
        stable_key(
            SEED,
            "balanced_output",
            r["Path"]
        )
    )
)


balanced_pp = Counter(
    r["patient_id"]
    for r in balanced_rows
)

max_balanced_pp = max(
    balanced_pp.values()
)

if max_balanced_pp > MAX_IMAGES_PER_PATIENT:
    raise RuntimeError(
        f"Balanced patient cap violated: {max_balanced_pp}"
    )


balanced_manifest = (
    OUT /
    "chexpert_balanced_comparison_v1.csv"
)

with open(
    balanced_manifest,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=native_fields
    )

    writer.writeheader()
    writer.writerows(balanced_rows)


# --------------------------------------------------
# 7. Patient-level summaries
# --------------------------------------------------

def cohort_stats(rows):

    pos = [
        r for r in rows
        if r["binary_target"] == 1
    ]

    neg = [
        r for r in rows
        if r["binary_target"] == 0
    ]

    pos_patients = {
        r["patient_id"]
        for r in pos
    }

    neg_patients = {
        r["patient_id"]
        for r in neg
    }

    all_patients = (
        pos_patients | neg_patients
    )

    both = (
        pos_patients & neg_patients
    )

    pp = Counter(
        r["patient_id"]
        for r in rows
    )

    return {
        "images_total":
            len(rows),

        "positive_images":
            len(pos),

        "negative_images":
            len(neg),

        "unique_patients":
            len(all_patients),

        "positive_patients":
            len(pos_patients),

        "negative_patients":
            len(neg_patients),

        "patients_contributing_to_both_classes":
            len(both),

        "maximum_images_from_any_patient":
            max(pp.values()) if pp else 0,

        "class_prevalence_positive":
            (
                len(pos) / len(rows)
                if rows else None
            ),
    }


native_stats = cohort_stats(
    native_rows
)

balanced_stats = cohort_stats(
    balanced_rows
)


# --------------------------------------------------
# 8. Metadata
# --------------------------------------------------

metadata = {
    "analysis_name":
        "chexpert_external_cohort_v1",

    "source_candidate_manifest":
        str(INPUT),

    "source_candidate_sha256":
        sha256_file(INPUT),

    "metadata_label_source":
        "train_cheXbert.csv",

    "task":
        "Pleural Effusion vs No Finding",

    "seed":
        SEED,

    "global_patient_cap":
        MAX_IMAGES_PER_PATIENT,

    "patient_cap_applied_before_balancing":
        True,

    "native_prevalence_cohort":
        native_stats,

    "balanced_comparison_cohort":
        balanced_stats,

    "balanced_target_per_class":
        balanced_n_per_class,

    "images_loaded":
        False,

    "model_inference_run":
        False,

    "nih_test_accessed":
        False,

    "generated_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),
}


metadata_path = (
    OUT /
    "cohort_metadata.json"
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

checksum_path = (
    OUT /
    "SHA256SUMS.txt"
)

with open(
    checksum_path,
    "w",
    encoding="utf-8"
) as f:

    for p in [
        native_manifest,
        balanced_manifest,
        metadata_path,
    ]:
        f.write(
            f"{sha256_file(p)}  {p.name}\n"
        )


# --------------------------------------------------
# 10. Terminal summary
# --------------------------------------------------

print("=" * 78)
print(
    "MEDTRUST-AI — CHEXPERT EXTERNAL COHORT BUILD V1"
)
print("=" * 78)

print(
    f"\nEligible audited candidates: "
    f"{len(rows):,}"
)

print("\nNative-prevalence cohort:")
print(
    f"  Total images:             "
    f"{native_stats['images_total']:,}"
)
print(
    f"  Pleural Effusion:         "
    f"{native_stats['positive_images']:,}"
)
print(
    f"  No Finding:               "
    f"{native_stats['negative_images']:,}"
)
print(
    f"  Positive prevalence:      "
    f"{native_stats['class_prevalence_positive']:.3f}"
)
print(
    f"  Unique patients:          "
    f"{native_stats['unique_patients']:,}"
)
print(
    f"  Patients in both classes: "
    f"{native_stats['patients_contributing_to_both_classes']:,}"
)
print(
    f"  Max images/patient:       "
    f"{native_stats['maximum_images_from_any_patient']}"
)

print("\nBalanced comparison cohort:")
print(
    f"  Total images:             "
    f"{balanced_stats['images_total']:,}"
)
print(
    f"  Pleural Effusion:         "
    f"{balanced_stats['positive_images']:,}"
)
print(
    f"  No Finding:               "
    f"{balanced_stats['negative_images']:,}"
)
print(
    f"  Unique patients:          "
    f"{balanced_stats['unique_patients']:,}"
)
print(
    f"  Patients in both classes: "
    f"{balanced_stats['patients_contributing_to_both_classes']:,}"
)
print(
    f"  Max images/patient:       "
    f"{balanced_stats['maximum_images_from_any_patient']}"
)

print("\nOutputs:")
print(f"  {native_manifest}")
print(f"  {balanced_manifest}")
print(f"  {metadata_path}")
print(f"  {checksum_path}")

print("\nImages loaded: NO")
print("Model inference run: NO")
print("NIH test accessed: NO")
print("=" * 78)
