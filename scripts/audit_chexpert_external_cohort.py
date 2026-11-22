#!/usr/bin/env python3

import csv
import json
import re
import sys
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone

OUT = Path(
    "experiments/EXP-DN121-BASE-S42/"
    "external_cohort_audit_v1"
)
OUT.mkdir(parents=True, exist_ok=True)

REQUIRED = {
    "Path",
    "Frontal/Lateral",
    "AP/PA",
    "Pleural Effusion",
    "No Finding",
}

PATIENT_RE = re.compile(r"(patient\d+)", re.I)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_label(x):
    if x is None:
        return None

    x = str(x).strip()

    if x == "":
        return None

    try:
        return float(x)
    except ValueError:
        return None


def patient_id_from_path(path):
    m = PATIENT_RE.search(path)
    return m.group(1).lower() if m else None


def locate_csv():
    # Optional direct argument:
    # python3 scripts/audit_chexpert_external_cohort.py /path/to/train.csv
    if len(sys.argv) > 1:
        p = Path(sys.argv[1]).expanduser()
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(
            f"Provided CSV does not exist: {p}"
        )

    candidates = [
        Path("data/chexpert/train.csv"),
        Path("data/CheXpert-v1.0-small/train.csv"),
        Path("data/CheXpert-v1.0/train.csv"),
        Path("datasets/chexpert/train.csv"),
        Path("datasets/CheXpert-v1.0-small/train.csv"),
        Path("datasets/CheXpert-v1.0/train.csv"),
        Path.home() / "data/CheXpert-v1.0-small/train.csv",
        Path.home() / "datasets/CheXpert-v1.0-small/train.csv",
    ]

    for p in candidates:
        if p.exists():
            return p.resolve()

    return None


csv_path = locate_csv()

if csv_path is None:
    print("=" * 76)
    print("MEDTRUST-AI — CHEXPERT EXTERNAL COHORT AUDIT V1")
    print("=" * 76)
    print()
    print("CheXpert train.csv was NOT found automatically.")
    print()
    print("No data were modified.")
    print("No images were loaded.")
    print("No model was run.")
    print()
    print("When train.csv is available, run:")
    print()
    print(
        "python3 scripts/audit_chexpert_external_cohort.py "
        "/FULL/PATH/TO/train.csv"
    )
    print("=" * 76)
    sys.exit(2)


with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    missing = REQUIRED - set(reader.fieldnames or [])

    if missing:
        raise RuntimeError(
            f"Missing required CheXpert columns: {sorted(missing)}\n"
            f"Found columns: {reader.fieldnames}"
        )

    rows = list(reader)


total_rows = len(rows)

view_counts = Counter()
orientation_counts = Counter()

pleural_counts = Counter()
no_finding_counts = Counter()

frontal_rows = []

missing_patient_id = 0

for row in rows:

    frontal = str(
        row.get("Frontal/Lateral", "")
    ).strip()

    orientation = str(
        row.get("AP/PA", "")
    ).strip()

    view_counts[frontal] += 1
    orientation_counts[orientation] += 1

    pe = parse_label(
        row.get("Pleural Effusion")
    )

    nf = parse_label(
        row.get("No Finding")
    )

    pleural_counts[str(pe)] += 1
    no_finding_counts[str(nf)] += 1

    pid = patient_id_from_path(
        row["Path"]
    )

    if pid is None:
        missing_patient_id += 1

    is_frontal = (
        frontal.lower() == "frontal"
        and orientation.upper() in {"AP", "PA"}
    )

    if is_frontal:
        frontal_rows.append({
            "Path": row["Path"],
            "patient_id": pid,
            "orientation": orientation.upper(),
            "pleural_effusion": pe,
            "no_finding": nf,
        })


# --------------------------------------------------
# Label consistency checks
# --------------------------------------------------

contradiction_positive_nf = []
contradiction_uncertain_nf = []

eligible_positive = []
eligible_negative = []

uncertain_effusion = []
other_excluded = []

for row in frontal_rows:

    pe = row["pleural_effusion"]
    nf = row["no_finding"]

    # Logical contradictions involving "No Finding"
    if nf == 1.0 and pe == 1.0:
        contradiction_positive_nf.append(row)
        continue

    if nf == 1.0 and pe == -1.0:
        contradiction_uncertain_nf.append(row)
        continue

    # Primary positive cohort
    if pe == 1.0:
        eligible_positive.append(row)
        continue

    # Explicit uncertainty excluded
    if pe == -1.0:
        uncertain_effusion.append(row)
        continue

    # Primary negative cohort
    if nf == 1.0 and pe in {0.0, None}:
        eligible_negative.append(row)
        continue

    other_excluded.append(row)


# --------------------------------------------------
# Patient-level summaries
# --------------------------------------------------

positive_patients = {
    r["patient_id"]
    for r in eligible_positive
    if r["patient_id"] is not None
}

negative_patients = {
    r["patient_id"]
    for r in eligible_negative
    if r["patient_id"] is not None
}

all_eligible_patients = (
    positive_patients | negative_patients
)

patients_in_both_classes = (
    positive_patients & negative_patients
)


def per_patient_counts(rows):
    d = defaultdict(int)

    for r in rows:
        if r["patient_id"] is not None:
            d[r["patient_id"]] += 1

    return d


pos_pp = per_patient_counts(
    eligible_positive
)

neg_pp = per_patient_counts(
    eligible_negative
)

combined_pp = per_patient_counts(
    eligible_positive + eligible_negative
)


def capped_total(counts, cap=5):
    return sum(
        min(v, cap)
        for v in counts.values()
    )


positive_after_individual_cap = capped_total(
    pos_pp, 5
)

negative_after_individual_cap = capped_total(
    neg_pp, 5
)

combined_after_global_cap = capped_total(
    combined_pp, 5
)


balanced_theoretical_per_class = min(
    positive_after_individual_cap,
    negative_after_individual_cap,
)


# --------------------------------------------------
# Save compact candidate manifest
# --------------------------------------------------

candidate_manifest = (
    OUT /
    "chexpert_eligible_candidates_v1.csv"
)

with open(
    candidate_manifest,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fields = [
        "Path",
        "patient_id",
        "orientation",
        "binary_target",
        "cohort_label",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()

    for row in eligible_positive:
        writer.writerow({
            "Path": row["Path"],
            "patient_id": row["patient_id"],
            "orientation": row["orientation"],
            "binary_target": 1,
            "cohort_label": "pleural_effusion",
        })

    for row in eligible_negative:
        writer.writerow({
            "Path": row["Path"],
            "patient_id": row["patient_id"],
            "orientation": row["orientation"],
            "binary_target": 0,
            "cohort_label": "no_finding",
        })


summary = {
    "analysis_name":
        "chexpert_external_cohort_audit_v1",

    "source_csv":
        str(csv_path),

    "source_csv_sha256":
        sha256_file(csv_path),

    "total_rows":
        total_rows,

    "frontal_ap_pa_rows":
        len(frontal_rows),

    "view_counts":
        dict(view_counts),

    "orientation_counts":
        dict(orientation_counts),

    "eligible_positive_images":
        len(eligible_positive),

    "eligible_negative_images":
        len(eligible_negative),

    "eligible_positive_patients":
        len(positive_patients),

    "eligible_negative_patients":
        len(negative_patients),

    "eligible_unique_patients":
        len(all_eligible_patients),

    "patients_with_both_positive_and_negative_images":
        len(patients_in_both_classes),

    "uncertain_effusion_frontal_rows":
        len(uncertain_effusion),

    "contradictory_no_finding_plus_positive_effusion":
        len(contradiction_positive_nf),

    "contradictory_no_finding_plus_uncertain_effusion":
        len(contradiction_uncertain_nf),

    "other_frontal_rows_excluded":
        len(other_excluded),

    "missing_patient_id_from_path":
        missing_patient_id,

    "positive_images_after_per_class_patient_cap_5":
        positive_after_individual_cap,

    "negative_images_after_per_class_patient_cap_5":
        negative_after_individual_cap,

    "eligible_images_after_global_patient_cap_5":
        combined_after_global_cap,

    "balanced_theoretical_max_per_class":
        balanced_theoretical_per_class,

    "balanced_theoretical_total":
        balanced_theoretical_per_class * 2,

    "pleural_effusion_raw_label_counts":
        dict(pleural_counts),

    "no_finding_raw_label_counts":
        dict(no_finding_counts),

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


summary_json = (
    OUT /
    "chexpert_cohort_audit_summary.json"
)

with open(
    summary_json,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


checksums = (
    OUT /
    "SHA256SUMS.txt"
)

with open(
    checksums,
    "w",
    encoding="utf-8"
) as f:

    for p in [
        candidate_manifest,
        summary_json,
    ]:
        f.write(
            f"{sha256_file(p)}  {p.name}\n"
        )


# --------------------------------------------------
# Terminal summary
# --------------------------------------------------

print("=" * 76)
print("MEDTRUST-AI — CHEXPERT EXTERNAL COHORT AUDIT V1")
print("=" * 76)

print(f"\nCSV: {csv_path}")
print(f"Total metadata rows:        {total_rows:,}")
print(f"Frontal AP/PA rows:         {len(frontal_rows):,}")

print("\nEligible primary task:")
print(
    f"  Pleural Effusion images:  {len(eligible_positive):,} "
    f"({len(positive_patients):,} patients)"
)
print(
    f"  No Finding images:        {len(eligible_negative):,} "
    f"({len(negative_patients):,} patients)"
)
print(
    f"  Unique eligible patients: {len(all_eligible_patients):,}"
)

print("\nImportant exclusions / checks:")
print(
    f"  Effusion uncertain:       {len(uncertain_effusion):,}"
)
print(
    f"  NoFinding + Effusion=1:   {len(contradiction_positive_nf):,}"
)
print(
    f"  NoFinding + Effusion=-1:  {len(contradiction_uncertain_nf):,}"
)
print(
    f"  Missing patient IDs:      {missing_patient_id:,}"
)
print(
    f"  Patients in both classes: {len(patients_in_both_classes):,}"
)

print("\nPatient-cap feasibility:")
print(
    f"  Positive after cap<=5:    "
    f"{positive_after_individual_cap:,}"
)
print(
    f"  Negative after cap<=5:    "
    f"{negative_after_individual_cap:,}"
)
print(
    f"  Balanced theoretical max: "
    f"{balanced_theoretical_per_class:,}/class "
    f"({balanced_theoretical_per_class * 2:,} total)"
)

print("\nOutputs:")
print(f"  {candidate_manifest}")
print(f"  {summary_json}")
print(f"  {checksums}")

print("\nImages loaded: NO")
print("Model inference run: NO")
print("NIH test accessed: NO")
print("=" * 76)
