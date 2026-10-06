"""Build and validate the frozen NIH Effusion vs No Finding cohort."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("data/derived/nih/nih_effusion_vs_nofinding_candidate_v0.1.csv")
MANIFEST = Path("manifests/nih_effusion_vs_nofinding_v1.csv")
SEED = 42
MAX_IMAGES_PER_PATIENT = 5
ENDPOINT = "Effusion_present_vs_No_Finding"
EXPECTED = {"train": 6192, "val": 1539, "test": 1932}
CHECKSUM_INPUTS = [
    Path("data/raw/nih/metadata/Data_Entry_2017_v2020.csv"),
    *[Path(f"data/raw/nih/splits/patient_level_{split}_list.txt")
      for split in EXPECTED],
]
REQUIRED = [
    "Image Index", "Patient ID", "Patient Age", "Patient Sex",
    "View Position", "Follow-up #", "Finding Labels", "binary_target",
    "split", "endpoint", "cohort_version", "sampling_seed",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    candidate = pd.read_csv(ROOT / SOURCE)
    assert set(REQUIRED).issubset(candidate.columns), "Missing required columns"
    assert candidate[REQUIRED].notna().all().all(), "Missing required values"
    assert set(candidate["split"]) == set(EXPECTED), "Unexpected splits"
    assert set(candidate["binary_target"]) == {0, 1}, "Unexpected targets"
    assert set(candidate["endpoint"]) == {ENDPOINT}, "Unexpected endpoint"
    assert not candidate["Image Index"].duplicated().any(), "Duplicate candidate images"
    assert candidate.groupby("Patient ID")["split"].nunique().max() == 1, (
        "Candidate patients overlap across splits"
    )
    for path in CHECKSUM_INPUTS:
        assert (ROOT / path).is_file(), f"Missing checksum input: {path}"

    selected = []
    for split in EXPECTED:
        rows = candidate[candidate["split"].eq(split)]
        positives = (
            rows[rows["binary_target"].eq(1)]
            .sample(frac=1, random_state=SEED)
            .groupby("Patient ID", sort=False).head(MAX_IMAGES_PER_PATIENT)
        )
        used = positives.groupby("Patient ID").size()
        negatives = rows[rows["binary_target"].eq(0)].sample(
            frac=1, random_state=SEED
        )
        capacity = MAX_IMAGES_PER_PATIENT - negatives["Patient ID"].map(used).fillna(0)
        eligible = negatives[
            negatives.groupby("Patient ID", sort=False).cumcount() < capacity
        ]
        assert len(eligible) >= len(positives), f"Insufficient negatives for {split}"
        negatives = eligible.sample(n=len(positives), random_state=SEED)
        selected.extend([positives, negatives])

    cohort = pd.concat(selected, ignore_index=True)
    cohort["cohort_version"] = "1.0"
    cohort["sampling_seed"] = SEED
    cohort["max_images_per_patient"] = MAX_IMAGES_PER_PATIENT
    cohort = cohort.sort_values(
        ["split", "Patient ID", "Image Index"], kind="stable"
    ).reset_index(drop=True)
    patients = {
        split: set(cohort.loc[cohort["split"].eq(split), "Patient ID"])
        for split in EXPECTED
    }
    split_counts = {}
    for split in EXPECTED:
        rows = cohort[cohort["split"].eq(split)]
        split_counts[split] = {
            "total_images": len(rows),
            "positive_images": int(rows["binary_target"].eq(1).sum()),
            "negative_images": int(rows["binary_target"].eq(0).sum()),
            "unique_patients": len(patients[split]),
        }
    summary = {
        "cohort_version": "1.0",
        "endpoint": ENDPOINT,
        "seed": SEED,
        "max_images_per_patient": MAX_IMAGES_PER_PATIENT,
        "total_images": len(cohort),
        "unique_patients": int(cohort["Patient ID"].nunique()),
        "splits": split_counts,
        "duplicate_image_count": int(cohort["Image Index"].duplicated().sum()),
        "maximum_images_contributed_by_one_patient": int(
            cohort.groupby("Patient ID").size().max()
        ),
        "train_val_patient_overlap": len(patients["train"] & patients["val"]),
        "train_test_patient_overlap": len(patients["train"] & patients["test"]),
        "val_test_patient_overlap": len(patients["val"] & patients["test"]),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_candidate_path": SOURCE.as_posix(),
    }
    print(json.dumps(summary, indent=2), flush=True)
    # All validation happens before any output file is written.
    assert summary["total_images"] == 19326, "Expected 19326 total images"
    assert summary["unique_patients"] == 10490, "Expected 10490 unique patients"
    for split, expected_positive in EXPECTED.items():
        counts = split_counts[split]
        assert counts["total_images"] == 2 * expected_positive, f"{split} total mismatch"
        assert counts["positive_images"] == expected_positive, f"{split} positive mismatch"
        assert counts["negative_images"] == expected_positive, f"{split} negative mismatch"
        assert counts["positive_images"] == counts["negative_images"], f"{split} imbalance"
    assert summary["duplicate_image_count"] == 0, "Duplicate images"
    for key in ("train_val_patient_overlap", "train_test_patient_overlap", "val_test_patient_overlap"):
        assert summary[key] == 0, f"Patient overlap: {key}"
    assert summary["maximum_images_contributed_by_one_patient"] <= 5, "Patient cap exceeded"
    assignments = candidate.set_index("Image Index")["split"]
    assert cohort["Image Index"].map(assignments).eq(cohort["split"]).all(), (
        "Original split assignments changed"
    )
    print("All assertions passed.", flush=True)

    (ROOT / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(ROOT / MANIFEST, index=False, lineterminator="\n")
    summary_path = MANIFEST.with_name("nih_effusion_vs_nofinding_v1_summary.json")
    (ROOT / summary_path).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    checksum_path = MANIFEST.with_name("nih_effusion_vs_nofinding_v1_checksums.txt")
    checksum_lines = [
        f"{sha256(ROOT / path)}  {path.as_posix()}"
        for path in [MANIFEST, *CHECKSUM_INPUTS]
    ]
    (ROOT / checksum_path).write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    for path in (MANIFEST, summary_path, checksum_path):
        print(f"Created: {path}")
    print("SHA256 checksums:")
    print("\n".join(checksum_lines))


if __name__ == "__main__":
    main()
