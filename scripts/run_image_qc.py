"""Flag, describe, and visualize the frozen cohort; never exclude images."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/nih_effusion_vs_nofinding_v1.csv"
IMAGES = ROOT / "data/processed/nih/images_224x224"
REPORTS = ROOT / "reports"
EXPECTED = 19326
SEED = 42
METADATA = ["Image Index", "Patient ID", "split", "binary_target",
            "View Position", "Patient Sex", "Patient Age"]
METRICS = ["min_intensity", "max_intensity", "mean_intensity",
           "std_intensity", "pct_zero", "pct_255"]
RULES = {
    "low_standard_deviation": "std_intensity < 5 on the 0–255 scale",
    "nearly_all_black": "pct_zero >= 99 or mean_intensity <= 5",
    "nearly_all_white": "pct_255 >= 99 or mean_intensity >= 250",
    "wrong_dimensions": "width != 224 or height != 224",
    "missing": "expected path is not a file",
    "zero_byte": "file size == 0",
    "unreadable": "Pillow cannot fully decode the image",
}


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect(row):
    result = dict(row)
    result.update({key: None for key in ["width", "height", "mode", "file_size",
                                        *METRICS, "sha256"]})
    reasons = []
    path = IMAGES / row["Image Index"]
    if not path.is_file():
        reasons.append("missing")
    else:
        result["file_size"] = path.stat().st_size
        if result["file_size"] == 0:
            reasons.append("zero_byte")
        result["sha256"] = hash_file(path)
        try:
            with Image.open(path) as image:
                result.update(width=image.width, height=image.height, mode=image.mode)
                image.load()  # Force full decoding, rather than reading only the header.
                if image.size != (224, 224):
                    reasons.append("wrong_dimensions")
                # L images retain their original values. Other modes are explicitly
                # converted to 8-bit grayscale for comparable intensity statistics.
                pixels = np.asarray(image if image.mode == "L" else image.convert("L"),
                                    dtype=np.float64)
                result.update(
                    min_intensity=float(pixels.min()),
                    max_intensity=float(pixels.max()),
                    mean_intensity=float(pixels.mean()),
                    std_intensity=float(pixels.std(ddof=0)),
                    pct_zero=float(100 * np.mean(pixels == 0)),
                    pct_255=float(100 * np.mean(pixels == 255)),
                )
                if result["std_intensity"] < 5:
                    reasons.append("low_standard_deviation")
                if result["pct_zero"] >= 99 or result["mean_intensity"] <= 5:
                    reasons.append("nearly_all_black")
                if result["pct_255"] >= 99 or result["mean_intensity"] >= 250:
                    reasons.append("nearly_all_white")
        except Exception as error:
            reasons.append("unreadable")
            reasons.append(f"decode_error={type(error).__name__}: {error}")
    result["qc_flag"] = bool(reasons)
    result["qc_reason"] = "; ".join(reasons)
    return result


def numeric_summary(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    def clean(value):
        return None if pd.isna(value) else float(value)
    return {"count": len(values), "mean": clean(values.mean()),
            "median": clean(values.median()), "std": clean(values.std(ddof=1)),
            "min": clean(values.min()), "max": clean(values.max())}


def counts(series):
    return {str(key): int(value) for key, value in series.value_counts().sort_index().items()}


def group_summary(frame):
    return {
        "images": len(frame),
        "successfully_read_images": int(frame["mean_intensity"].notna().sum()),
        "flagged_images": int(frame["qc_flag"].sum()),
        "image_mode_counts": counts(frame["mode"]),
        "intensity_summary": {metric: numeric_summary(frame[metric]) for metric in METRICS},
    }


def visual_sheet(frame):
    # Two uniformly sampled images per split/class stratum, in manifest order.
    rng = np.random.default_rng(SEED)
    samples = []
    for split in ("train", "val", "test"):
        for target in (0, 1):
            group = frame[frame["split"].eq(split) & frame["binary_target"].eq(target)]
            indices = rng.choice(len(group), size=min(2, len(group)), replace=False)
            samples.extend(group.iloc[indices].to_dict("records"))
    tile_width, tile_height = 250, 280
    sheet = Image.new("RGB", (4 * tile_width, 3 * tile_height + 35), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), "NIH frozen cohort v1: stratified random QC sample, seed 42", fill="black")
    for index, row in enumerate(samples):
        x, y = (index % 4) * tile_width, 35 + (index // 4) * tile_height
        draw.text((x + 8, y + 4), row["Image Index"], fill="black")
        draw.text((x + 8, y + 20), f"{row['split']} | binary_target={row['binary_target']}", fill="black")
        try:
            with Image.open(IMAGES / row["Image Index"]) as image:
                preview = image.convert("RGB")
                preview.thumbnail((224, 224))
                sheet.paste(preview, (x + 13, y + 42))
        except Exception:
            draw.text((x + 13, y + 80), "Missing or unreadable", fill="red")
    sheet.save(REPORTS / "nih_image_qc_sample_v1.png")
    return [{key: row[key] for key in ("Image Index", "split", "binary_target")}
            for row in samples]


def main():
    manifest = pd.read_csv(MANIFEST)
    assert len(manifest) == EXPECTED, f"Expected {EXPECTED} manifest rows"
    assert manifest["Image Index"].duplicated().sum() == 0, "Duplicate Image Index"
    assert set(METADATA).issubset(manifest.columns), "Missing metadata columns"
    assert manifest["Image Index"].notna().all(), "Missing filename"
    assert all(Path(name).name == name and "/" not in name and "\\" not in name
               for name in manifest["Image Index"]), "Invalid filename"
    results = []
    for index, row in enumerate(manifest[METADATA].to_dict("records"), 1):
        results.append(inspect(row))
        if index % 2000 == 0 or index == EXPECTED:
            print(f"QC progress: {index}/{EXPECTED}", flush=True)
    frame = pd.DataFrame(results)
    assert len(frame) == EXPECTED, "QC must retain every manifest row"
    assert frame["Image Index"].tolist() == manifest["Image Index"].tolist(), (
        "No automatic exclusions or row reassignment are allowed"
    )
    reason_counts = {
        rule: sum(rule in value.split("; ") for value in frame["qc_reason"])
        for rule in RULES
    }
    duplicates = []
    for digest, group in frame.dropna(subset=["sha256"]).groupby("sha256", sort=True):
        if len(group) > 1:
            duplicates.append({"sha256": digest, "images": len(group),
                               "splits": sorted(group["split"].unique().tolist()),
                               "filenames": group["Image Index"].tolist()})
    cross_split = [group for group in duplicates if len(group["splits"]) > 1]
    REPORTS.mkdir(parents=True, exist_ok=True)
    summary = {
        "expected_images": EXPECTED,
        "successfully_read_images": int(frame["mean_intensity"].notna().sum()),
        "unreadable_images": reason_counts["unreadable"],
        "missing_images": reason_counts["missing"],
        "wrong_dimension_images": reason_counts["wrong_dimensions"],
        "zero_byte_images": reason_counts["zero_byte"],
        "flagged_images": int(frame["qc_flag"].sum()),
        "qc_rules": RULES,
        "qc_reason_counts": reason_counts,
        "automatic_exclusions": 0,
        "image_mode_counts": counts(frame["mode"]),
        "intensity_basis": "Original 8-bit L values; other modes converted by Pillow to 8-bit L. Percentages are 0–100. Per-image std uses ddof=0; summary std uses ddof=1. No source images are changed.",
        "intensity_summary": group_summary(frame)["intensity_summary"],
        "overall_summary": group_summary(frame),
        "split_summaries": {str(key): group_summary(group) for key, group in frame.groupby("split")},
        "class_summaries": {str(key): group_summary(group) for key, group in frame.groupby("binary_target")},
        "ap_pa_counts": {position: int(frame["View Position"].eq(position).sum()) for position in ("AP", "PA")},
        "view_position_summaries": {str(key): group_summary(group) for key, group in frame.groupby("View Position")},
        "sex_counts": counts(frame["Patient Sex"]),
        "sex_summaries": {str(key): group_summary(group) for key, group in frame.groupby("Patient Sex")},
        "age_summary_basis": "Image-weighted, unmodified manifest ages; no deduplication or age filtering.",
        "age_summaries": {
            "overall": numeric_summary(frame["Patient Age"]),
            "by_split": {str(key): numeric_summary(group["Patient Age"]) for key, group in frame.groupby("split")},
            "by_binary_target": {str(key): numeric_summary(group["Patient Age"]) for key, group in frame.groupby("binary_target")},
            "by_split_and_binary_target": {
                split: {str(target): numeric_summary(group["Patient Age"])
                        for target, group in rows.groupby("binary_target")}
                for split, rows in frame.groupby("split")
            },
        },
        "exact_duplicate_content_summary": {
            "method": "SHA256 of complete file bytes; not decoded-pixel similarity",
            "duplicated_content_hashes": len(duplicates),
            "images_involved": sum(group["images"] for group in duplicates),
            "groups": duplicates,
        },
        "cross_split_duplicate_content_count": len(cross_split),
        "cross_split_duplicate_content_images": sum(group["images"] for group in cross_split),
        "visual_sample_seed": SEED,
        "visual_sample": visual_sheet(frame),
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    frame.to_csv(REPORTS / "nih_image_qc_v1.csv", index=False, lineterminator="\n")
    report = json.dumps(summary, indent=2, allow_nan=False)
    (REPORTS / "nih_image_qc_v1_summary.json").write_text(report + "\n", encoding="utf-8")
    print(report, flush=True)
    # Save diagnostics even if a required file is missing, then fail explicitly.
    assert summary["missing_images"] == 0, "Expected local images are missing; see QC reports"
    assert all((IMAGES / name).is_file() for name in manifest["Image Index"]), (
        "All 19326 expected files must be present before QC completes"
    )
    assert summary["automatic_exclusions"] == 0, "Automatic exclusions are forbidden"
    print("PASS: all 19326 expected files are present; every image retained; QC reports created.")


if __name__ == "__main__":
    main()
