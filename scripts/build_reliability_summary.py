#!/usr/bin/env python3

import csv
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

BASE = Path("experiments/EXP-DN121-BASE-S42")

S1 = BASE / "shift_sensitivity_v1" / "shift_sensitivity_severe_ranking.csv"
S2 = BASE / "error_transition_v1" / "error_transition_severe_ranking.csv"
S3 = BASE / "confidence_transition_v1" / "confidence_transition_severe_summary.csv"

OUT = BASE / "reliability_summary_v1"
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


s1 = {r["shift_family"]: r for r in read_csv(S1)}
s2 = {r["shift_family"]: r for r in read_csv(S2)}
s3 = {r["shift_family"]: r for r in read_csv(S3)}

families = [
    "gaussian_blur",
    "gaussian_noise",
    "resolution_degradation",
    "brightness",
    "contrast",
]

rows = []

for family in families:
    a = s1[family]
    b = s2[family]
    c = s3[family]

    rows.append({
        "shift_family": family,

        "overall_damage_rank":
            int(a["overall_severe_damage_rank"]),

        "accuracy_drop_pp":
            float(a["accuracy_drop_percentage_points"]),

        "auroc_drop":
            float(a["auroc_drop"]),

        "failure_rate_increase_pp":
            float(a["failure_rate_increase_percentage_points"]),

        "fd_auroc_drop":
            float(a["fd_auroc_drop"]),

        "aurc_increase":
            float(a["aurc_increase"]),

        "correct_to_wrong":
            int(b["correct_to_wrong"]),

        "wrong_to_correct":
            int(b["wrong_to_correct"]),

        "net_new_errors":
            int(b["net_new_errors"]),

        "fp_change_vs_clean":
            int(b["fp_change_vs_clean"]),

        "fn_change_vs_clean":
            int(b["fn_change_vs_clean"]),

        "c2w_more_uncertain_rate":
            float(c["c2w_more_uncertain_rate"]),

        "c2w_high_confidence_90_count":
            int(c["c2w_high_confidence_90_count"]),

        "all_wrong_high_confidence_90_count":
            int(c["all_wrong_high_confidence_90_count"]),
    })


csv_path = OUT / "reliability_severe_shift_summary.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(rows)


md_path = OUT / "RELIABILITY_SUMMARY.md"

with open(md_path, "w", encoding="utf-8") as f:

    f.write("# MedTrust-AI — Consolidated Reliability Summary\n\n")

    f.write(
        "This report consolidates the completed validation-only "
        "shift sensitivity, error-transition, and confidence-transition analyses. "
        "No training or inference was rerun.\n\n"
    )

    f.write("## Severe shift comparison\n\n")

    f.write(
        "| Rank | Shift | Acc drop (pp) | AUROC drop | "
        "Failure +pp | FD-AUROC drop | AURC + | "
        "Net new errors | FP change | FN change | "
        "C→W more uncertain | Wrong conf≥0.90 |\n"
    )

    f.write(
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )

    for r in rows:
        f.write(
            f"| {r['overall_damage_rank']} "
            f"| {r['shift_family']} "
            f"| {r['accuracy_drop_pp']:.2f} "
            f"| {r['auroc_drop']:.4f} "
            f"| {r['failure_rate_increase_pp']:.2f} "
            f"| {r['fd_auroc_drop']:.4f} "
            f"| {r['aurc_increase']:.4f} "
            f"| {r['net_new_errors']} "
            f"| {r['fp_change_vs_clean']:+d} "
            f"| {r['fn_change_vs_clean']:+d} "
            f"| {r['c2w_more_uncertain_rate']:.1%} "
            f"| {r['all_wrong_high_confidence_90_count']} |\n"
        )

    f.write("\n## Main findings\n\n")

    f.write(
        "- **Gaussian blur** was the most damaging overall severe shift. "
        "It caused the largest accuracy drop, failure-rate increase, "
        "AURC increase, and net increase in errors.\n"
    )

    f.write(
        "- **Gaussian noise** caused the largest classification AUROC drop.\n"
    )

    f.write(
        "- **Resolution degradation** caused the largest drop in "
        "failure-detection AUROC.\n"
    )

    f.write(
        "- Severe blur produced a strong rise in false positives, "
        "while severe contrast produced the largest false-negative increase.\n"
    )

    f.write(
        "- Under blur, noise, and resolution degradation, most newly wrong "
        "predictions became more uncertain. In contrast and brightness shifts, "
        "the uncertainty response was weaker.\n"
    )

    f.write(
        "- High-confidence wrong predictions remained especially visible "
        "under severe contrast and brightness shifts.\n"
    )

    f.write("\n## Interpretation\n\n")

    f.write(
        "The reliability impact of distribution shift is not uniform. "
        "Some shifts mainly reduce classification performance, others weaken "
        "failure detection, and some leave the model confidently wrong. "
        "Therefore reliability should be evaluated using multiple complementary "
        "metrics rather than accuracy alone.\n\n"
    )

    f.write(
        "All findings are validation-development results and should not be "
        "interpreted as independent test conclusions or statistical significance tests.\n"
    )


metadata = {
    "analysis_name": "reliability_summary_v1",
    "inputs": {
        str(S1): sha256_file(S1),
        str(S2): sha256_file(S2),
        str(S3): sha256_file(S3),
    },
    "validation_only": True,
    "test_set_accessed": False,
    "model_rerun": False,
    "inference_rerun": False,
    "generated_utc": datetime.now(timezone.utc).isoformat(),
}

meta_path = OUT / "analysis_metadata.json"

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)


checksum_path = OUT / "SHA256SUMS.txt"

with open(checksum_path, "w", encoding="utf-8") as f:
    for p in [csv_path, md_path, meta_path]:
        f.write(f"{sha256_file(p)}  {p.name}\n")


print("=" * 72)
print("MEDTRUST-AI — CONSOLIDATED RELIABILITY SUMMARY V1")
print("=" * 72)

for r in rows:
    print(
        f"{r['overall_damage_rank']}. "
        f"{r['shift_family']:<24} "
        f"acc_drop={r['accuracy_drop_pp']:.2f}pp | "
        f"net_errors={r['net_new_errors']:+d} | "
        f"FD_drop={r['fd_auroc_drop']:.4f} | "
        f"wrong_conf>=0.90={r['all_wrong_high_confidence_90_count']}"
    )

print("\nOutputs:")
print(f"  {csv_path}")
print(f"  {md_path}")
print(f"  {meta_path}")
print(f"  {checksum_path}")

print("\nTest set accessed: NO")
print("Model/inference rerun: NO")
print("=" * 72)
