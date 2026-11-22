#!/usr/bin/env python3

import csv
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone

BASE = Path("experiments/EXP-DN121-BASE-S42")

S1 = BASE / "shift_sensitivity_v1" / "shift_sensitivity_condition_deltas.csv"
S2 = BASE / "error_transition_v1" / "error_transition_summary_all_conditions.csv"
S3 = BASE / "confidence_transition_v1" / "confidence_transition_summary_all_conditions.csv"

OUT = BASE / "failure_signature_robustness_v1"
OUT.mkdir(parents=True, exist_ok=True)

VALIDATION_N = 3078

SIGNATURES = [
    "performance_collapse",
    "failure_awareness_collapse",
    "silent_confidence_failure",
    "directional_bias",
]

SEVERITY_ORDER = {
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}


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
    return max(float(x), 0.0)


def identity(x):
    return float(x)


def minmax(values):
    lo = min(values)
    hi = max(values)

    if hi == lo:
        return [0.0 for _ in values]

    return [(x - lo) / (hi - lo) for x in values]


def rank_normalize(values):
    """
    Average rank converted to [0,1].
    Higher raw value => higher normalized harm.
    """
    n = len(values)

    indexed = list(enumerate(values))
    indexed.sort(key=lambda x: x[1])

    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i

        while (
            j + 1 < n
            and indexed[j + 1][1] == indexed[i][1]
        ):
            j += 1

        avg_rank = (i + j) / 2.0

        for k in range(i, j + 1):
            original_idx = indexed[k][0]
            ranks[original_idx] = avg_rank

        i = j + 1

    if n == 1:
        return [0.0]

    return [r / (n - 1) for r in ranks]


def mean_weighted(items):
    """
    items = [(value, weight), ...]
    """
    total_weight = sum(w for _, w in items)

    if total_weight == 0:
        return 0.0

    return sum(v * w for v, w in items) / total_weight


# --------------------------------------------------
# 1. Load aligned source data
# --------------------------------------------------

s1_rows = read_csv(S1)
s2_rows = read_csv(S2)
s3_rows = read_csv(S3)

s1 = {r["condition_id"]: r for r in s1_rows}
s2 = {r["condition_id"]: r for r in s2_rows}
s3 = {r["condition_id"]: r for r in s3_rows}

conditions = sorted(set(s1) & set(s2) & set(s3))

if len(conditions) != 15:
    raise RuntimeError(
        f"Expected 15 aligned shift conditions, found {len(conditions)}"
    )

if set(s1) != set(s2) or set(s1) != set(s3):
    raise RuntimeError(
        "S1/S2/S3 condition IDs do not align perfectly."
    )


# --------------------------------------------------
# 2. Build primitive raw features
# --------------------------------------------------

base_rows = []

for condition_id in conditions:

    a = s1[condition_id]
    b = s2[condition_id]
    c = s3[condition_id]

    correct_to_wrong = int(b["correct_to_wrong"])

    c2w_high_conf_90 = int(
        c["c2w_high_confidence_90_count"]
    )

    new_wrong_high_conf_rate = (
        c2w_high_conf_90 / correct_to_wrong
        if correct_to_wrong > 0
        else 0.0
    )

    unaware_new_error_rate = (
        1.0 - float(c["c2w_more_uncertain_rate"])
    )

    fp_change = int(b["fp_change_vs_clean"])
    fn_change = int(b["fn_change_vs_clean"])

    directional_bias_rate = (
        abs(fp_change - fn_change) / VALIDATION_N
    )

    base_rows.append({
        "condition_id": condition_id,
        "shift_family": a["shift_family"],
        "severity": a["severity"],

        "accuracy_drop_pp":
            float(a["accuracy_drop_percentage_points"]),

        "classification_auroc_drop":
            float(a["auroc_drop"]),

        "failure_rate_increase_pp":
            float(a["failure_rate_increase_percentage_points"]),

        "fd_auroc_drop":
            float(a["fd_auroc_drop"]),

        "aurc_increase":
            float(a["aurc_increase"]),

        "new_error_unaware_rate":
            unaware_new_error_rate,

        "all_wrong_high_conf_90_rate":
            float(c["all_wrong_high_confidence_90_rate"]),

        "new_wrong_high_conf_90_rate":
            new_wrong_high_conf_rate,

        "directional_bias_rate":
            directional_bias_rate,
    })


FEATURE_GROUPS = {
    "performance_collapse": [
        "accuracy_drop_pp",
        "classification_auroc_drop",
        "failure_rate_increase_pp",
    ],

    "failure_awareness_collapse": [
        "fd_auroc_drop",
        "aurc_increase",
        "new_error_unaware_rate",
    ],

    "silent_confidence_failure": [
        "all_wrong_high_conf_90_rate",
        "new_wrong_high_conf_90_rate",
        "new_error_unaware_rate",
    ],

    "directional_bias": [
        "directional_bias_rate",
    ],
}


# --------------------------------------------------
# 3. Define analysis variants
# --------------------------------------------------

VARIANTS = []

# Baseline
VARIANTS.append({
    "name": "baseline_minmax_equal_clip",
    "normalization": "minmax",
    "clip_adverse": True,
    "weight_scheme": "equal",
    "drop_component": None,
})

# Rank normalization
VARIANTS.append({
    "name": "rank_equal_clip",
    "normalization": "rank",
    "clip_adverse": True,
    "weight_scheme": "equal",
    "drop_component": None,
})

# No adverse clipping
VARIANTS.append({
    "name": "minmax_equal_no_clip",
    "normalization": "minmax",
    "clip_adverse": False,
    "weight_scheme": "equal",
    "drop_component": None,
})

# Alternative weights
VARIANTS.append({
    "name": "minmax_primary_weighted_clip",
    "normalization": "minmax",
    "clip_adverse": True,
    "weight_scheme": "primary_weighted",
    "drop_component": None,
})

VARIANTS.append({
    "name": "rank_primary_weighted_clip",
    "normalization": "rank",
    "clip_adverse": True,
    "weight_scheme": "primary_weighted",
    "drop_component": None,
})


# Leave-one-component-out variants
for signature, features in FEATURE_GROUPS.items():

    if len(features) <= 1:
        continue

    for feature in features:

        VARIANTS.append({
            "name":
                f"loo_{signature}__drop_{feature}",

            "normalization":
                "minmax",

            "clip_adverse":
                True,

            "weight_scheme":
                "equal",

            "drop_component":
                (signature, feature),
        })


# --------------------------------------------------
# 4. Utility to compute one variant
# --------------------------------------------------

def compute_variant(variant):

    processed = []

    # prepare raw values, optionally adverse clipping
    for r in base_rows:

        row = dict(r)

        for feature in {
            f
            for fs in FEATURE_GROUPS.values()
            for f in fs
        }:

            value = float(row[feature])

            if variant["clip_adverse"]:
                value = harm(value)

            row[f"processed_{feature}"] = value

        processed.append(row)

    # normalize each primitive feature
    for feature in {
        f
        for fs in FEATURE_GROUPS.values()
        for f in fs
    }:

        vals = [
            r[f"processed_{feature}"]
            for r in processed
        ]

        if variant["normalization"] == "minmax":
            normed = minmax(vals)

        elif variant["normalization"] == "rank":
            normed = rank_normalize(vals)

        else:
            raise ValueError(
                f"Unknown normalization: {variant['normalization']}"
            )

        for r, v in zip(processed, normed):
            r[f"norm_{feature}"] = v

    # compute signature scores
    results = []

    for r in processed:

        scores = {}

        for signature, features in FEATURE_GROUPS.items():

            active_features = list(features)

            drop_component = variant["drop_component"]

            if (
                drop_component is not None
                and drop_component[0] == signature
                and drop_component[1] in active_features
            ):
                active_features.remove(
                    drop_component[1]
                )

            weighted_items = []

            for idx, feature in enumerate(active_features):

                value = r[f"norm_{feature}"]

                if variant["weight_scheme"] == "equal":
                    weight = 1.0

                elif variant["weight_scheme"] == "primary_weighted":
                    # Gives first component of each signature
                    # modestly greater influence, but not dominance.
                    weight = 2.0 if idx == 0 else 1.0

                else:
                    raise ValueError(
                        f"Unknown weighting: "
                        f"{variant['weight_scheme']}"
                    )

                weighted_items.append(
                    (value, weight)
                )

            score = mean_weighted(
                weighted_items
            )

            scores[signature] = score

        ordered = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        results.append({
            "variant": variant["name"],
            "condition_id": r["condition_id"],
            "shift_family": r["shift_family"],
            "severity": r["severity"],

            "performance_collapse_score":
                scores["performance_collapse"],

            "failure_awareness_collapse_score":
                scores["failure_awareness_collapse"],

            "silent_confidence_failure_score":
                scores["silent_confidence_failure"],

            "directional_bias_score":
                scores["directional_bias"],

            "dominant_signature":
                ordered[0][0],

            "secondary_signature":
                ordered[1][0],

            "dominance_margin":
                ordered[0][1] - ordered[1][1],
        })

    return results


# --------------------------------------------------
# 5. Run all variants
# --------------------------------------------------

all_results = []

for variant in VARIANTS:
    all_results.extend(
        compute_variant(variant)
    )


# --------------------------------------------------
# 6. Save all variant-level results
# --------------------------------------------------

all_csv = (
    OUT /
    "robustness_all_variants.csv"
)

with open(
    all_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            all_results[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        all_results
    )


# --------------------------------------------------
# 7. Condition-level stability
# --------------------------------------------------

condition_summary = []

for condition_id in conditions:

    relevant = [
        r
        for r in all_results
        if r["condition_id"] == condition_id
    ]

    dominant_counts = Counter(
        r["dominant_signature"]
        for r in relevant
    )

    secondary_counts = Counter(
        r["secondary_signature"]
        for r in relevant
    )

    consensus_dom, consensus_dom_n = (
        dominant_counts.most_common(1)[0]
    )

    consensus_sec, consensus_sec_n = (
        secondary_counts.most_common(1)[0]
    )

    total = len(relevant)

    sample = relevant[0]

    condition_summary.append({
        "condition_id":
            condition_id,

        "shift_family":
            sample["shift_family"],

        "severity":
            sample["severity"],

        "variants_tested":
            total,

        "consensus_dominant_signature":
            consensus_dom,

        "dominant_consensus_rate":
            consensus_dom_n / total,

        "consensus_secondary_signature":
            consensus_sec,

        "secondary_consensus_rate":
            consensus_sec_n / total,

        "unique_dominant_signatures":
            len(dominant_counts),

        "dominant_signature_counts":
            json.dumps(
                dict(dominant_counts),
                sort_keys=True
            ),
    })


condition_summary.sort(
    key=lambda r: (
        r["shift_family"],
        SEVERITY_ORDER[r["severity"]]
    )
)


condition_csv = (
    OUT /
    "robustness_condition_consensus.csv"
)

with open(
    condition_csv,
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
    writer.writerows(
        condition_summary
    )


# --------------------------------------------------
# 8. Family-level dominant trajectory robustness
# --------------------------------------------------

family_variant_sequences = []

families = sorted(
    set(r["shift_family"] for r in base_rows)
)

variant_names = [
    v["name"]
    for v in VARIANTS
]

for family in families:

    for variant_name in variant_names:

        rel = [
            r
            for r in all_results
            if (
                r["shift_family"] == family
                and r["variant"] == variant_name
            )
        ]

        rel.sort(
            key=lambda r:
                SEVERITY_ORDER[r["severity"]]
        )

        seq = [
            r["dominant_signature"]
            for r in rel
        ]

        family_variant_sequences.append({
            "shift_family":
                family,

            "variant":
                variant_name,

            "mild":
                seq[0],

            "moderate":
                seq[1],

            "severe":
                seq[2],

            "trajectory":
                " -> ".join(seq),
        })


family_seq_csv = (
    OUT /
    "robustness_family_trajectories.csv"
)

with open(
    family_seq_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            family_variant_sequences[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        family_variant_sequences
    )


# --------------------------------------------------
# 9. Consensus trajectory per family
# --------------------------------------------------

family_consensus = []

for family in families:

    rel = [
        r
        for r in family_variant_sequences
        if r["shift_family"] == family
    ]

    trajectory_counts = Counter(
        r["trajectory"]
        for r in rel
    )

    top_trajectory, top_n = (
        trajectory_counts.most_common(1)[0]
    )

    family_consensus.append({
        "shift_family":
            family,

        "variants_tested":
            len(rel),

        "consensus_trajectory":
            top_trajectory,

        "trajectory_consensus_rate":
            top_n / len(rel),

        "unique_trajectories":
            len(trajectory_counts),

        "trajectory_counts":
            json.dumps(
                dict(trajectory_counts),
                sort_keys=True
            ),
    })


family_consensus_csv = (
    OUT /
    "robustness_family_consensus.csv"
)

with open(
    family_consensus_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            family_consensus[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        family_consensus
    )


# --------------------------------------------------
# 10. Human-readable report
# --------------------------------------------------

report = (
    OUT /
    "FAILURE_SIGNATURE_ROBUSTNESS_REPORT.md"
)

with open(
    report,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "# MedTrust-AI — Failure Signature Robustness Analysis v1\n\n"
    )

    f.write(
        "This analysis tests whether the exploratory failure-signature "
        "conclusions depend strongly on arbitrary scoring choices.\n\n"
    )

    f.write(
        "Variants include alternative normalization, weighting, "
        "adverse clipping, and leave-one-component-out analyses.\n\n"
    )

    f.write(
        "**Important:** robustness of a descriptive signature does not "
        "make it a validated clinical metric or prove novelty.\n\n"
    )

    f.write(
        f"Total scoring variants tested: **{len(VARIANTS)}**\n\n"
    )

    f.write(
        "## Condition-level dominant-signature consensus\n\n"
    )

    f.write(
        "| Condition | Consensus dominant | Agreement | "
        "Unique dominant labels |\n"
    )

    f.write(
        "|---|---|---:|---:|\n"
    )

    for r in condition_summary:

        f.write(
            f"| {r['condition_id']} "
            f"| {r['consensus_dominant_signature']} "
            f"| {r['dominant_consensus_rate']:.1%} "
            f"| {r['unique_dominant_signatures']} |\n"
        )

    f.write(
        "\n## Family trajectory robustness\n\n"
    )

    f.write(
        "| Shift family | Consensus trajectory | Agreement | "
        "Unique trajectories |\n"
    )

    f.write(
        "|---|---|---:|---:|\n"
    )

    for r in family_consensus:

        f.write(
            f"| {r['shift_family']} "
            f"| {r['consensus_trajectory']} "
            f"| {r['trajectory_consensus_rate']:.1%} "
            f"| {r['unique_trajectories']} |\n"
        )

    f.write(
        "\n## Interpretation\n\n"
    )

    f.write(
        "High agreement across reasonable scoring variants suggests that "
        "the observed failure pattern is less likely to be an artifact of "
        "one normalization or weighting choice. Low agreement indicates "
        "that the descriptive taxonomy is unstable and should not be "
        "formalized without further methodological work.\n"
    )


# --------------------------------------------------
# 11. Metadata + hashes
# --------------------------------------------------

metadata = {
    "analysis_name":
        "failure_signature_robustness_v1",

    "analysis_type":
        "exploratory_robustness",

    "variants_tested":
        len(VARIANTS),

    "variant_definitions":
        VARIANTS,

    "source_files": {
        str(S1): sha256_file(S1),
        str(S2): sha256_file(S2),
        str(S3): sha256_file(S3),
    },

    "validated_metric":
        False,

    "novelty_proven":
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
    all_csv,
    condition_csv,
    family_seq_csv,
    family_consensus_csv,
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
# 12. Terminal output
# --------------------------------------------------

print("=" * 82)
print(
    "MEDTRUST-AI — FAILURE SIGNATURE ROBUSTNESS ANALYSIS V1"
)
print("=" * 82)

print(
    f"\nVariants tested: {len(VARIANTS)}"
)

print(
    "\nCondition-level dominant-signature consensus:"
)

for r in condition_summary:

    print(
        f"  {r['condition_id']:<32} "
        f"{r['consensus_dominant_signature']:<30} "
        f"{r['dominant_consensus_rate']*100:>5.1f}%"
    )


print(
    "\nFamily-level trajectory consensus:"
)

for r in family_consensus:

    print(
        f"  {r['shift_family']:<24} "
        f"{r['trajectory_consensus_rate']*100:>5.1f}% | "
        f"{r['consensus_trajectory']}"
    )


high_consensus = [
    r
    for r in condition_summary
    if r["dominant_consensus_rate"] >= 0.80
]

low_consensus = [
    r
    for r in condition_summary
    if r["dominant_consensus_rate"] < 0.60
]


print(
    "\nRobustness summary:"
)

print(
    f"  Conditions with >=80% dominant-signature agreement: "
    f"{len(high_consensus)}/15"
)

print(
    f"  Conditions with <60% dominant-signature agreement: "
    f"{len(low_consensus)}/15"
)

print(
    "\nOutputs:"
)

for p in targets + [checksum_path]:
    print(f"  {p}")


print("\nExploratory robustness analysis: YES")
print("Validated metric: NO")
print("Novelty proven: NO")
print("Test set accessed: NO")
print("Model/inference rerun: NO")

print("=" * 82)
