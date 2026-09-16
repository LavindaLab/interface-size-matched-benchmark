#!/usr/bin/env python3
"""Freeze a family-clean, homomer-only, area-matched DC external validation panel.

This stage is deliberately descriptor blind.  It uses only published labels, exact EPPIC
interface identity/area, RCSB entity identity, and official 30%-identity sequence clusters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def variance(values: list[float]) -> float:
    center = mean(values)
    return sum((value - center) ** 2 for value in values) / (len(values) - 1)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def standardized_difference(a: list[float], b: list[float]) -> float:
    denominator = math.sqrt((variance(a) + variance(b)) / 2)
    return (mean(a) - mean(b)) / denominator if denominator else 0.0


def ordered_caliper_pairs(a: list[float], b: list[float], caliper: float) -> list[tuple[int, int]]:
    """Maximize pair count within a caliper, then minimize total distance (ordered 1-D match)."""
    n, m = len(a), len(b)
    states: list[list[tuple[int, float, list[tuple[int, int]]] | None]] = [
        [None] * (m + 1) for _ in range(n + 1)
    ]
    states[0][0] = (0, 0.0, [])

    def improve(i: int, j: int, candidate: tuple[int, float, list[tuple[int, int]]]) -> None:
        current = states[i][j]
        if current is None or (candidate[0], -candidate[1]) > (current[0], -current[1]):
            states[i][j] = candidate

    for i in range(n + 1):
        for j in range(m + 1):
            state = states[i][j]
            if state is None:
                continue
            if i < n:
                improve(i + 1, j, state)
            if j < m:
                improve(i, j + 1, state)
            if i < n and j < m and abs(a[i] - b[j]) <= caliper:
                improve(i + 1, j + 1, (state[0] + 1, state[1] + abs(a[i] - b[j]), state[2] + [(i, j)]))
    return states[n][m][2]  # type: ignore[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dc", type=Path, required=True)
    parser.add_argument("--many", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--area-min", type=float, default=800.0)
    parser.add_argument("--area-max", type=float, default=2000.0)
    parser.add_argument("--log-area-caliper", type=float, default=0.05)
    args = parser.parse_args()

    dc, many = read_tsv(args.dc), read_tsv(args.many)
    seen_pdbs = {row["pdb_id"].upper() for row in many}
    seen_clusters = {
        row[key]
        for row in many
        for key in ("sequence_cluster30_chain1", "sequence_cluster30_chain2")
        if row.get(key)
    }

    accounting = Counter()
    eligible = []
    for row in dc:
        accounting["input_interfaces"] += 1
        if row.get("resolution_status") != "resolved":
            accounting["excluded_unresolved"] += 1
            continue
        if row.get("rcsb_entity1") != row.get("rcsb_entity2"):
            accounting["excluded_heteromer"] += 1
            continue
        if row["pdb_id"].upper() in seen_pdbs:
            accounting["excluded_many_pdb_overlap"] += 1
            continue
        clusters = {row.get("sequence_cluster30_chain1", ""), row.get("sequence_cluster30_chain2", "")}
        if "" in clusters or clusters & seen_clusters:
            accounting["excluded_many_family_overlap_or_unmapped"] += 1
            continue
        area = float(row["area_eppic_a2"])
        if not args.area_min <= area <= args.area_max:
            accounting["excluded_outside_literature_ambiguity_band"] += 1
            continue
        eligible.append(row)

    bio = sorted(
        (row for row in eligible if row["label"] == "biological"),
        key=lambda row: (float(row["area_eppic_a2"]), row["pdb_id"], int(row["interface_serial"])),
    )
    xtal = sorted(
        (row for row in eligible if row["label"] == "crystal"),
        key=lambda row: (float(row["area_eppic_a2"]), row["pdb_id"], int(row["interface_serial"])),
    )
    bio_log = [math.log(float(row["area_eppic_a2"])) for row in bio]
    xtal_log = [math.log(float(row["area_eppic_a2"])) for row in xtal]
    matched_indices = ordered_caliper_pairs(bio_log, xtal_log, args.log_area_caliper)

    matched = []
    pair_differences = []
    for pair_id, (i, j) in enumerate(matched_indices, 1):
        for row in (bio[i], xtal[j]):
            output = dict(row)
            output.update(
                {
                    "external_match_pair": str(pair_id),
                    "external_family_group": row["sequence_cluster30_chain1"],
                    "external_panel_status": "FROZEN_BEFORE_DESCRIPTOR_COMPUTATION",
                }
            )
            matched.append(output)
        pair_differences.append(abs(bio_log[i] - xtal_log[j]))
    matched.sort(key=lambda row: (int(row["external_match_pair"]), row["label"]))
    write_tsv(args.output, matched)

    bio_area = [float(row["area_eppic_a2"]) for row in matched if row["label"] == "biological"]
    xtal_area = [float(row["area_eppic_a2"]) for row in matched if row["label"] == "crystal"]
    audit = {
        "status": "FROZEN_BEFORE_DESCRIPTOR_COMPUTATION",
        "source": "Duarte-Capitani active curated DCbio/DCxtal repository files",
        "source_interface_convention": (
            "plain PDB rows are largest interface / EPPIC interface 1; all listed integers are expanded"
        ),
        "eligibility": {
            "homomer_definition": "RCSB polymer entity 1 equals polymer entity 2",
            "family_exclusion": "remove either partner cluster seen in the full Many-2000 panel at RCSB 30% identity",
            "area_band_A2": [args.area_min, args.area_max],
            "area_band_basis": "Duarte et al. 2012 literature-defined crystal/bio overlap region",
        },
        "matching": (
            "maximum-cardinality ordered one-to-one assignment within an absolute log(EPPIC area) "
            f"caliper of {args.log_area_caliper}, then minimizing total distance; labels and area used, "
            "no coordinates or chemistry descriptors read"
        ),
        "accounting": dict(accounting),
        "eligible_by_class": dict(Counter(row["label"] for row in eligible)),
        "frozen_rows": len(matched),
        "frozen_pairs": len(matched) // 2,
        "frozen_unique_pdbs": len({row["pdb_id"] for row in matched}),
        "frozen_unique_families": len({row["external_family_group"] for row in matched}),
        "area_balance": {
            "biological_mean": mean(bio_area),
            "crystal_mean": mean(xtal_area),
            "biological_median": median(bio_area),
            "crystal_median": median(xtal_area),
            "standardized_mean_difference_bio_minus_crystal": standardized_difference(bio_area, xtal_area),
            "median_absolute_log_pair_difference": median(pair_differences),
            "max_absolute_log_pair_difference": max(pair_differences),
        },
        "locked_models": {
            "area_prior": "balanced logistic regression on interface_area_A2",
            "chemistry_rescue": [
                "contacts_per_100A2", "hbonds_per_100A2", "salt_bridges_per_100A2",
                "hydrophobic_contacts_per_100A2", "elec_complementarity",
            ],
            "training_data": "hard_improvement_development.tsv only; no refitting or feature selection on DC",
        },
        "primary_test": (
            "chemistry-rescue ROC AUC minus area-prior ROC AUC on the complete frozen panel, "
            "with 95% family-cluster bootstrap CI"
        ),
        "success_rule": "chemistry AUC >= 0.65 and lower 95% family-bootstrap bound for delta AUC > 0",
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
