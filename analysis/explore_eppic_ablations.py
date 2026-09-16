#!/usr/bin/env python3
"""Post hoc single-feature/leave-one-out audit under strict 30%-sequence grouping."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import evaluate_eppic_sequence_grouped as grouped


AREA = "interface_area_A2"
CHEMISTRY = [
    "contacts_per_100A2",
    "hbonds_per_100A2",
    "salt_bridges_per_100A2",
    "hydrophobic_contacts_per_100A2",
    "elec_complementarity",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    rows = grouped.read_tsv(args.input)
    y = np.asarray([row["label"] == "biological" for row in rows], dtype=int)
    groups = grouped.component_groups(rows)
    grouped.FEATURE_SETS = {
        "area_only": [AREA],
        **{f"area_plus_{feature}": [AREA, feature] for feature in CHEMISTRY},
        "area_plus_all_normalized_chemistry": [AREA, *CHEMISTRY],
        "area_plus_all_except_hydrophobic_density": [
            AREA, *[feature for feature in CHEMISTRY if feature != "hydrophobic_contacts_per_100A2"]
        ],
    }
    result = grouped.evaluate_grouping(
        rows, y, groups, args.splits, args.repeats, args.bootstrap, args.seed
    )
    report = {
        "status": "post hoc descriptor ablation; hypothesis-generating only",
        "grouping": "strict connected components of shared RCSB 30%-identity chain clusters",
        "n_rows": len(rows),
        "class_counts": dict(Counter(row["label"] for row in rows)),
        "feature_sets": grouped.FEATURE_SETS,
        "result": result,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
