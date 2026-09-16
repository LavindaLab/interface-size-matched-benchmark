#!/usr/bin/env python3
"""Build a provenance manifest from the published EPPIC interface benchmark files.

ManyBio/ManyXtal provide a PDB identifier and EPPIC interface serial.  In the DC source files,
rows containing only a PDB identifier denote the largest crystal interface (EPPIC serial 1), and
following integers enumerate every labelled interface for entries with more than one.  This is
the convention documented by Duarte et al. (2012), Additional file 1, Tables S1/S2: the reported
area is that of the largest interface.  Expanding all integers reproduces the active repository's
81 DCbio and 82 DCxtal interfaces (the publication originally reported 83/82; two biological
entries were subsequently removed in the curated source file).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_METADATA = {
    "Many": {
        "citation": "Baskaran et al., BMC Structural Biology 2014, 14:22",
        "doi": "10.1186/s12900-014-0022-0",
        "role": "large benchmark",
    },
    "DC": {
        "citation": "Duarte et al., BMC Bioinformatics 2012, 13:334",
        "doi": "10.1186/1471-2105-13-334",
        "role": "difficult-case benchmark",
    },
}


def parse_many(path: Path, label: str) -> list[dict[str, str]]:
    rows = []
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 2:
            raise ValueError(f"{path}:{line_number}: expected PDB and interface serial")
        rows.append(
            {
                "pdb_id": fields[0].upper(),
                "interface_serial": fields[1],
                "label": label,
                "dataset": "Many",
                "source_file": path.name,
                "source_line": str(line_number),
                "pair_specific": "yes",
                "curation_status": "published_interface_label",
            }
        )
    return rows


def parse_dc(path: Path, label: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows, exclusions = [], []
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            exclusions.append(
                {
                    "dataset": "DC",
                    "label": label,
                    "source_file": path.name,
                    "source_line": str(line_number),
                    "source_text": stripped,
                }
            )
            continue
        fields = stripped.split()
        pdb_id = fields[0].upper()
        serials = [field for field in fields[1:] if field.isdigit()] or ["1"]
        for interface_serial in serials:
            rows.append(
                {
                    "pdb_id": pdb_id,
                    "interface_serial": interface_serial,
                    "label": label,
                    "dataset": "DC",
                    "source_file": path.name,
                    "source_line": str(line_number),
                    "pair_specific": "yes",
                    "curation_status": (
                        "published_interface_label"
                        if len(fields) > 1
                        else "published_largest_interface_serial_1"
                    ),
                }
            )
    return rows, exclusions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    many = parse_many(args.data_dir / "ManyBio.txt", "biological")
    many += parse_many(args.data_dir / "ManyXtal.txt", "crystal")
    dc_bio, excluded_bio = parse_dc(args.data_dir / "DCbio.txt", "biological")
    dc_xtal, excluded_xtal = parse_dc(args.data_dir / "DCxtal.txt", "crystal")
    dc = dc_bio + dc_xtal
    exclusions = excluded_bio + excluded_xtal
    all_rows = many + dc

    by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        if row["interface_serial"]:
            by_key[(row["pdb_id"], row["interface_serial"])].append(row)
    conflicts = []
    for (pdb_id, interface_serial), records in sorted(by_key.items()):
        labels = sorted({record["label"] for record in records})
        if len(labels) > 1:
            conflicts.append(
                {
                    "pdb_id": pdb_id,
                    "interface_serial": interface_serial,
                    "labels": ",".join(labels),
                    "datasets": ",".join(sorted({record["dataset"] for record in records})),
                }
            )

    many_pdbs = {row["pdb_id"] for row in many}
    dc_pdbs = {row["pdb_id"] for row in dc}
    cross_dataset_pdbs = sorted(many_pdbs & dc_pdbs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "pdb_id", "interface_serial", "label", "dataset", "source_file", "source_line",
        "pair_specific", "curation_status",
    ]
    with (args.out_dir / "eppic_interface_manifest.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_rows)
    with (args.out_dir / "eppic_source_exclusions.tsv").open("w", newline="") as handle:
        fields_excluded = ["dataset", "label", "source_file", "source_line", "source_text"]
        writer = csv.DictWriter(handle, fieldnames=fields_excluded, delimiter="\t")
        writer.writeheader()
        writer.writerows(exclusions)
    with (args.out_dir / "eppic_label_conflicts.tsv").open("w", newline="") as handle:
        fields_conflict = ["pdb_id", "interface_serial", "labels", "datasets"]
        writer = csv.DictWriter(handle, fieldnames=fields_conflict, delimiter="\t")
        writer.writeheader()
        writer.writerows(conflicts)
    (args.out_dir / "eppic_cross_dataset_pdbs.txt").write_text(
        "".join(f"{pdb_id}\n" for pdb_id in cross_dataset_pdbs)
    )

    report = {
        "source_metadata": SOURCE_METADATA,
        "counts": {
            "many_rows": len(many),
            "many_classes": dict(Counter(row["label"] for row in many)),
            "many_unique_pdbs": len(many_pdbs),
            "dc_rows": len(dc),
            "dc_classes": dict(Counter(row["label"] for row in dc)),
            "dc_pair_specific_rows": sum(row["pair_specific"] == "yes" for row in dc),
            "dc_rows_blocked_for_adjudication": sum(row["pair_specific"] == "no" for row in dc),
            "source_comment_or_exclusion_lines": len(exclusions),
            "pair_level_label_conflicts": len(conflicts),
            "pdbs_shared_between_many_and_dc": len(cross_dataset_pdbs),
        },
        "split_rule": (
            "Group by PDB and sequence/family cluster. Any PDB shared between Many and DC must be "
            "removed from the training source before DC can serve as a source-held-out test."
        ),
    }
    (args.out_dir / "eppic_manifest_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
