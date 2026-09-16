#!/usr/bin/env python3
"""Replace unresolved primary EPPIC rows from a predeclared resolved reserve."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--reserve", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    primary = read_tsv(args.primary)
    reserve = read_tsv(args.reserve)
    kept = [row for row in primary if row.get("resolution_status") == "resolved"]
    missing = Counter(row["label"] for row in primary if row.get("resolution_status") != "resolved")
    replacements = []
    for label in ("biological", "crystal"):
        candidates = [
            row for row in reserve if row["label"] == label and row.get("resolution_status") == "resolved"
        ]
        if len(candidates) < missing[label]:
            raise SystemExit(
                f"only {len(candidates)} resolved reserve rows for {label}; need {missing[label]}"
            )
        for row in candidates[: missing[label]]:
            selected = dict(row)
            selected["panel_role"] = "reserve_replacement"
            replacements.append(selected)
    for row in kept:
        row["panel_role"] = "primary"
    final = kept + replacements
    final.sort(key=lambda row: (row["label"], row["pdb_id"], int(row["interface_serial"])))

    classes = Counter(row["label"] for row in final)
    if classes != Counter({"biological": 1000, "crystal": 1000}):
        raise SystemExit(f"unexpected final class counts: {classes}")
    if len({row["pdb_id"] for row in final}) != len(final):
        raise SystemExit("final panel contains repeated PDB entries")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(final[0].keys()) if final else []
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final)
    report = {
        "primary_rows": len(primary),
        "primary_unresolved": sum(missing.values()),
        "unresolved_by_class": dict(missing),
        "replacement_rows": len(replacements),
        "final_rows": len(final),
        "final_classes": dict(classes),
        "final_unique_pdbs": len({row["pdb_id"] for row in final}),
        "current_eppic_call_disagreements_retained": sum(
            row.get("label_matches_current_eppic_call") == "false" for row in final
        ),
    }
    args.audit.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

