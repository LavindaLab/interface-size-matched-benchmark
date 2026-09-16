#!/usr/bin/env python3
"""Attach official RCSB 30%-identity sequence-cluster IDs to an interface table."""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path


ENDPOINT = "https://data.rcsb.org/graphql"
QUERY = """query($ids:[String!]!){entries(entry_ids:$ids){
  rcsb_id
  polymer_entities{
    rcsb_id
    rcsb_polymer_entity_container_identifiers{auth_asym_ids}
    rcsb_polymer_entity_group_membership{aggregation_method similarity_cutoff group_id}
  }
}}"""


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def fetch_batch(ids: list[str], attempts: int = 4) -> list[dict[str, object]]:
    payload = json.dumps({"query": QUERY, "variables": {"ids": ids}}).encode()
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                ENDPOINT,
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "EnergeticDialects/0.1"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.load(response)
            if result.get("errors"):
                raise RuntimeError(result["errors"])
            return result["data"]["entries"]
        except Exception as error:  # network/API error is logged after bounded retries
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(f"RCSB query failed for {ids[0]}..{ids[-1]}: {last_error}")


def cluster30(entity: dict[str, object]) -> str:
    for membership in entity.get("rcsb_polymer_entity_group_membership") or []:
        if (
            membership.get("aggregation_method") == "sequence_identity"
            and float(membership.get("similarity_cutoff") or -1) == 30.0
        ):
            return str(membership.get("group_id") or "")
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    rows = read_tsv(args.input)
    pdb_ids = sorted({row["pdb_id"].upper() for row in rows})
    if args.cache.exists():
        entries = json.loads(args.cache.read_text())
    else:
        entries = []
        for start in range(0, len(pdb_ids), args.batch_size):
            entries.extend(fetch_batch(pdb_ids[start : start + args.batch_size]))
            print(f"[{min(start + args.batch_size, len(pdb_ids))}/{len(pdb_ids)}] RCSB entries", flush=True)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(json.dumps(entries, indent=2) + "\n")

    chain_map: dict[tuple[str, str], tuple[str, str]] = {}
    for entry in entries:
        if not entry:
            continue
        pdb_id = str(entry["rcsb_id"]).upper()
        for entity in entry.get("polymer_entities") or []:
            group = cluster30(entity)
            entity_id = str(entity.get("rcsb_id") or "")
            identifiers = entity.get("rcsb_polymer_entity_container_identifiers") or {}
            for chain in identifiers.get("auth_asym_ids") or []:
                chain_map[(pdb_id, str(chain))] = (entity_id, group)

    status = Counter()
    for row in rows:
        key1 = (row["pdb_id"].upper(), row["chain1"])
        key2 = (row["pdb_id"].upper(), row["chain2"])
        entity1, group1 = chain_map.get(key1, ("", ""))
        entity2, group2 = chain_map.get(key2, ("", ""))
        row.update(
            {
                "rcsb_entity1": entity1,
                "rcsb_entity2": entity2,
                "sequence_cluster30_chain1": group1,
                "sequence_cluster30_chain2": group2,
                "sequence_cluster30_pair": "|".join(sorted((group1, group2))) if group1 and group2 else "",
            }
        )
        status["mapped"] += bool(group1 and group2)
        status["unmapped"] += not bool(group1 and group2)

    fields = list(rows[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "source": ENDPOINT,
        "cluster_definition": "RCSB MMseqs2 sequence_identity group at similarity_cutoff 30",
        "rows": len(rows),
        "unique_pdbs": len(pdb_ids),
        "mapping_status": dict(status),
        "unique_chain_clusters": len(
            {row[key] for row in rows for key in ("sequence_cluster30_chain1", "sequence_cluster30_chain2") if row[key]}
        ),
        "unique_cluster_pairs": len({row["sequence_cluster30_pair"] for row in rows if row["sequence_cluster30_pair"]}),
    }
    args.audit.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
