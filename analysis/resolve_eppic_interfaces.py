#!/usr/bin/env python3
"""Resolve published EPPIC interface serials to exact interface metadata.

The resolver uses the public EPPIC REST endpoint, caches one JSON response per PDB entry, and can
resume safely. It does not download interface coordinates; the resolved `interface_cif_url` points
to the exact transformed two-partner mmCIF for the subsequent descriptor stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_BASE = "https://eppic-rest.rcsb.org"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def fetch_json(url: str, attempts: int = 4) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "EnergeticDialects/0.1"})
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed after {attempts} attempts: {url}: {last_error}")


def load_or_fetch_interfaces(pdb_id: str, cache_dir: Path, base_url: str) -> list[dict[str, object]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{pdb_id.lower()}.json"
    if cache_path.exists() and cache_path.stat().st_size:
        return json.loads(cache_path.read_text())
    url = f"{base_url}/rest/api/v3/job/interfaces/{pdb_id.lower()}"
    payload = fetch_json(url)
    if not isinstance(payload, list):
        raise RuntimeError(f"unexpected interface response for {pdb_id}: {type(payload).__name__}")
    cache_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def score_call(interface: dict[str, object], method: str) -> tuple[str, str, str]:
    for score in interface.get("interfaceScores", []):
        if score.get("method") == method:
            return (
                str(score.get("callName", "")),
                str(score.get("score", "")),
                str(score.get("confidence", "")),
            )
    return "", "", ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument("--dataset", default="Many")
    parser.add_argument("--label", choices=("biological", "crystal"), default=None)
    parser.add_argument("--limit-pdbs", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    rows = [
        row
        for row in read_tsv(args.manifest)
        if row.get("dataset") == args.dataset
        and row.get("interface_serial")
        and (args.label is None or row.get("label") == args.label)
    ]
    allowed_pdbs = []
    seen = set()
    for row in rows:
        if row["pdb_id"] not in seen:
            seen.add(row["pdb_id"])
            allowed_pdbs.append(row["pdb_id"])
    if args.limit_pdbs:
        allowed_pdbs = allowed_pdbs[: args.limit_pdbs]
    allowed = set(allowed_pdbs)
    rows = [row for row in rows if row["pdb_id"] in allowed]

    resolved_by_pdb: dict[str, list[dict[str, str]]] = {}
    by_pdb: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_pdb.setdefault(row["pdb_id"], []).append(row)

    def resolve_one(pdb_id: str) -> list[dict[str, str]]:
        resolved = []
        try:
            interfaces = load_or_fetch_interfaces(pdb_id, args.cache_dir, args.base_url)
            by_id = {str(interface.get("interfaceId")): interface for interface in interfaces}
            for source_row in by_pdb[pdb_id]:
                interface = by_id.get(source_row["interface_serial"])
                output = dict(source_row)
                if interface is None:
                    output.update({"resolution_status": "interface_serial_not_found"})
                else:
                    final_call, final_score, final_confidence = score_call(interface, "eppic")
                    output.update(
                        {
                            "resolution_status": "resolved",
                            "chain1": str(interface.get("chain1", "")),
                            "chain2": str(interface.get("chain2", "")),
                            "operator": str(interface.get("operator", "")),
                            "operator_type": str(interface.get("operatorType", "")),
                            "operator_id": str(interface.get("operatorId", "")),
                            "area_eppic_a2": str(interface.get("area", "")),
                            "infinite": str(interface.get("infinite", "")).lower(),
                            "isologous": str(interface.get("isologous", "")).lower(),
                            "eppic_current_call": final_call,
                            "eppic_current_score": final_score,
                            "eppic_current_confidence": final_confidence,
                            "label_matches_current_eppic_call": str(
                                (source_row["label"] == "biological" and final_call == "bio")
                                or (source_row["label"] == "crystal" and final_call == "xtal")
                            ).lower(),
                            "interface_cif_url": (
                                f"{args.base_url}/rest/api/v3/job/interfaceCifFile/"
                                f"{pdb_id.lower()}/{source_row['interface_serial']}"
                            ),
                        }
                    )
                resolved.append(output)
        except Exception as error:  # keep the batch resumable and auditable
            for source_row in by_pdb[pdb_id]:
                output = dict(source_row)
                output.update({"resolution_status": "request_error", "resolution_error": str(error)})
                resolved.append(output)
        if args.delay:
            time.sleep(args.delay)
        return resolved

    if args.workers <= 1:
        for index, pdb_id in enumerate(allowed_pdbs, 1):
            resolved_by_pdb[pdb_id] = resolve_one(pdb_id)
            print(f"[{index}/{len(allowed_pdbs)}] {pdb_id}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(resolve_one, pdb_id): pdb_id for pdb_id in allowed_pdbs}
            for index, future in enumerate(as_completed(futures), 1):
                pdb_id = futures[future]
                try:
                    resolved_by_pdb[pdb_id] = future.result()
                except Exception as error:  # defensive: resolve_one normally records errors
                    resolved_by_pdb[pdb_id] = [
                        {
                            **source_row,
                            "resolution_status": "worker_error",
                            "resolution_error": str(error),
                        }
                        for source_row in by_pdb[pdb_id]
                    ]
                if index % 25 == 0 or index == len(allowed_pdbs):
                    print(f"[{index}/{len(allowed_pdbs)}] resolved", flush=True)

    resolved = [row for pdb_id in allowed_pdbs for row in resolved_by_pdb[pdb_id]]

    extra_fields = [
        "resolution_status", "resolution_error", "chain1", "chain2", "operator",
        "operator_type", "operator_id", "area_eppic_a2", "infinite", "isologous",
        "eppic_current_call", "eppic_current_score", "eppic_current_confidence",
        "label_matches_current_eppic_call", "interface_cif_url",
    ]
    source_fields = list(rows[0].keys()) if rows else []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=source_fields + extra_fields, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(resolved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
