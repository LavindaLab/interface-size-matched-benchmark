#!/usr/bin/env python3
"""Download exact EPPIC interface mmCIFs and compute lightweight shared descriptors.

The interface-specific EPPIC coordinate file already contains the two crystallographic partners
after application of the published symmetry operator. Descriptor computation reuses the historical
pipeline functions, but records conventional one-side interface area (`BSA_total / 2`) explicitly.
No EPPIC prediction score is used as a model feature.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path


import portable_interface_descriptors as contacts
from portable_interface_descriptors import buried_surface


AA3 = contacts.AA3
CONTACT_CUTOFF = 4.0


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def fetch_file(url: str, destination: Path, attempts: int = 4) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 100:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "EnergeticDialects/0.1"})
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
            if len(payload) <= 100:
                raise RuntimeError(f"short response ({len(payload)} bytes)")
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(destination)
            return
        except (urllib.error.URLError, TimeoutError, RuntimeError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(f"download failed after {attempts} attempts: {url}: {last_error}")


def protein_chain_counts(atoms: list[dict[str, object]]) -> Counter[str]:
    residues: dict[str, set[tuple[object, object, object]]] = {}
    for atom in atoms:
        if atom["resname"] not in AA3:
            continue
        chain = str(atom["chain"])
        residues.setdefault(chain, set()).add((atom.get("auth_seq"), atom.get("seq"), atom["resname"]))
    return Counter({chain: len(keys) for chain, keys in residues.items() if len(keys) >= 20})


def compute_one(row: dict[str, str], cache_dir: Path, compute_sasa: bool) -> dict[str, str]:
    output = dict(row)
    # Keep the error column in the schema even when the first completed/input-ordered row succeeds.
    output["descriptor_error"] = ""
    pdb_id, interface_id = row["pdb_id"].lower(), row["interface_serial"]
    cif = cache_dir / pdb_id[1:3] / f"{pdb_id}_interface_{interface_id}.cif"
    try:
        fetch_file(row["interface_cif_url"], cif)
        atoms = contacts.parse_atom_site(cif)
        chain_counts = protein_chain_counts(atoms)
        if len(chain_counts) != 2:
            raise RuntimeError(f"expected exactly 2 protein partners; found {dict(chain_counts)}")
        chain_a, chain_b = sorted(chain_counts)
        atoms_a = [atom for atom in atoms if atom["chain"] == chain_a and atom["resname"] in AA3]
        atoms_b = [atom for atom in atoms if atom["chain"] == chain_b and atom["resname"] in AA3]
        residues_a, residues_b, atom_pairs = contacts.interface_contacts(
            atoms, chain_a, chain_a, chain_b, chain_b, CONTACT_CUTOFF
        )
        proxy = contacts.interface_energy_proxy(atoms, chain_a, chain_b)
        ec = contacts.electrostatic_complementarity(atoms, chain_a, chain_b)
        if compute_sasa:
            bsa = buried_surface(atoms_a, atoms_b)
            bsa_total = float(bsa.get("BSA_A2", 0.0) or 0.0)
            interface_area = bsa_total / 2.0
            pct_hydrophobic_bsa = str(bsa.get("pct_hydrophobic_BSA", ""))
            burial_asymmetry = str(bsa.get("burial_asymmetry", ""))
            area_source = "independent_shrake_rupley"
        else:
            bsa_total = 0.0
            interface_area = float(row.get("area_eppic_a2", "") or 0.0)
            pct_hydrophobic_bsa = ""
            burial_asymmetry = ""
            area_source = "published_eppic_interface_area"

        def per_100(value: object) -> str:
            if not interface_area:
                return ""
            return f"{100.0 * float(value or 0.0) / interface_area:.6f}"

        output.update(
            {
                "descriptor_status": "ok",
                "coordinate_cache": str(cif),
                "resolved_chain_a": chain_a,
                "resolved_chain_b": chain_b,
                "chain_a_residues": str(chain_counts[chain_a]),
                "chain_b_residues": str(chain_counts[chain_b]),
                "interface_residues": str(len(residues_a) + len(residues_b)),
                "n_interface_contacts": str(atom_pairs),
                "BSA_total_A2": f"{bsa_total:.3f}" if compute_sasa else "",
                "interface_area_A2": f"{interface_area:.3f}",
                "interface_area_source": area_source,
                "pct_hydrophobic_BSA": pct_hydrophobic_bsa,
                "burial_asymmetry": burial_asymmetry,
                "interchain_hbonds": str(proxy.get("hbond_like_atom_pairs", "")),
                "salt_bridges": str(proxy.get("salt_bridge_atom_pairs", "")),
                "hydrophobic_contacts": str(proxy.get("hydrophobic_atom_pairs", "")),
                "elec_complementarity": str(ec.get("elec_complementarity", "")),
                "elec_favorable_pairs": str(ec.get("elec_favorable_pairs", "")),
                "elec_unfavorable_pairs": str(ec.get("elec_unfavorable_pairs", "")),
                "contacts_per_100A2": per_100(atom_pairs),
                "hbonds_per_100A2": per_100(proxy.get("hbond_like_atom_pairs", 0)),
                "salt_bridges_per_100A2": per_100(proxy.get("salt_bridge_atom_pairs", 0)),
                "hydrophobic_contacts_per_100A2": per_100(
                    proxy.get("hydrophobic_atom_pairs", 0)
                ),
                "eppic_area_ratio_ours_over_eppic": (
                    f"{interface_area / float(row['area_eppic_a2']):.6f}"
                    if compute_sasa and row.get("area_eppic_a2") and float(row["area_eppic_a2"])
                    else ""
                ),
            }
        )
    except Exception as error:
        output.update({"descriptor_status": "error", "descriptor_error": str(error)})
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--executor", choices=("process", "thread"), default="process")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--compute-sasa", action="store_true")
    args = parser.parse_args()

    rows = read_tsv(args.panel)
    if args.limit:
        rows = rows[: args.limit]
    results: list[dict[str, str] | None] = [None] * len(rows)
    executor_class = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor
    with executor_class(max_workers=args.workers) as pool:
        futures = {
            pool.submit(compute_one, row, args.cache_dir, args.compute_sasa): index
            for index, row in enumerate(rows)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            results[futures[future]] = future.result()
            if completed % 25 == 0 or completed == len(rows):
                print(f"[{completed}/{len(rows)}] descriptors", flush=True)
    final = [row for row in results if row is not None]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(final[0].keys()) if final else []
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final)

    status = Counter(row.get("descriptor_status", "") for row in final)
    classes_ok = Counter(row["label"] for row in final if row.get("descriptor_status") == "ok")
    ratios = [
        float(row["eppic_area_ratio_ours_over_eppic"])
        for row in final
        if row.get("descriptor_status") == "ok" and row.get("eppic_area_ratio_ours_over_eppic")
    ]
    report = {
        "rows": len(final),
        "status": dict(status),
        "successful_classes": dict(classes_ok),
        "median_area_ratio_ours_over_eppic": (
            sorted(ratios)[len(ratios) // 2] if ratios else None
        ),
        "area_policy": (
            "independent Shrake-Rupley area"
            if args.compute_sasa
            else "published EPPIC interface area used as the explicit size baseline"
        ),
        "feature_policy": "EPPIC prediction scores are audit fields only and are not model features",
    }
    args.audit.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
