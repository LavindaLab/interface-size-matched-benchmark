#!/usr/bin/env python3
"""Portable geometric descriptors used by the Energetic Dialects benchmark.

This module contains the exact atom parsing, contact counting, formal-charge,
carbon-pair, charge-balance, and optional Shrake-Rupley calculations needed by
``compute_eppic_descriptors.py``.  It has no project-external imports.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
POLAR_ELEMENTS = {"N", "O", "S"}
POLAR_SIDECHAIN_ATOMS = {
    "HIS": {"ND1", "NE2"},
    "CYS": {"SG"},
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
    "ASN": {"OD1", "ND2"},
    "GLN": {"OE1", "NE2"},
    "MET": {"SD"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
}
POSITIVE_ATOMS = {
    "ARG": {"NH1", "NH2", "NE"},
    "LYS": {"NZ"},
    "HIS": {"ND1", "NE2"},
}
NEGATIVE_ATOMS = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
}
HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO", "TYR"}
VDW_RADII = {
    "H": 1.10, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
    "CU": 1.40, "ZN": 1.39, "FE": 1.47, "MG": 1.73, "MN": 1.61,
    "CA": 1.97, "NA": 2.27, "K": 2.75, "CL": 1.75, "F": 1.47,
}


def parse_atom_site(cif_path: Path) -> list[dict[str, object]]:
    """Parse the atom-site loop needed by the descriptor calculations."""
    atoms: list[dict[str, object]] = []
    headers: list[str] = []
    in_atom_loop = False
    for line in cif_path.read_text(errors="ignore").splitlines():
        if line.startswith("_atom_site."):
            in_atom_loop = True
            headers.append(line.strip().split(".", 1)[1])
            continue
        if in_atom_loop:
            if line.startswith("#"):
                break
            if not line or line.startswith("loop_"):
                continue
            parts = line.split()
            if len(parts) < len(headers):
                continue
            row = dict(zip(headers, parts))
            try:
                atoms.append(
                    {
                        "element": row["type_symbol"].upper(),
                        "atom": row["label_atom_id"].upper(),
                        "resname": row["label_comp_id"].upper(),
                        "chain": row["label_asym_id"],
                        "auth_chain": row.get("auth_asym_id", row["label_asym_id"]),
                        "seq": None if row["label_seq_id"] in {".", "?"} else int(row["label_seq_id"]),
                        "auth_seq": (
                            None if row["auth_seq_id"] in {".", "?"} else int(row["auth_seq_id"])
                        ),
                        "x": float(row["Cartn_x"]),
                        "y": float(row["Cartn_y"]),
                        "z": float(row["Cartn_z"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    return atoms


def _residue_label(atom: dict[str, object], protein: str) -> str:
    seq = atom["seq"] if atom["seq"] is not None else atom["auth_seq"]
    return f"{protein}:{atom['resname']}{seq}"


def _grid_pairs(
    atoms_a: list[dict[str, object]],
    atoms_b: list[dict[str, object]],
    cutoff: float,
):
    cell_size = cutoff
    cutoff2 = cutoff * cutoff

    def cell(atom: dict[str, object]) -> tuple[int, int, int]:
        return (
            int(math.floor(float(atom["x"]) / cell_size)),
            int(math.floor(float(atom["y"]) / cell_size)),
            int(math.floor(float(atom["z"]) / cell_size)),
        )

    grid_b: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(list)
    for atom_b in atoms_b:
        grid_b[cell(atom_b)].append(atom_b)
    offsets = list(itertools.product((-1, 0, 1), repeat=3))
    for atom_a in atoms_a:
        cx, cy, cz = cell(atom_a)
        for dx, dy, dz in offsets:
            for atom_b in grid_b.get((cx + dx, cy + dy, cz + dz), []):
                distance2 = (
                    (float(atom_a["x"]) - float(atom_b["x"])) ** 2
                    + (float(atom_a["y"]) - float(atom_b["y"])) ** 2
                    + (float(atom_a["z"]) - float(atom_b["z"])) ** 2
                )
                if distance2 <= cutoff2:
                    yield atom_a, atom_b, math.sqrt(distance2)


def interface_contacts(atoms, chain_a, protein_a, chain_b, protein_b, cutoff):
    atoms_a = [a for a in atoms if a["chain"] == chain_a and a["resname"] in AA3]
    atoms_b = [a for a in atoms if a["chain"] == chain_b and a["resname"] in AA3]
    residues_a, residues_b = set(), set()
    atom_pair_count = 0
    for atom_a, atom_b, _ in _grid_pairs(atoms_a, atoms_b, cutoff):
        residues_a.add(_residue_label(atom_a, protein_a))
        residues_b.add(_residue_label(atom_b, protein_b))
        atom_pair_count += 1
    return residues_a, residues_b, atom_pair_count


def charged_sign(atom: dict[str, object]) -> int:
    if atom["atom"] in POSITIVE_ATOMS.get(str(atom["resname"]), set()):
        return 1
    if atom["atom"] in NEGATIVE_ATOMS.get(str(atom["resname"]), set()):
        return -1
    return 0


def interface_energy_proxy(atoms, chain_a, chain_b, cutoff=8.0):
    """Return the fixed geometric atom-pair counts used in the benchmark."""
    atoms_a = [a for a in atoms if a["chain"] == chain_a and a["resname"] in AA3]
    atoms_b = [a for a in atoms if a["chain"] == chain_b and a["resname"] in AA3]
    polar_pairs = opposite_pairs = hydrophobic_pairs = 0
    close_contacts = clashes = 0
    score = 0.0
    for atom_a, atom_b, distance in _grid_pairs(atoms_a, atoms_b, cutoff):
        if distance < 2.0:
            score += 10.0
            clashes += 1
        elif distance < 2.5:
            score += 3.0
            close_contacts += 1
        sign_a, sign_b = charged_sign(atom_a), charged_sign(atom_b)
        if sign_a and sign_b and distance <= 4.0:
            if sign_a != sign_b:
                score -= 1.5
                opposite_pairs += 1
            else:
                score += 0.75
        if (
            atom_a["element"] in POLAR_ELEMENTS
            and atom_a["atom"] in POLAR_SIDECHAIN_ATOMS.get(str(atom_a["resname"]), set())
            and atom_b["element"] in POLAR_ELEMENTS
            and atom_b["atom"] in POLAR_SIDECHAIN_ATOMS.get(str(atom_b["resname"]), set())
            and 2.5 <= distance <= 3.5
        ):
            score -= 0.5
            polar_pairs += 1
        if (
            atom_a["element"] == atom_b["element"] == "C"
            and atom_a["resname"] in HYDROPHOBIC_RESIDUES
            and atom_b["resname"] in HYDROPHOBIC_RESIDUES
            and distance <= 4.0
        ):
            score -= 0.08
            hydrophobic_pairs += 1
    return {
        "simple_interface_energy_proxy": round(score, 3),
        "salt_bridge_atom_pairs": opposite_pairs,
        "hbond_like_atom_pairs": polar_pairs,
        "hydrophobic_atom_pairs": hydrophobic_pairs,
        "close_contact_penalties": close_contacts,
        "clash_penalties": clashes,
    }


def electrostatic_complementarity(atoms, chain_a, chain_b, cutoff=6.0):
    charged_a = [
        a for a in atoms if a["chain"] == chain_a and a["resname"] in AA3 and charged_sign(a)
    ]
    charged_b = [
        a for a in atoms if a["chain"] == chain_b and a["resname"] in AA3 and charged_sign(a)
    ]
    favorable = unfavorable = 0
    for atom_a, atom_b, _ in _grid_pairs(charged_a, charged_b, cutoff):
        if charged_sign(atom_a) != charged_sign(atom_b):
            favorable += 1
        else:
            unfavorable += 1
    total = favorable + unfavorable
    balance = (favorable - unfavorable) / total if total else ""
    return {
        "elec_complementarity": round(balance, 3) if balance != "" else "",
        "elec_favorable_pairs": favorable,
        "elec_unfavorable_pairs": unfavorable,
    }


def _sphere_points(n: int = 62) -> np.ndarray:
    points = np.empty((n, 3))
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(n):
        y = 1.0 - (index / (n - 1)) * 2.0
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * index
        points[index] = (math.cos(theta) * radius, y, math.sin(theta) * radius)
    return points


_SPHERE = _sphere_points()


def _sasa(atoms: list[dict[str, object]], probe: float = 1.4) -> np.ndarray:
    if not atoms:
        return np.zeros(0)
    coordinates = np.asarray([[a["x"], a["y"], a["z"]] for a in atoms], dtype=float)
    radii = np.asarray([VDW_RADII.get(str(a["element"]), 1.70) for a in atoms]) + probe
    cell_size = float(radii.max()) * 2.0
    keys = np.floor(coordinates / cell_size).astype(int)
    grid: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        grid[tuple(key)].append(index)
    values = np.zeros(len(atoms))
    for index, key in enumerate(keys):
        neighbors: list[int] = []
        for dx, dy, dz in itertools.product((-1, 0, 1), repeat=3):
            neighbors.extend(grid.get((key[0] + dx, key[1] + dy, key[2] + dz), []))
        neighbors = [item for item in neighbors if item != index]
        points = coordinates[index] + _SPHERE * radii[index]
        if neighbors:
            neighbor_coordinates = coordinates[neighbors]
            neighbor_radii = radii[neighbors]
            distance2 = ((points[:, None, :] - neighbor_coordinates[None, :, :]) ** 2).sum(-1)
            accessible = ~(distance2 < (neighbor_radii[None, :] ** 2)).any(axis=1)
            count = int(accessible.sum())
        else:
            count = len(_SPHERE)
        values[index] = 4.0 * math.pi * radii[index] ** 2 * count / len(_SPHERE)
    return values


def _sasa_split(atoms: list[dict[str, object]]) -> tuple[float, float, float]:
    values = _sasa(atoms)
    total = float(values.sum())
    polar = float(sum(value for value, atom in zip(values, atoms) if atom["element"] in POLAR_ELEMENTS))
    carbon = float(sum(value for value, atom in zip(values, atoms) if atom["element"] == "C"))
    return total, polar, carbon


def buried_surface(atoms_a, atoms_b):
    sasa_a, polar_a, carbon_a = _sasa_split(atoms_a)
    sasa_b, polar_b, carbon_b = _sasa_split(atoms_b)
    combined = atoms_a + atoms_b
    combined_values = _sasa(combined)
    split = len(atoms_a)
    total_combined = float(combined_values.sum())
    polar_combined = float(
        sum(value for value, atom in zip(combined_values, combined) if atom["element"] in POLAR_ELEMENTS)
    )
    carbon_combined = float(
        sum(value for value, atom in zip(combined_values, combined) if atom["element"] == "C")
    )
    bsa = sasa_a + sasa_b - total_combined
    bsa_polar = polar_a + polar_b - polar_combined
    bsa_carbon = carbon_a + carbon_b - carbon_combined
    buried_a = sasa_a - float(combined_values[:split].sum())
    buried_b = sasa_b - float(combined_values[split:].sum())
    return {
        "BSA_A2": round(bsa, 1),
        "BSA_polar_A2": round(bsa_polar, 1),
        "BSA_hydrophobic_A2": round(bsa_carbon, 1),
        "pct_hydrophobic_BSA": round(100.0 * bsa_carbon / bsa, 1) if bsa > 0 else "",
        "burial_asymmetry": (
            round(abs(buried_a - buried_b) / (buried_a + buried_b), 3)
            if buried_a + buried_b > 0
            else ""
        ),
    }

