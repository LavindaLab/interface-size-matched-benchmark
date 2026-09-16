#!/usr/bin/env python3
"""Post hoc same-panel evaluation of the published PRODIGY-cryst classifier.

This analysis is intentionally separate from the locked DC confirmation. It
uses the unchanged classifier distributed by the official
``haddocking/interface-classifier`` repository and evaluates it on the exact
80-interface DC panel. The compatibility unpickler reads the scikit-learn 0.22
random-forest state without changing its tree nodes or predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


FEATURE_ORDER = [
    "CP",
    "AC",
    "AP",
    "AA",
    "ALA",
    "CYS",
    "GLU",
    "ASP",
    "GLY",
    "PHE",
    "ILE",
    "HIS",
    "MET",
    "LEU",
    "GLN",
    "PRO",
    "SER",
    "ARG",
    "THR",
    "VAL",
    "TYR",
]


class LegacyState:
    """Minimal state container for legacy scikit-learn pickle objects."""

    def __new__(cls, *args: object, **kwargs: object) -> "LegacyState":
        return super().__new__(cls)

    def __setstate__(self, state: object) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["state"] = state


class CompatibilityUnpickler(pickle.Unpickler):
    """Map legacy estimator classes to inert containers.

    Modern scikit-learn cannot directly load the 0.22 tree-node dtype because
    newer versions added a missing-value field. The stored node arrays are
    sufficient to reproduce traversal exactly, so no estimator refit or model
    conversion is performed.
    """

    LEGACY_CLASSES = {
        ("sklearn.ensemble.forest", "RandomForestClassifier"),
        ("sklearn.tree.tree", "DecisionTreeClassifier"),
        ("sklearn.tree._tree", "Tree"),
    }

    def find_class(self, module: str, name: str) -> object:
        if (module, name) in self.LEGACY_CLASSES:
            return LegacyState
        return super().find_class(module, name)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_probability(tree: LegacyState, features: np.ndarray) -> np.ndarray:
    node = 0
    while int(tree.nodes["left_child"][node]) != -1:
        feature = int(tree.nodes["feature"][node])
        threshold = float(tree.nodes["threshold"][node])
        branch = "left_child" if features[feature] <= threshold else "right_child"
        node = int(tree.nodes[branch][node])
    values = np.asarray(tree.values[node, 0], dtype=float)
    return values / values.sum()


def forest_probability(model: LegacyState, features: np.ndarray) -> np.ndarray:
    return np.mean(
        [tree_probability(estimator.tree_, features) for estimator in model.estimators_],
        axis=0,
    )


def residue_contacts(
    structure: object,
    chain_a: str,
    chain_b: str,
    is_aa: object,
    cutoff: float = 5.0,
) -> list[tuple[object, object]]:
    def protein_atoms(chain: object) -> list[object]:
        return [
            atom
            for residue in chain.get_residues()
            if is_aa(residue, standard=True)
            for atom in residue.get_atoms()
            if atom.element != "H"
        ]

    atoms_a = protein_atoms(structure[0][chain_a])
    atoms_b = protein_atoms(structure[0][chain_b])
    neighbors = cKDTree(np.asarray([atom.coord for atom in atoms_a])).query_ball_tree(
        cKDTree(np.asarray([atom.coord for atom in atoms_b])), cutoff
    )
    contacts: list[tuple[object, object]] = []
    seen: set[tuple[object, object]] = set()
    for index_a, indices_b in enumerate(neighbors):
        residue_a = atoms_a[index_a].parent
        for index_b in indices_b:
            residue_b = atoms_b[index_b].parent
            key = (residue_a.full_id, residue_b.full_id)
            if key not in seen:
                seen.add(key)
                contacts.append((residue_a, residue_b))
    if not contacts:
        raise RuntimeError(f"no residue contacts for {chain_a}/{chain_b}")
    return contacts


def prodigy_features(
    structure: object,
    chain_a: str,
    chain_b: str,
    is_aa: object,
    analyse_contacts: object,
) -> tuple[np.ndarray, int, float]:
    contacts = residue_contacts(structure, chain_a, chain_b, is_aa)
    bins = analyse_contacts(contacts)
    left, right = zip(*contacts)
    max_contacts = len(set(left)) * len(set(right))
    link_density = len(contacts) / max_contacts
    values = np.asarray([float(bins[key]) for key in FEATURE_ORDER] + [link_density])
    return values, len(contacts), link_density


def metric_block(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    prediction = score >= 0.5
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "brier_score": float(brier_score_loss(y, score)),
        "accuracy_at_0_5": float(accuracy_score(y, prediction)),
        "sensitivity_at_0_5": float(np.mean(prediction[y == 1])),
        "specificity_at_0_5": float(np.mean(~prediction[y == 0])),
    }


def paired_component_bootstrap(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    components: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    component_rows = {
        name: np.flatnonzero(components == name) for name in np.unique(components)
    }
    names = np.asarray(list(component_rows), dtype=object)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(names, size=len(names), replace=True)
        chosen = np.concatenate([component_rows[name] for name in sampled])
        if len(np.unique(y[chosen])) < 2:
            continue
        deltas.append(
            float(roc_auc_score(y[chosen], score_a[chosen]))
            - float(roc_auc_score(y[chosen], score_b[chosen]))
        )
    values = np.asarray(deltas)
    return {
        "observed_delta_auc": float(roc_auc_score(y, score_a) - roc_auc_score(y, score_b)),
        "dependency_component_bootstrap_p02_5": float(np.percentile(values, 2.5)),
        "dependency_component_bootstrap_p97_5": float(np.percentile(values, 97.5)),
        "fraction_delta_le_zero": float(np.mean(values <= 0)),
        "valid_replicates": len(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--locked-predictions", type=Path, required=True)
    parser.add_argument("--coordinate-root", type=Path, required=True)
    parser.add_argument("--prodigy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--prodigy-commit", default="")
    args = parser.parse_args()

    sys.path.insert(0, str(args.prodigy_root / "src"))
    from Bio.PDB import MMCIFParser, PDBParser  # noqa: PLC0415
    from Bio.PDB.Polypeptide import is_aa  # noqa: PLC0415
    from prodigy_cryst.interface_classifier import analyse_contacts  # noqa: PLC0415

    model_path = args.prodigy_root / "src/prodigy_cryst/data/classifier.sav"
    with model_path.open("rb") as handle:
        model = CompatibilityUnpickler(handle).load()

    # Exact compatibility check against the output documented by the official
    # repository for its bundled 1PPE example.
    example_path = args.prodigy_root / "examples/1ppe.pdb"
    example = PDBParser(QUIET=True).get_structure("1ppe", example_path)
    example_features, example_contacts, example_density = prodigy_features(
        example, "E", "I", is_aa, analyse_contacts
    )
    example_probability = forest_probability(model, example_features)
    if (
        example_contacts != 71
        or round(example_density, 2) != 0.14
        or not np.allclose(example_probability, [0.804, 0.196], atol=1e-12)
    ):
        raise RuntimeError("official 1PPE example compatibility check failed")

    panel = read_tsv(args.panel)
    locked = read_tsv(args.locked_predictions)
    locked_by_key = {
        (row["pdb_id"].lower(), row["interface_serial"], row["label"]): row
        for row in locked
    }
    output_rows: list[dict[str, str]] = []
    for row in panel:
        coordinate = args.coordinate_root / row["coordinate_cache"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure = MMCIFParser(QUIET=True, auth_chains=False).get_structure(
                f"{row['pdb_id']}_{row['interface_serial']}", coordinate
            )
        features, contact_count, link_density = prodigy_features(
            structure,
            row["resolved_chain_a"],
            row["resolved_chain_b"],
            is_aa,
            analyse_contacts,
        )
        probability = forest_probability(model, features)
        key = (row["pdb_id"].lower(), row["interface_serial"], row["label"])
        if key not in locked_by_key:
            raise RuntimeError(f"locked prediction not found for {key}")
        locked_row = locked_by_key[key]
        output_rows.append(
            {
                "pdb_id": row["pdb_id"],
                "interface_serial": row["interface_serial"],
                "label": row["label"],
                "external_match_pair": row["external_match_pair"],
                "external_dependency_component": locked_row[
                    "external_dependency_component"
                ],
                "prodigy_cryst_biological_probability": f"{probability[0]:.10f}",
                "normalized_contact_probability": locked_row[
                    "chemistry_rescue_probability"
                ],
                "prodigy_residue_contacts": str(contact_count),
                "prodigy_link_density": f"{link_density:.10f}",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(output_rows[0]), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(output_rows)

    y = np.asarray([row["label"] == "biological" for row in output_rows], dtype=int)
    prodigy_score = np.asarray(
        [float(row["prodigy_cryst_biological_probability"]) for row in output_rows]
    )
    contact_score = np.asarray(
        [float(row["normalized_contact_probability"]) for row in output_rows]
    )
    components = np.asarray(
        [row["external_dependency_component"] for row in output_rows], dtype=object
    )
    audit = {
        "status": "POST_HOC_SAME_PANEL_COMPARATOR_NOT_PART_OF_LOCKED_CONFIRMATION",
        "panel_rows": len(output_rows),
        "prodigy_repository": "https://github.com/haddocking/interface-classifier",
        "prodigy_commit": args.prodigy_commit,
        "prodigy_model_sha256": sha256(model_path),
        "feature_policy": "unchanged published PRODIGY-cryst feature order and random-forest nodes",
        "parser_policy": (
            "Biopython mmCIF parser with label-asym chain identifiers; standard amino acids and "
            "non-hydrogen atoms; residue pairs within the classifier's effective 5.0-A cutoff"
        ),
        "compatibility_check_1ppe": {
            "contacts": example_contacts,
            "link_density": example_density,
            "biological_probability": float(example_probability[0]),
            "crystal_probability": float(example_probability[1]),
            "matches_official_readme": True,
        },
        "metrics": {
            "prodigy_cryst": metric_block(y, prodigy_score),
            "normalized_contact": metric_block(y, contact_score),
        },
        "normalized_contact_minus_prodigy_cryst": paired_component_bootstrap(
            y,
            contact_score,
            prodigy_score,
            components,
            args.bootstrap,
            args.seed,
        ),
        "score_pearson_correlation": float(np.corrcoef(contact_score, prodigy_score)[0, 1]),
        "interpretation": (
            "The lightweight normalized-contact score and the published contact classifier rank "
            "the exact area-matched panel similarly. This post hoc comparator supports benchmark "
            "interpretation and is not a new confirmatory test."
        ),
    }
    args.audit.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
