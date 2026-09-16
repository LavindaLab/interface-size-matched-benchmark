#!/usr/bin/env python3
"""Develop hard-case models without reading the prospectively locked challenge table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


AREA_NORMALIZED = [
    "interface_area_A2",
    "contacts_per_100A2",
    "hbonds_per_100A2",
    "salt_bridges_per_100A2",
    "hydrophobic_contacts_per_100A2",
    "elec_complementarity",
]
CHEMISTRY_ONLY = AREA_NORMALIZED[1:]
EXPANDED = [
    "log_area",
    "contacts_per_100A2",
    "hbonds_per_100A2",
    "salt_bridges_per_100A2",
    "hydrophobic_contacts_per_100A2",
    "elec_complementarity",
    "log_interface_residues",
    "interface_residue_fraction",
    "area_per_interface_residue",
    "contacts_per_interface_residue",
    "hydrophobic_fraction_contacts",
    "hbond_fraction_contacts",
    "salt_fraction_contacts",
    "charge_pairs_per_100A2",
    "chain_log_geomean",
    "chain_length_asymmetry",
]
INTERFACE_LOCAL = [
    feature for feature in EXPANDED
    if feature not in {"interface_residue_fraction", "chain_log_geomean", "chain_length_asymmetry"}
]


def number(row: dict[str, str], key: str) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value not in ("", "NA", "nan") else np.nan
    except ValueError:
        return np.nan


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if np.isfinite(numerator) and np.isfinite(denominator) and denominator > 0 else np.nan


def derive(row: dict[str, str]) -> dict[str, float]:
    area = number(row, "interface_area_A2")
    interface_residues = number(row, "interface_residues")
    chain_a, chain_b = number(row, "chain_a_residues"), number(row, "chain_b_residues")
    contacts = number(row, "n_interface_contacts")
    hydrophobic = number(row, "hydrophobic_contacts")
    hbonds = number(row, "interchain_hbonds")
    salts = number(row, "salt_bridges")
    favorable = number(row, "elec_favorable_pairs")
    unfavorable = number(row, "elec_unfavorable_pairs")
    charge_pairs = favorable + unfavorable if np.isfinite(favorable) and np.isfinite(unfavorable) else np.nan
    return {
        "interface_area_A2": area,
        "contacts_per_100A2": number(row, "contacts_per_100A2"),
        "hbonds_per_100A2": number(row, "hbonds_per_100A2"),
        "salt_bridges_per_100A2": number(row, "salt_bridges_per_100A2"),
        "hydrophobic_contacts_per_100A2": number(row, "hydrophobic_contacts_per_100A2"),
        "elec_complementarity": number(row, "elec_complementarity"),
        "log_area": math.log1p(area) if np.isfinite(area) and area >= 0 else np.nan,
        "log_interface_residues": (
            math.log1p(interface_residues) if np.isfinite(interface_residues) and interface_residues >= 0 else np.nan
        ),
        "interface_residue_fraction": safe_ratio(interface_residues, chain_a + chain_b),
        "area_per_interface_residue": safe_ratio(area, interface_residues),
        "contacts_per_interface_residue": safe_ratio(contacts, interface_residues),
        "hydrophobic_fraction_contacts": safe_ratio(hydrophobic, contacts),
        "hbond_fraction_contacts": safe_ratio(hbonds, contacts),
        "salt_fraction_contacts": safe_ratio(salts, contacts),
        "charge_pairs_per_100A2": 100.0 * safe_ratio(charge_pairs, area),
        "chain_log_geomean": (
            0.5 * (math.log1p(chain_a) + math.log1p(chain_b))
            if np.isfinite(chain_a) and np.isfinite(chain_b) and chain_a >= 0 and chain_b >= 0
            else np.nan
        ),
        "chain_length_asymmetry": safe_ratio(abs(chain_a - chain_b), chain_a + chain_b),
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def feature_matrix(rows: list[dict[str, str]], features: list[str]) -> np.ndarray:
    derived = [derive(row) for row in rows]
    return np.asarray([[values[feature] for feature in features] for values in derived], dtype=float)


def model_specifications() -> dict[str, tuple[list[str], callable]]:
    linear = lambda: Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0, random_state=0)),
        ]
    )
    return {
        "area_only": (["interface_area_A2"], linear),
        "chemistry_only": (CHEMISTRY_ONLY, linear),
        "area_plus_normalized": (AREA_NORMALIZED, linear),
        "expanded_logistic": (EXPANDED, linear),
        "expanded_spline_logistic": (
            EXPANDED,
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler()),
                    ("spline", SplineTransformer(n_knots=4, degree=2, include_bias=False)),
                    ("model", LogisticRegression(max_iter=5000, class_weight="balanced", C=0.3, random_state=0)),
                ]
            ),
        ),
        "expanded_shallow_gbt": (
            EXPANDED,
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("model", GradientBoostingClassifier(
                        n_estimators=200, learning_rate=0.03, max_depth=2, min_samples_leaf=15,
                        subsample=0.8, random_state=0,
                    )),
                ]
            ),
        ),
        "expanded_hist_gbt": (
            EXPANDED,
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("model", HistGradientBoostingClassifier(
                        max_iter=200, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=20,
                        l2_regularization=1.0, random_state=0,
                    )),
                ]
            ),
        ),
        "expanded_extra_trees": (
            EXPANDED,
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("model", ExtraTreesClassifier(
                        n_estimators=400, min_samples_leaf=10, max_features=0.7,
                        class_weight="balanced", n_jobs=-1, random_state=0,
                    )),
                ]
            ),
        ),
        "local_expanded_logistic": (INTERFACE_LOCAL, linear),
        "local_expanded_shallow_gbt": (
            INTERFACE_LOCAL,
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("model", GradientBoostingClassifier(
                        n_estimators=200, learning_rate=0.03, max_depth=2, min_samples_leaf=15,
                        subsample=0.8, random_state=0,
                    )),
                ]
            ),
        ),
        "local_expanded_hist_gbt": (
            INTERFACE_LOCAL,
            lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("model", HistGradientBoostingClassifier(
                        max_iter=200, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=20,
                        l2_regularization=1.0, random_state=0,
                    )),
                ]
            ),
        ),
    }


def group_bootstrap_delta(
    y: np.ndarray,
    hard: np.ndarray,
    groups: np.ndarray,
    predictions: dict[str, np.ndarray],
    reference: str,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    indices = np.flatnonzero(hard)
    hard_groups = groups[indices]
    group_rows = {group: indices[hard_groups == group] for group in np.unique(hard_groups)}
    names = np.asarray(list(group_rows), dtype=object)
    rng = np.random.default_rng(seed)
    values = {name: [] for name in predictions if name != reference}
    for _ in range(replicates):
        sampled = rng.choice(names, size=len(names), replace=True)
        chosen = np.concatenate([group_rows[group] for group in sampled])
        if len(np.unique(y[chosen])) < 2:
            continue
        ref_auc = roc_auc_score(y[chosen], predictions[reference][chosen])
        for name, prediction in predictions.items():
            if name != reference:
                values[name].append(roc_auc_score(y[chosen], prediction[chosen]) - ref_auc)
    output = {}
    for name, samples in values.items():
        array = np.asarray(samples)
        output[name] = {
            "hard_delta_auc": float(
                roc_auc_score(y[hard], predictions[name][hard])
                - roc_auc_score(y[hard], predictions[reference][hard])
            ),
            "group_bootstrap_p02_5": float(np.percentile(array, 2.5)),
            "group_bootstrap_p97_5": float(np.percentile(array, 97.5)),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    rows = read_rows(args.development)
    y = np.asarray([row["label"] == "biological" for row in rows], dtype=int)
    hard = np.asarray([row["hard_disagreement"] == "true" for row in rows], dtype=bool)
    groups = np.asarray([row["sequence_cluster30_component"] for row in rows], dtype=object)
    specs = model_specifications()
    matrices = {name: feature_matrix(rows, features) for name, (features, _) in specs.items()}
    sums = {name: np.zeros(len(rows)) for name in specs}
    counts = np.zeros(len(rows), dtype=int)
    repeat_metrics = {name: [] for name in specs}

    for repeat in range(args.repeats):
        splitter = StratifiedGroupKFold(n_splits=args.splits, shuffle=True, random_state=args.seed + repeat)
        current = {name: np.full(len(rows), np.nan) for name in specs}
        for train, test in splitter.split(np.zeros(len(y)), y, groups):
            for name, (_, factory) in specs.items():
                current[name][test] = factory().fit(matrices[name][train], y[train]).predict_proba(matrices[name][test])[:, 1]
            counts[test] += 1
        for name, prediction in current.items():
            if np.isnan(prediction).any():
                raise RuntimeError(f"incomplete OOF predictions for {name}, repeat {repeat}")
            sums[name] += prediction
            repeat_metrics[name].append(
                {
                    "overall_auc": float(roc_auc_score(y, prediction)),
                    "hard_auc": float(roc_auc_score(y[hard], prediction[hard])),
                }
            )

    predictions = {name: total / counts for name, total in sums.items()}
    models = {}
    for name, prediction in predictions.items():
        models[name] = {
            "features": specs[name][0],
            "overall_auc": float(roc_auc_score(y, prediction)),
            "overall_brier": float(brier_score_loss(y, prediction)),
            "hard_auc": float(roc_auc_score(y[hard], prediction[hard])),
            "hard_brier": float(brier_score_loss(y[hard], prediction[hard])),
            "repeat_overall_auc_min": float(min(item["overall_auc"] for item in repeat_metrics[name])),
            "repeat_overall_auc_max": float(max(item["overall_auc"] for item in repeat_metrics[name])),
            "repeat_hard_auc_min": float(min(item["hard_auc"] for item in repeat_metrics[name])),
            "repeat_hard_auc_max": float(max(item["hard_auc"] for item in repeat_metrics[name])),
        }

    reference = "area_plus_normalized"
    minimum_overall = models[reference]["overall_auc"] - 0.005
    minimum_hard = models[reference]["hard_auc"] + 0.03
    eligible = [
        name for name in specs
        if name.startswith("local_")
        and models[name]["overall_auc"] >= minimum_overall
        and models[name]["hard_auc"] >= minimum_hard
    ]
    selected = max(eligible, key=lambda name: (models[name]["hard_auc"], models[name]["overall_auc"])) if eligible else ""
    deltas = group_bootstrap_delta(y, hard, groups, predictions, reference, args.bootstrap, args.seed)
    report = {
        "status": "development only; locked challenge was not read",
        "n_rows": len(rows),
        "class_counts": dict(Counter(row["label"] for row in rows)),
        "hard_rows": int(hard.sum()),
        "hard_class_counts": dict(Counter(row["label"] for row, keep in zip(rows, hard) if keep)),
        "group_count": len(np.unique(groups)),
        "prohibited_feature_policy": (
            "no EPPIC prediction/call/confidence, disagreement flag, symmetry/operator, infinite, or isologous field is a feature"
        ),
        "models": models,
        "hard_deltas_vs_area_plus_normalized": deltas,
        "selection_rule": {
            "reference": reference,
            "overall_auc_floor": minimum_overall,
            "hard_auc_floor": minimum_hard,
            "rule": (
                "protocol amendment before opening challenge: whole-chain size/asymmetry candidates are shortcut "
                "sensitivity analyses only because XtalMany is entirely homomeric; among interface-local candidates "
                "meeting both floors, select highest development hard-case AUC; break ties by overall AUC; if none "
                "qualify, do not open challenge"
            ),
            "eligible": eligible,
            "selected": selected,
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.selection.write_text(json.dumps({"selected": selected, "features": specs[selected][0] if selected else []}, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
