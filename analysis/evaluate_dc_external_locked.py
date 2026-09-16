#!/usr/bin/env python3
"""Run the once-only locked evaluation on the frozen DC external panel."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from develop_hard_case_models import feature_matrix, model_specifications, read_rows


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def dependency_components(rows: list[dict[str, str]]) -> np.ndarray:
    union = UnionFind(len(rows))
    for key in ("external_match_pair", "external_family_group"):
        members: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            members[row[key]].append(index)
        for indices in members.values():
            for index in indices[1:]:
                union.union(indices[0], index)
    roots, names = {}, []
    for index in range(len(rows)):
        root = union.find(index)
        if root not in roots:
            roots[root] = f"component_{len(roots) + 1:03d}"
        names.append(roots[root])
    return np.asarray(names, dtype=object)


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, prediction)),
        "average_precision": float(average_precision_score(y, prediction)),
        "brier_score": float(brier_score_loss(y, prediction)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--lock-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    lock = json.loads(args.lock_audit.read_text())
    if lock.get("status") != "FROZEN_BEFORE_DESCRIPTOR_COMPUTATION":
        raise RuntimeError("external panel lock is absent or invalid")
    development, external = read_rows(args.development), read_rows(args.external)
    if len(external) != lock["frozen_rows"]:
        raise RuntimeError("external descriptor row count differs from frozen lock")
    if Counter(row["label"] for row in external) != Counter({"biological": 40, "crystal": 40}):
        raise RuntimeError("external class counts differ from frozen panel")

    y_train = np.asarray([row["label"] == "biological" for row in development], dtype=int)
    y = np.asarray([row["label"] == "biological" for row in external], dtype=int)
    predictions = {}
    feature_sets = {}
    for name in ("area_only", "chemistry_only"):
        features, factory = model_specifications()[name]
        model = factory().fit(feature_matrix(development, features), y_train)
        predictions[name] = model.predict_proba(feature_matrix(external, features))[:, 1]
        feature_sets[name] = features

    components = dependency_components(external)
    component_rows = {name: np.flatnonzero(components == name) for name in np.unique(components)}
    component_names = np.asarray(list(component_rows), dtype=object)
    rng = np.random.default_rng(args.seed)
    deltas, chemistry_aucs, area_aucs = [], [], []
    for _ in range(args.bootstrap):
        sampled = rng.choice(component_names, size=len(component_names), replace=True)
        chosen = np.concatenate([component_rows[name] for name in sampled])
        if len(np.unique(y[chosen])) < 2:
            continue
        chemistry_auc = roc_auc_score(y[chosen], predictions["chemistry_only"][chosen])
        area_auc = roc_auc_score(y[chosen], predictions["area_only"][chosen])
        chemistry_aucs.append(chemistry_auc)
        area_aucs.append(area_auc)
        deltas.append(chemistry_auc - area_auc)
    delta_array = np.asarray(deltas)
    chemistry_array = np.asarray(chemistry_aucs)
    area_array = np.asarray(area_aucs)
    observed_delta = roc_auc_score(y, predictions["chemistry_only"]) - roc_auc_score(
        y, predictions["area_only"]
    )
    lower_delta, upper_delta = np.percentile(delta_array, [2.5, 97.5])
    chemistry_auc = roc_auc_score(y, predictions["chemistry_only"])
    success = chemistry_auc >= 0.65 and lower_delta > 0

    result = {
        "status": "LOCKED_EXTERNAL_EVALUATION_OPENED",
        "panel": {
            "rows": len(external),
            "positive": int(y.sum()),
            "negative": int(len(y) - y.sum()),
            "unique_pdbs": len({row["pdb_id"] for row in external}),
            "unique_families": len({row["external_family_group"] for row in external}),
            "bootstrap_dependency_components": len(component_names),
        },
        "training": {
            "rows": len(development),
            "positive": int(y_train.sum()),
            "negative": int(len(y_train) - y_train.sum()),
            "source": str(args.development),
        },
        "features": feature_sets,
        "metrics": {name: metrics(y, prediction) for name, prediction in predictions.items()},
        "primary_comparison": {
            "delta_auc_chemistry_minus_area": float(observed_delta),
            "dependency_component_bootstrap_p02_5": float(lower_delta),
            "dependency_component_bootstrap_p97_5": float(upper_delta),
            "fraction_delta_le_zero": float(np.mean(delta_array <= 0)),
            "valid_replicates": len(delta_array),
            "chemistry_auc_bootstrap_p02_5": float(np.percentile(chemistry_array, 2.5)),
            "chemistry_auc_bootstrap_p97_5": float(np.percentile(chemistry_array, 97.5)),
            "area_auc_bootstrap_p02_5": float(np.percentile(area_array, 2.5)),
            "area_auc_bootstrap_p97_5": float(np.percentile(area_array, 97.5)),
        },
        "locked_success_rule": lock["success_rule"],
        "success": bool(success),
        "interpretation": (
            "The pre-registered external success rule was met."
            if success
            else "The pre-registered external success rule was not met."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    for index, row in enumerate(external):
        row["external_dependency_component"] = str(components[index])
        row["area_prior_probability"] = f"{predictions['area_only'][index]:.10f}"
        row["chemistry_rescue_probability"] = f"{predictions['chemistry_only'][index]:.10f}"
    with args.predictions.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(external[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(external)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
