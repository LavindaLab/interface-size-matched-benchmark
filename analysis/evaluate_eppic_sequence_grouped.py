#!/usr/bin/env python3
"""Sequence-grouped audit of the frozen EPPIC pre-grant panel."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_eppic_pregrant import FEATURE_SETS, number


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def component_groups(rows: list[dict[str, str]]) -> np.ndarray:
    uf = UnionFind()
    for row in rows:
        uf.union(row["sequence_cluster30_chain1"], row["sequence_cluster30_chain2"])
    return np.asarray([uf.find(row["sequence_cluster30_chain1"]) for row in rows], dtype=object)


def model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=0)),
        ]
    )


def grouped_bootstrap_delta(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    groups: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    group_to_rows = {group: np.flatnonzero(groups == group) for group in np.unique(groups)}
    group_names = np.asarray(list(group_to_rows), dtype=object)
    deltas = {name: [] for name in predictions if name != "area_only"}
    for _ in range(replicates):
        sampled = rng.choice(group_names, size=len(group_names), replace=True)
        selected = np.concatenate([group_to_rows[group] for group in sampled])
        if len(np.unique(y[selected])) < 2:
            continue
        area_auc = roc_auc_score(y[selected], predictions["area_only"][selected])
        for name, prediction in predictions.items():
            if name != "area_only":
                deltas[name].append(roc_auc_score(y[selected], prediction[selected]) - area_auc)
    output = {}
    for name, values in deltas.items():
        array = np.asarray(values)
        output[name] = {
            "delta_auc": float(roc_auc_score(y, predictions[name]) - roc_auc_score(y, predictions["area_only"])),
            "group_bootstrap_p02_5": float(np.percentile(array, 2.5)),
            "group_bootstrap_p97_5": float(np.percentile(array, 97.5)),
            "fraction_delta_le_zero": float(np.mean(array <= 0)),
        }
    return output


def evaluate_grouping(
    rows: list[dict[str, str]],
    y: np.ndarray,
    groups: np.ndarray,
    splits: int,
    repeats: int,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    matrices = {
        name: np.asarray([[number(row.get(feature)) for feature in features] for row in rows])
        for name, features in FEATURE_SETS.items()
    }
    sums = {name: np.zeros(len(rows)) for name in FEATURE_SETS}
    counts = np.zeros(len(rows), dtype=int)
    repeat_auc = {name: [] for name in FEATURE_SETS}
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed + repeat)
        repeat_predictions = {name: np.full(len(rows), np.nan) for name in FEATURE_SETS}
        for train, test in splitter.split(np.zeros(len(y)), y, groups):
            for name, X in matrices.items():
                repeat_predictions[name][test] = model().fit(X[train], y[train]).predict_proba(X[test])[:, 1]
            counts[test] += 1
        for name, prediction in repeat_predictions.items():
            if np.isnan(prediction).any():
                raise RuntimeError(f"incomplete prediction for repeat {repeat}, model {name}")
            repeat_auc[name].append(float(roc_auc_score(y, prediction)))
            sums[name] += prediction
    predictions = {name: values / counts for name, values in sums.items()}
    group_sizes = Counter(groups.tolist())
    return {
        "n_groups": len(group_sizes),
        "largest_group_rows": max(group_sizes.values()),
        "largest_group_fraction": max(group_sizes.values()) / len(rows),
        "models": {
            name: {
                "crossfit_averaged_auc": float(roc_auc_score(y, prediction)),
                "repeat_mean_auc": float(np.mean(repeat_auc[name])),
                "repeat_min_auc": float(np.min(repeat_auc[name])),
                "repeat_max_auc": float(np.max(repeat_auc[name])),
            }
            for name, prediction in predictions.items()
        },
        "paired_deltas_vs_area": grouped_bootstrap_delta(
            y, predictions, groups, bootstrap, seed
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    rows = read_tsv(args.input)
    y = np.asarray([row["label"] == "biological" for row in rows], dtype=int)
    groupings = {
        "sequence_cluster30_pair": np.asarray([row["sequence_cluster30_pair"] for row in rows], dtype=object),
        "strict_shared_cluster_component": component_groups(rows),
    }
    report = {
        "status": "exploratory sequence-grouped audit; no threshold selected",
        "cluster_source": "official RCSB MMseqs2 sequence_identity clusters at 30% cutoff",
        "n_rows": len(rows),
        "class_counts": dict(Counter(row["label"] for row in rows)),
        "splits": args.splits,
        "repeats": args.repeats,
        "groupings": {
            name: evaluate_grouping(rows, y, groups, args.splits, args.repeats, args.bootstrap, args.seed)
            for name, groups in groupings.items()
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
