#!/usr/bin/env python3
"""Evaluate the frozen EPPIC pre-grant panel with prespecified lightweight models.

This is an exploratory cross-fitted benchmark, not a locked external validation.  The principal
contrast is interface area alone versus area plus size-normalized chemical descriptors.  Current
EPPIC calls and scores are deliberately excluded from every model feature set.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_SETS = {
    "area_only": ["interface_area_A2"],
    "normalized_chemistry_only": [
        "contacts_per_100A2",
        "hbonds_per_100A2",
        "salt_bridges_per_100A2",
        "hydrophobic_contacts_per_100A2",
        "elec_complementarity",
    ],
    "area_plus_normalized_chemistry": [
        "interface_area_A2",
        "contacts_per_100A2",
        "hbonds_per_100A2",
        "salt_bridges_per_100A2",
        "hydrophobic_contacts_per_100A2",
        "elec_complementarity",
    ],
    "raw_chemistry_only": [
        "n_interface_contacts",
        "interchain_hbonds",
        "salt_bridges",
        "hydrophobic_contacts",
        "elec_complementarity",
    ],
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def number(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "", "NA", "nan") else np.nan
    except ValueError:
        return np.nan


def model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=0)),
        ]
    )


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, prediction)),
        "average_precision": float(average_precision_score(y, prediction)),
        "brier_score": float(brier_score_loss(y, prediction)),
    }


def paired_stratified_bootstrap(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    reference: str,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(y == value) for value in (0, 1)]
    samples: dict[str, list[float]] = {name: [] for name in predictions if name != reference}
    for _ in range(replicates):
        selected = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        rng.shuffle(selected)
        reference_auc = roc_auc_score(y[selected], predictions[reference][selected])
        for name, prediction in predictions.items():
            if name != reference:
                samples[name].append(roc_auc_score(y[selected], prediction[selected]) - reference_auc)

    result: dict[str, object] = {}
    for name, values in samples.items():
        array = np.asarray(values)
        result[name] = {
            "reference": reference,
            "delta_auc": float(
                roc_auc_score(y, predictions[name]) - roc_auc_score(y, predictions[reference])
            ),
            "bootstrap_p02_5": float(np.percentile(array, 2.5)),
            "bootstrap_p97_5": float(np.percentile(array, 97.5)),
            "fraction_delta_le_zero": float(np.mean(array <= 0)),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    rows = [row for row in read_tsv(args.input) if row.get("descriptor_status") == "ok"]
    y = np.asarray([1 if row.get("label") == "biological" else 0 for row in rows], dtype=int)
    splitter = RepeatedStratifiedKFold(
        n_splits=args.splits, n_repeats=args.repeats, random_state=args.seed
    )
    predictions = {name: np.zeros(len(rows), dtype=float) for name in FEATURE_SETS}
    prediction_counts = np.zeros(len(rows), dtype=int)
    repeat_aucs = {name: [] for name in FEATURE_SETS}
    fold_records: list[dict[str, object]] = []

    for split_number, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y)):
        repeat = split_number // args.splits
        fold = split_number % args.splits
        for name, features in FEATURE_SETS.items():
            X = np.asarray([[number(row.get(feature)) for feature in features] for row in rows])
            fitted = model().fit(X[train], y[train])
            fold_prediction = fitted.predict_proba(X[test])[:, 1]
            predictions[name][test] += fold_prediction
            fold_records.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "model": name,
                    "n_train": len(train),
                    "n_test": len(test),
                    "roc_auc": float(roc_auc_score(y[test], fold_prediction)),
                }
            )
        prediction_counts[test] += 1

    for name in predictions:
        predictions[name] /= prediction_counts
        for repeat in range(args.repeats):
            records = [
                record for record in fold_records
                if record["model"] == name and record["repeat"] == repeat
            ]
            repeat_aucs[name].append(float(np.mean([record["roc_auc"] for record in records])))

    report: dict[str, object] = {
        "status": "exploratory repeated cross-fitting on the frozen development panel; not locked external validation",
        "input": str(args.input),
        "n_rows": len(rows),
        "class_counts": dict(Counter(row.get("label", "") for row in rows)),
        "pdb_unique": len({row.get("pdb_id", "") for row in rows}),
        "feature_sets": FEATURE_SETS,
        "prohibited_features": ["eppic_current_call", "eppic_current_score", "eppic_current_confidence"],
        "cross_fitting": {"splits": args.splits, "repeats": args.repeats, "seed": args.seed},
        "models": {},
    }
    for name, prediction in predictions.items():
        values = np.asarray(repeat_aucs[name])
        report["models"][name] = {
            **metrics(y, prediction),
            "mean_fold_auc_by_repeat": float(values.mean()),
            "repeat_min": float(values.min()),
            "repeat_max": float(values.max()),
        }
    disagreement = np.asarray(
        [row.get("label_matches_current_eppic_call", "").lower() == "false" for row in rows]
    )
    report["current_eppic_disagreement_audit"] = {
        "note": (
            "post hoc difficulty audit only; current EPPIC calls are never model inputs and this "
            "subset is not an independent test set"
        ),
        "n_rows": int(disagreement.sum()),
        "class_counts": dict(Counter(row.get("label", "") for row, keep in zip(rows, disagreement) if keep)),
        "models": {
            name: metrics(y[disagreement], prediction[disagreement])
            for name, prediction in predictions.items()
        },
    }
    report["paired_auc_deltas_vs_area"] = paired_stratified_bootstrap(
        y, predictions, "area_only", args.bootstrap, args.seed
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "eppic_pregrant_evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (args.out_dir / "eppic_pregrant_crossfit_predictions.tsv").open("w", newline="") as handle:
        fields = ["pdb_id", "interface_serial", "label", *[f"p_{name}" for name in FEATURE_SETS]]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "pdb_id": row.get("pdb_id", ""),
                    "interface_serial": row.get("interface_serial", ""),
                    "label": row.get("label", ""),
                    **{f"p_{name}": predictions[name][index] for name in FEATURE_SETS},
                }
            )
    with (args.out_dir / "eppic_pregrant_fold_metrics.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fold_records[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(fold_records)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
