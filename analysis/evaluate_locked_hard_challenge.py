#!/usr/bin/env python3
"""Fit on development and open the prospectively locked hard challenge exactly once."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from develop_hard_case_models import feature_matrix, model_specifications, read_rows


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
    }


def group_bootstrap(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    groups: np.ndarray,
    reference: str,
    selected: str,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    group_rows = {group: np.flatnonzero(groups == group) for group in np.unique(groups)}
    names = np.asarray(list(group_rows), dtype=object)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(replicates):
        sampled = rng.choice(names, size=len(names), replace=True)
        chosen = np.concatenate([group_rows[group] for group in sampled])
        if len(np.unique(y[chosen])) < 2:
            continue
        deltas.append(
            roc_auc_score(y[chosen], predictions[selected][chosen])
            - roc_auc_score(y[chosen], predictions[reference][chosen])
        )
    array = np.asarray(deltas)
    return {
        "valid_replicates": int(len(array)),
        "delta_auc": float(roc_auc_score(y, predictions[selected]) - roc_auc_score(y, predictions[reference])),
        "group_bootstrap_p02_5": float(np.percentile(array, 2.5)),
        "group_bootstrap_p97_5": float(np.percentile(array, 97.5)),
        "fraction_delta_le_zero": float(np.mean(array <= 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--lock-audit", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    development = read_rows(args.development)
    challenge = read_rows(args.challenge)
    lock = json.loads(args.lock_audit.read_text())
    selection = json.loads(args.selection.read_text())
    selected = selection["selected"]
    if not selected.startswith("local_"):
        raise SystemExit(f"refusing non-interface-local selection: {selected}")

    development_groups = {row["sequence_cluster30_component"] for row in development}
    challenge_groups = {row["sequence_cluster30_component"] for row in challenge}
    if development_groups & challenge_groups:
        raise SystemExit("sequence-family components overlap development and challenge")
    hard_rows = [row for row in challenge if row["hard_disagreement"] == "true"]
    key_text = "\n".join(
        sorted(f"{row['pdb_id']}\t{row['interface_serial']}\t{row['label']}" for row in hard_rows)
    ) + "\n"
    observed_hash = hashlib.sha256(key_text.encode()).hexdigest()
    if observed_hash != lock["challenge_hard_key_sha256"]:
        raise SystemExit("locked hard-challenge hash mismatch")

    specs = model_specifications()
    names = ["area_only", "area_plus_normalized", selected]
    y_train = np.asarray([row["label"] == "biological" for row in development], dtype=int)
    y = np.asarray([row["label"] == "biological" for row in challenge], dtype=int)
    hard = np.asarray([row["hard_disagreement"] == "true" for row in challenge], dtype=bool)
    groups = np.asarray([row["sequence_cluster30_component"] for row in challenge], dtype=object)
    predictions: dict[str, np.ndarray] = {}
    for name in names:
        features, factory = specs[name]
        fitted = factory().fit(feature_matrix(development, features), y_train)
        predictions[name] = fitted.predict_proba(feature_matrix(challenge, features))[:, 1]

    hard_metrics = {name: metrics(y[hard], prediction[hard]) for name, prediction in predictions.items()}
    all_metrics = {name: metrics(y, prediction) for name, prediction in predictions.items()}
    reference = "area_plus_normalized"
    delta = group_bootstrap(
        y[hard], {name: p[hard] for name, p in predictions.items()}, groups[hard],
        reference, selected, args.bootstrap, args.seed,
    )
    hard_homomer = hard & np.asarray([row["chain1"] == row["chain2"] for row in challenge])
    composition = {
        "all_hard": dict(Counter(row["label"] for row, keep in zip(challenge, hard) if keep)),
        "homomeric_hard": dict(Counter(row["label"] for row, keep in zip(challenge, hard_homomer) if keep)),
        "heteromeric_hard": dict(Counter(
            row["label"] for row, is_hard, is_homomer in zip(
                challenge, hard, [row["chain1"] == row["chain2"] for row in challenge]
            ) if is_hard and not is_homomer
        )),
    }
    homomer_metrics = {}
    if len(np.unique(y[hard_homomer])) == 2:
        homomer_metrics = {
            name: metrics(y[hard_homomer], prediction[hard_homomer])
            for name, prediction in predictions.items()
        }

    hard_success = (
        hard_metrics[selected]["roc_auc"] >= 0.65
        and delta["delta_auc"] >= 0.10
        and delta["group_bootstrap_p02_5"] > 0
    )
    overall_noninferior = (
        all_metrics[selected]["roc_auc"] >= all_metrics[reference]["roc_auc"] - 0.01
    )
    report = {
        "status": "locked challenge opened once after model selection and robustness gates",
        "lock_hash_verified": True,
        "development_rows": len(development),
        "challenge_component_rows": len(challenge),
        "challenge_hard_rows": int(hard.sum()),
        "development_components": len(development_groups),
        "challenge_components": len(challenge_groups),
        "component_overlap": 0,
        "selected": selected,
        "models_all_challenge_component_rows": all_metrics,
        "models_hard_challenge": hard_metrics,
        "hard_composition": composition,
        "models_hard_homomeric_when_two_classes_present": homomer_metrics,
        "selected_delta_vs_area_plus_normalized_hard": delta,
        "success_rule": {
            "hard_selected_auc_at_least": 0.65,
            "hard_delta_at_least": 0.10,
            "group_bootstrap_lower_bound_above_zero": True,
            "all_challenge_auc_noninferiority_margin": -0.01,
        },
        "hard_success": hard_success,
        "overall_noninferior": overall_noninferior,
        "passed": hard_success and overall_noninferior,
        "interpretation_guardrail": (
            "challenge membership is defined by disagreement with current EPPIC calls; EPPIC fields are not "
            "features. XtalMany is homomer-only, so all-case performance cannot be generalized to arbitrary "
            "heteromeric crystal contacts, and subgroup composition must accompany the headline result."
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    fields = [
        "pdb_id", "interface_serial", "label", "hard_disagreement",
        "sequence_cluster30_component", "homomeric", *[f"p_{name}" for name in names],
    ]
    with args.predictions.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index, row in enumerate(challenge):
            writer.writerow(
                {
                    "pdb_id": row["pdb_id"],
                    "interface_serial": row["interface_serial"],
                    "label": row["label"],
                    "hard_disagreement": row["hard_disagreement"],
                    "sequence_cluster30_component": row["sequence_cluster30_component"],
                    "homomeric": str(row["chain1"] == row["chain2"]).lower(),
                    **{f"p_{name}": predictions[name][index] for name in names},
                }
            )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
