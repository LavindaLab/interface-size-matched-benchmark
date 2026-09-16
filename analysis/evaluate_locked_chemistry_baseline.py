#!/usr/bin/env python3
"""Add the pre-existing chemistry-only baseline omitted from the first locked report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from develop_hard_case_models import feature_matrix, model_specifications, read_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    development, challenge = read_rows(args.development), read_rows(args.challenge)
    y_train = np.asarray([row["label"] == "biological" for row in development], dtype=int)
    hard = np.asarray([row["hard_disagreement"] == "true" for row in challenge], dtype=bool)
    homomer = np.asarray([row["chain1"] == row["chain2"] for row in challenge], dtype=bool)
    all_y = np.asarray([row["label"] == "biological" for row in challenge], dtype=int)
    y = all_y[hard]
    features, factory = model_specifications()["chemistry_only"]
    fitted = factory().fit(feature_matrix(development, features), y_train)
    prediction = fitted.predict_proba(feature_matrix(challenge, features))[:, 1][hard]
    reference_predictions = {}
    for reference in ("area_only", "area_plus_normalized"):
        ref_features, ref_factory = model_specifications()[reference]
        ref_fit = ref_factory().fit(feature_matrix(development, ref_features), y_train)
        reference_predictions[reference] = ref_fit.predict_proba(
            feature_matrix(challenge, ref_features)
        )[:, 1][hard]

    hard_groups = np.asarray(
        [row["sequence_cluster30_component"] for row, keep in zip(challenge, hard) if keep],
        dtype=object,
    )
    group_rows = {group: np.flatnonzero(hard_groups == group) for group in np.unique(hard_groups)}
    group_names = np.asarray(list(group_rows), dtype=object)
    rng = np.random.default_rng(args.seed)
    samples = {reference: [] for reference in reference_predictions}
    for _ in range(args.bootstrap):
        sampled = rng.choice(group_names, size=len(group_names), replace=True)
        chosen = np.concatenate([group_rows[group] for group in sampled])
        if len(np.unique(y[chosen])) < 2:
            continue
        chemistry_auc = roc_auc_score(y[chosen], prediction[chosen])
        for reference, ref_prediction in reference_predictions.items():
            samples[reference].append(chemistry_auc - roc_auc_score(y[chosen], ref_prediction[chosen]))

    deltas = {}
    for reference, values in samples.items():
        array = np.asarray(values)
        deltas[reference] = {
            "delta_auc": float(roc_auc_score(y, prediction) - roc_auc_score(y, reference_predictions[reference])),
            "group_bootstrap_p02_5": float(np.percentile(array, 2.5)),
            "group_bootstrap_p97_5": float(np.percentile(array, 97.5)),
            "fraction_delta_le_zero": float(np.mean(array <= 0)),
        }
    hard_homomer = homomer[hard]
    homomer_result = {}
    if len(np.unique(y[hard_homomer])) == 2:
        homomer_result = {
            "n": int(hard_homomer.sum()),
            "positive": int(y[hard_homomer].sum()),
            "negative": int(hard_homomer.sum() - y[hard_homomer].sum()),
            "roc_auc": float(roc_auc_score(y[hard_homomer], prediction[hard_homomer])),
            "average_precision": float(average_precision_score(y[hard_homomer], prediction[hard_homomer])),
            "brier_score": float(brier_score_loss(y[hard_homomer], prediction[hard_homomer])),
        }
    report = {
        "status": (
            "post-opening calculation of a model defined before the lock; omitted baseline only, "
            "not candidate selection or tuning"
        ),
        "model": "chemistry_only",
        "features": features,
        "n_hard": int(len(y)),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "roc_auc": float(roc_auc_score(y, prediction)),
        "average_precision": float(average_precision_score(y, prediction)),
        "brier_score": float(brier_score_loss(y, prediction)),
        "paired_group_bootstrap_deltas": deltas,
        "hard_homomeric_subset": homomer_result,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
