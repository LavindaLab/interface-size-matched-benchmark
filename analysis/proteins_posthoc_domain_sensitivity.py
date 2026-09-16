#!/usr/bin/env python3
"""Run the post hoc family/domain-separation and quality-covariate audit."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from develop_hard_case_models import feature_matrix, model_specifications, read_rows


ENDPOINT = "https://data.rcsb.org/graphql"
QUERY = """query($ids:[String!]!){polymer_entities(entity_ids:$ids){
  rcsb_id
  rcsb_polymer_entity_annotation{
    annotation_id type annotation_lineage{id name}
  }
  polymer_entity_instances{
    rcsb_id
    rcsb_polymer_instance_annotation{
      annotation_id type annotation_lineage{id name}
    }
  }
  entry{
    rcsb_id
    rcsb_accession_info{initial_release_date}
    rcsb_entry_info{resolution_combined}
  }
}}"""
FAMILY_TYPES = ("PFAM", "CATH", "ECOD", "SCOP")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> object:
    return json.loads(path.read_text())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_batch(entity_ids: list[str], attempts: int = 5) -> list[dict[str, object]]:
    payload = json.dumps({"query": QUERY, "variables": {"ids": entity_ids}}).encode()
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                ENDPOINT,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "EnergeticDialects-PROTEINS-sensitivity/1.0",
                },
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.load(response)
            if result.get("errors"):
                raise RuntimeError(result["errors"])
            return [item for item in result["data"]["polymer_entities"] if item]
        except Exception as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(f"RCSB annotation query failed: {last_error}")


def normalize_type(value: object) -> str:
    upper = str(value or "").upper().replace("-", "").replace("_", "")
    if upper.startswith("PFAM"):
        return "PFAM"
    if upper.startswith("CATH"):
        return "CATH"
    if upper.startswith("ECOD"):
        return "ECOD"
    if upper.startswith("SCOP"):
        return "SCOP"
    return ""


def annotation_tokens(annotation: dict[str, object]) -> set[str]:
    family_type = normalize_type(annotation.get("type"))
    if not family_type:
        return set()
    identifiers = [str(annotation.get("annotation_id") or "").strip()]
    lineage = annotation.get("annotation_lineage") or []
    if lineage:
        identifiers.append(str(lineage[-1].get("id") or "").strip())
    return {f"{family_type}:{identifier}" for identifier in identifiers if identifier}


def entity_record(entity: dict[str, object]) -> dict[str, object]:
    tokens: set[str] = set()
    type_counts: Counter[str] = Counter()
    annotations = list(entity.get("rcsb_polymer_entity_annotation") or [])
    for instance in entity.get("polymer_entity_instances") or []:
        annotations.extend(instance.get("rcsb_polymer_instance_annotation") or [])
    for annotation in annotations:
        found = annotation_tokens(annotation)
        tokens.update(found)
        for token in found:
            type_counts[token.split(":", 1)[0]] += 1
    entry = entity.get("entry") or {}
    entry_info = entry.get("rcsb_entry_info") or {}
    accession = entry.get("rcsb_accession_info") or {}
    resolutions = entry_info.get("resolution_combined") or []
    return {
        "tokens": sorted(tokens),
        "annotation_types": dict(type_counts),
        "resolution_angstrom": float(resolutions[0]) if resolutions else None,
        "initial_release_date": accession.get("initial_release_date"),
    }


def row_tokens(row: dict[str, str], records: dict[str, dict[str, object]]) -> set[str]:
    tokens: set[str] = set()
    for key in ("rcsb_entity1", "rcsb_entity2"):
        tokens.update(records.get(row.get(key, ""), {}).get("tokens", []))
    return tokens


def metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "brier_score": float(brier_score_loss(y, scores)),
    }


def dependency_bootstrap(
    y: np.ndarray,
    original: np.ndarray,
    sensitivity: np.ndarray,
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
        selected = np.concatenate([component_rows[name] for name in sampled])
        if len(np.unique(y[selected])) < 2:
            continue
        deltas.append(
            roc_auc_score(y[selected], sensitivity[selected])
            - roc_auc_score(y[selected], original[selected])
        )
    values = np.asarray(deltas)
    return {
        "valid_replicates": int(len(values)),
        "delta_auc_sensitivity_minus_original": float(
            roc_auc_score(y, sensitivity) - roc_auc_score(y, original)
        ),
        "p02_5": float(np.percentile(values, 2.5)),
        "p97_5": float(np.percentile(values, 97.5)),
        "fraction_delta_le_zero": float(np.mean(values <= 0)),
    }


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def standardized_mean_difference(values: np.ndarray, y: np.ndarray) -> float | None:
    positive, negative = values[y == 1], values[y == 0]
    positive, negative = positive[np.isfinite(positive)], negative[np.isfinite(negative)]
    if len(positive) < 2 or len(negative) < 2:
        return None
    pooled = np.sqrt((positive.var(ddof=1) + negative.var(ddof=1)) / 2.0)
    return float((positive.mean() - negative.mean()) / pooled) if pooled else 0.0


def year_from_date(value: object) -> float:
    try:
        return float(str(value)[:4])
    except (TypeError, ValueError):
        return np.nan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--external-predictions", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--limit-entities", type=int, default=0)
    args = parser.parse_args()

    development = read_rows(args.development)
    external = read_rows(args.external_predictions)
    if len(development) != 1865 or len(external) != 80:
        raise RuntimeError("immutable panel row counts differ from 1,865 development and 80 DC")
    entity_ids = sorted(
        {
            row[key]
            for row in development + external
            for key in ("rcsb_entity1", "rcsb_entity2")
            if row.get(key)
        }
    )
    if args.limit_entities:
        entity_ids = entity_ids[: args.limit_entities]
    if args.cache.exists():
        raw_entities = read_json(args.cache)
    else:
        raw_entities = []
        for start in range(0, len(entity_ids), args.batch_size):
            batch = entity_ids[start : start + args.batch_size]
            raw_entities.extend(fetch_batch(batch))
            print(f"[{min(start + len(batch), len(entity_ids))}/{len(entity_ids)}] entities", flush=True)
        write_json(args.cache, raw_entities)
    records = {
        str(entity["rcsb_id"]): entity_record(entity)
        for entity in raw_entities
        if entity and entity.get("rcsb_id")
    }

    development_tokens = [row_tokens(row, records) for row in development]
    external_tokens = [row_tokens(row, records) for row in external]
    features, factory = model_specifications()["chemistry_only"]
    y_train = np.asarray([row["label"] == "biological" for row in development], dtype=int)
    y_external = np.asarray([row["label"] == "biological" for row in external], dtype=int)
    original = np.asarray([float(row["chemistry_rescue_probability"]) for row in external])
    sensitivity = np.full(len(external), np.nan)
    retained_counts = np.zeros(len(external), dtype=int)

    model_cache: dict[tuple[str, ...], tuple[object, list[int]]] = {}
    for index, tokens in enumerate(external_tokens):
        if not tokens:
            continue
        key = tuple(sorted(tokens))
        if key not in model_cache:
            kept = [
                row_index
                for row_index, candidate_tokens in enumerate(development_tokens)
                if tokens.isdisjoint(candidate_tokens)
            ]
            kept_y = y_train[kept]
            if len(kept) < 100 or len(np.unique(kept_y)) < 2:
                continue
            model = factory().fit(feature_matrix([development[item] for item in kept], features), kept_y)
            model_cache[key] = (model, kept)
        model, kept = model_cache[key]
        sensitivity[index] = model.predict_proba(feature_matrix([external[index]], features))[0, 1]
        retained_counts[index] = len(kept)

    mapped = np.isfinite(sensitivity)
    if mapped.sum() < 20 or len(np.unique(y_external[mapped])) < 2:
        raise RuntimeError(f"insufficient annotated DC coverage: {int(mapped.sum())} rows")

    output_rows: list[dict[str, object]] = []
    for index, row in enumerate(external):
        record = records.get(row.get("rcsb_entity1", ""), {})
        release_date = record.get("initial_release_date")
        output_rows.append(
            {
                "pdb_id": row["pdb_id"],
                "interface_serial": row["interface_serial"],
                "label": row["label"],
                "dependency_component": row["external_dependency_component"],
                "family_domain_tokens": ";".join(sorted(external_tokens[index])),
                "annotation_count": len(external_tokens[index]),
                "retained_development_rows": int(retained_counts[index]),
                "original_locked_score": float(original[index]),
                "family_purged_score": (
                    float(sensitivity[index]) if np.isfinite(sensitivity[index]) else ""
                ),
                "resolution_angstrom": record.get("resolution_angstrom"),
                "initial_release_date": release_date,
                "initial_release_year": (
                    int(year_from_date(release_date))
                    if np.isfinite(year_from_date(release_date))
                    else ""
                ),
                "chain_a_residues": row.get("chain_a_residues", ""),
                "chain_b_residues": row.get("chain_b_residues", ""),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "dc_family_domain_sensitivity.tsv"
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)

    mapped_y = y_external[mapped]
    mapped_original = original[mapped]
    mapped_sensitivity = sensitivity[mapped]
    mapped_components = np.asarray(
        [row["external_dependency_component"] for row in external], dtype=object
    )[mapped]
    coverage_types = Counter(
        token.split(":", 1)[0] for tokens in external_tokens for token in tokens
    )
    resolution = np.asarray(
        [
            as_float(records.get(row.get("rcsb_entity1", ""), {}).get("resolution_angstrom"))
            for row in external
        ]
    )
    release_year = np.asarray(
        [
            year_from_date(
                records.get(row.get("rcsb_entity1", ""), {}).get("initial_release_date")
            )
            for row in external
        ]
    )
    chain_length = np.asarray(
        [
            (as_float(row.get("chain_a_residues")) + as_float(row.get("chain_b_residues"))) / 2.0
            for row in external
        ]
    )

    quality: dict[str, object] = {}
    for name, values in (
        ("resolution_angstrom", resolution),
        ("initial_release_year", release_year),
        ("mean_chain_residues", chain_length),
    ):
        finite = np.isfinite(values)
        correlation = spearmanr(values[finite], original[finite]) if finite.sum() >= 3 else None
        quality[name] = {
            "mapped_rows": int(finite.sum()),
            "biological_mean": float(np.nanmean(values[y_external == 1])),
            "crystal_mean": float(np.nanmean(values[y_external == 0])),
            "class_standardized_mean_difference_biological_minus_crystal": (
                standardized_mean_difference(values, y_external)
            ),
            "spearman_with_frozen_contact_score": (
                float(correlation.statistic) if correlation is not None else None
            ),
            "spearman_pvalue": float(correlation.pvalue) if correlation is not None else None,
        }

    audit = {
        "status": "POST_HOC_PROTEINS_SENSITIVITY_COMPLETE",
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_sha256": file_sha256(args.protocol),
        "annotation_source": ENDPOINT,
        "graphql_query": QUERY,
        "inputs": {
            "development": str(args.development),
            "development_sha256": file_sha256(args.development),
            "external_predictions": str(args.external_predictions),
            "external_predictions_sha256": file_sha256(args.external_predictions),
            "development_rows": len(development),
            "external_rows": len(external),
        },
        "annotation_coverage": {
            "requested_entities": len(entity_ids),
            "returned_entities": len(records),
            "dc_rows_with_family_domain_annotation": int(mapped.sum()),
            "dc_rows_total": len(external),
            "token_counts_by_type": dict(coverage_types),
        },
        "family_domain_purged_sensitivity": {
            "annotated_rows": int(mapped.sum()),
            "positive": int(mapped_y.sum()),
            "negative": int(len(mapped_y) - mapped_y.sum()),
            "retained_development_rows_min": int(retained_counts[mapped].min()),
            "retained_development_rows_median": float(np.median(retained_counts[mapped])),
            "retained_development_rows_max": int(retained_counts[mapped].max()),
            "original_score_same_rows": metrics(mapped_y, mapped_original),
            "family_purged_score": metrics(mapped_y, mapped_sensitivity),
            "dependency_bootstrap": dependency_bootstrap(
                mapped_y,
                mapped_original,
                mapped_sensitivity,
                mapped_components,
                args.bootstrap,
                args.seed,
            ),
        },
        "quality_covariates": quality,
        "interpretation_policy": (
            "Post hoc sensitivity analysis; immutable locked predictions remain the primary result."
        ),
    }
    audit_path = args.output_dir / "proteins_posthoc_domain_sensitivity_audit.json"
    write_json(audit_path, audit)

    sensitivity_metrics = audit["family_domain_purged_sensitivity"]
    report = f"""# PROTEINS post hoc improvement analysis

## Family/domain separation

RCSB supplied family/domain annotations for {int(mapped.sum())} of 80 DC
interfaces. Row-specific removal of annotation-sharing Many interfaces retained
a median of {sensitivity_metrics['retained_development_rows_median']:.0f}
development rows. On the annotated DC subset, the original frozen score reached
ROC AUC {sensitivity_metrics['original_score_same_rows']['roc_auc']:.3f}; the
family/domain-purged refits reached ROC AUC
{sensitivity_metrics['family_purged_score']['roc_auc']:.3f}.

The dependency-bootstrap difference for the sensitivity score relative to the
original score was
{sensitivity_metrics['dependency_bootstrap']['delta_auc_sensitivity_minus_original']:+.3f}
with a 95% interval from
{sensitivity_metrics['dependency_bootstrap']['p02_5']:+.3f} to
{sensitivity_metrics['dependency_bootstrap']['p97_5']:+.3f}.

## Structural quality covariates

The machine-readable audit reports class balance and score associations for
experimental resolution, initial release year, and chain length.

## Status

This is a post hoc sensitivity analysis. The frozen 80-interface result remains
the primary confirmation.
"""
    (args.output_dir / "PROTEINS_POSTHOC_IMPROVEMENT_RESULTS.md").write_text(report)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

