# Area-matched protein-interface benchmark

Public reproducibility package for:

> Olga Lavinda. *Contact organization separates biological from crystal protein interfaces matched on area and disjoint in sequence.* ChemRxiv (2026). https://doi.org/10.26434/chemrxiv.15010687/v1

The ChemRxiv preprint is the current version of the manuscript. The manuscript files in this repository are the earlier version of the same study, posted under the title *When interface size misleads: an area-matched, sequence-cluster-disjoint benchmark for biological-versus-crystal interface classification*. The data, code and locked results are the same.

The repository contains the portable analysis scripts, exact derived descriptor tables, frozen panel and prediction records, grouped-development results, post hoc comparator records, software requirements, and manuscript files supporting the reported benchmark.

## Primary entry points

- `manuscript/Interface_Benchmark_bioRxiv_PREPRINT.md`: manuscript source, earlier version.
- `manuscript/When_Interface_Size_Misleads_bioRxiv_PREPRINT_2026-09-16.pdf`: manuscript PDF, earlier version.
- `data/dc_external_LOCKED_predictions.tsv`: exact locked DC predictions and descriptor rows.
- `results/dc_external_LOCKED_RESULT.json`: frozen primary result.
- `results/dc_external_panel_FROZEN_audit.json`: panel-freeze audit.
- `data/eppic_pregrant_descriptors_2000_clustered.tsv`: grouped-development input.
- `analysis/`: panel construction, descriptor, grouped evaluation, locked evaluation, comparator, and sensitivity scripts.
- `requirements-lock.txt`: exact clean-reproduction environment.

## Reproduction

From the repository root:

```bash
python3 -m pip install -r requirements-lock.txt
python3 analysis/evaluate_eppic_sequence_grouped.py \
  --input data/eppic_pregrant_descriptors_2000_clustered.tsv \
  --output reproduced_sequence_grouped_evaluation.json

python3 analysis/evaluate_dc_external_locked.py \
  --development data/hard_improvement_development.tsv \
  --external data/dc_external_panel_descriptors.tsv \
  --lock-audit results/dc_external_panel_FROZEN_audit.json \
  --output reproduced_dc_external_result.json \
  --predictions reproduced_dc_external_predictions.tsv
```

The immutable release corresponding to the preprint upload is tagged `v1.0.1-biorxiv`. File hashes are recorded in `SHA256SUMS.txt` and the release manifest.

## External source records

- Preserved EPPIC benchmark dataset commit: https://github.com/eppic-team/datasets/tree/8f0b00f2a82d3f796219f9ba41dc64acf13bd65f
- RCSB PDB data services: https://www.rcsb.org/

