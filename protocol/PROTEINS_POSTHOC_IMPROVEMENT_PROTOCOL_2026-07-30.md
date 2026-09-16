# PROTEINS post hoc improvement analysis protocol

Date frozen: July 30, 2026

## Purpose

This analysis evaluates the stability of the completed DC benchmark result
under protein-family/domain annotation separation and describes structural
quality covariates. It is a post hoc sensitivity analysis designed to
strengthen the evaluation-control contribution for *PROTEINS*.

## Immutable inputs

- The 1,865-row Many development partition remains the training source.
- The 80-row frozen DC panel and its labels remain unchanged.
- The five contact-derived features, preprocessing pipeline, logistic
  regression specification, and headline locked predictions remain unchanged.
- The saved locked outputs retain their original bytes.

## Annotation source

RCSB PDB Data API annotations will be collected for the exact polymer entities
in the Many development and DC panels. Family/domain identifiers will be drawn
from Pfam, CATH, ECOD, SCOP, and SCOP2 annotations returned by RCSB. The
retrieval cache and query will be preserved with the outputs.

## Primary sensitivity calculation

For each DC interface with at least one family/domain identifier:

1. collect the union of identifiers assigned to its two partner entities;
2. remove Many development interfaces sharing any collected identifier;
3. fit the unchanged normalized-contact pipeline to the remaining Many rows;
4. predict that DC interface once; and
5. aggregate the row-specific predictions across the annotated DC subset.

The procedure uses structural annotations to define training exclusions. DC
outcomes and locked prediction errors play no role in exclusion, fitting, or
feature selection.

Reported quantities are annotation coverage, retained training counts, ROC
AUC, average precision, Brier score, and a dependency-component bootstrap
comparison with the original frozen score on the same annotated rows.

## Quality-covariate audit

For the DC panel, the analysis will summarize experimental resolution, initial
release year, and chain lengths by published class and their association with
the frozen contact score. These descriptive results will identify any
structure-quality imbalance that merits discussion.

## Interpretation

The analysis is explicitly post hoc. It supplements the locked confirmation
and cannot replace, modify, or tune that result.

