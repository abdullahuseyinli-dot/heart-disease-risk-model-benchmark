# HeartShift benchmark card

## Summary

HeartShift is an auditable clinical tabular benchmark for hospital,
measurement-policy, and prevalence shift. The primary task combines four
historical UCI Heart Disease cohorts and evaluates each hospital as an outer
target. Hospital identity is used for partitioning and auditing only.

The endpoint is angiographically defined disease status. The benchmark does not
estimate prospective population risk or establish clinical utility.

## Dataset and license

- Dataset: UCI Heart Disease, DOI `10.24432/C52P4X`.
- License: Creative Commons Attribution 4.0.
- Records: 920.
- Sites: Cleveland (303), Hungary (294), Switzerland (123), and VA Long Beach (200).
- Predictors: 13 clinical variables with their natural observed/missing state.
- Endpoint: `1[num > 0]`.
- Duplicate handling: exact duplicate clusters are recorded in the canonical
  profile and remain visible to split and sensitivity audits.

Raw-source hashes, transformations, site prevalence, and feature missingness are
recorded in [the data card](DATA_CARD.md) and under `data/`.

## Evaluation protocol

Each outer fold holds out one hospital. Development uses only the other three
hospitals, with nested leave-one-source-hospital-out selection. Fold-fitted
preprocessing, model selection, early stopping, calibration, and thresholds
cannot access the outer target endpoint.

Hospital identity:

- defines outer and inner partitions;
- defines environment-balanced losses and audits;
- is excluded from disease-prediction features.

The outer outcomes have been consumed. Existing reports can be reconstructed,
but the four hospitals cannot support another independently confirmatory model
selection cycle.

## Measurement-policy interface

The evaluation bank contains named natural, MCAR, MAR, empirical, and
whole-panel deletion policies. Every policy starts from the naturally observed
record and can only remove information. A hidden value is never revealed or
reconstructed from a placeholder.

Methods are compared on identical `sample_id`, policy, replicate, and mask
keys. Reports reject duplicate keys, incomplete method coverage, or mismatched
masks.

## Endpoints and uncertainty

The primary endpoint is macro hospital worst-policy balanced log loss:

1. compute balanced log loss for each hospital-policy cell;
2. take the worst policy within each hospital;
3. average the four hospital values.

Secondary endpoints include worst individual site-policy loss, natural-policy
AUROC, Brier score, calibration summaries, site-level results, and registered
paired contrasts. Record bootstrap resampling occurs within each observed
hospital. It quantifies uncertainty conditional on these four
hospitals and fitted models; it is not random-effects inference over future
hospitals.

## Evidence classes

- **Locked heart:** versioned source development, freeze records, one-time outer
  evaluation, preserved execution failures, and deterministic reconstruction.
- **Independent readmission:** patient-disjoint admission-source shift with a
  separate endpoint and domain definition.
- **Post-outcome sensitivity:** ten-seed, architecture, and routing analyses
  after the heart outcomes were known.
- **Synthetic development:** controlled mechanism tests with explicit pass/fail gates.
- **Pipeline smoke:** public eICU demo data used only to verify code and schema execution.
- **Archived v1:** the original holdout benchmark, retained for provenance.

Evidence classes are not interchangeable. In particular, post-outcome
sensitivity cannot be relabelled as locked confirmation.

## Intended use

- Comparing tabular predictors under a fixed hospital and missing-feature protocol.
- Auditing robustness/discrimination trade-offs.
- Reconstructing published aggregate tables from sample-level predictions.
- Testing source-only model-selection and abstention rules.
- Reusing the evaluator, mask bank, and evidence contracts on lawfully obtained data.

## Out-of-scope use

- Diagnosis, triage, treatment, or other patient-level decisions.
- Prospective cardiovascular risk estimation.
- Claims about a population of hospitals from four historical cohorts.
- Target-label tuning presented as source-only generalization.
- Treating synthetic MNAR experiments as unrestricted MNAR robustness.
- Treating the eICU demo smoke test as external validation.

## Maintenance

Changes to a dataset version, endpoint, split, policy bank, primary metric, or
target-access status define a new benchmark version. Existing raw data, locks,
predictions, failures, and reports remain immutable and retain their original
evidence class.
