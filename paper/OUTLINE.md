# Manuscript outline

## Working title

**HeartShift: Auditing hospital and measurement-policy shift in small clinical
tabular benchmarks**

The defensible contribution is an auditable benchmark and a set of empirical
findings. PS-MaskDRO, prior separation, ShiftGuard, and support-aware routing are
evaluated research hypotheses; the current evidence does not justify a general
new-architecture or clinical-validity claim.

## 1. Abstract

- Motivation: patient mix and recorded measurements change together across
  clinical sites.
- Design: four historical UCI heart cohorts, leave-one-hospital-out evaluation,
  nested source-only selection, deterministic measurement-deletion policies,
  modern and classical comparators, and immutable prediction-level evidence.
- Main findings: report the locked primary estimand and its uncertainty; state
  the negative superiority and routing results; distinguish the independent
  readmission task from heart validation.
- Limit: 920 referred-cohort records and four sites cannot establish prospective
  risk prediction, clinical utility, or inference to future hospitals.

## 2. Introduction

1. Explain why missingness can encode measurement policy rather than random
   noise.
2. Separate hospital population shift, measurement-policy shift, label shift,
   and concept shift.
3. Identify reproducibility problems addressed by HeartShift: target leakage,
   ambiguous evidence status, aggregate-only reporting, silent comparator
   substitutions, and mutable artifacts.
4. State contributions narrowly:
   - a versioned evaluation and evidence contract;
   - paired site-by-policy evaluation with source-only selection;
   - a broad, provenance-bound comparator inventory;
   - visible negative results and adaptation abstention; and
   - independent cross-task evidence plus explicit limits.

## 3. Related work

- Tabular distribution-shift benchmarks, especially TableShift.
- Learning and adaptation under missingness shift.
- Group robustness and proper-score evaluation.
- Tabular foundation models and trained-from-scratch neural comparators.
- Label-shift estimation, compatibility tests, and abstaining adaptation.

Use `references.bib` and the frozen `docs/LITERATURE_MATRIX.md`. Before
submission, rerun a documented systematic search and preserve query strings,
databases, dates, inclusion criteria, and screening decisions. Do not convert
the existing targeted matrix into an “exhaustive review” claim.

## 4. Data and endpoints

- UCI Heart: cohort provenance, 13 variables, natural missingness, zero
  sentinels, `num > 0` endpoint, 920 records, and site imbalance.
- UCI Diabetes Readmission: independent method-validation role; distinguish the
  TableShift-compatible any-readmission endpoint from the under-30-day
  sensitivity label.
- eICU demo: pipeline smoke test only.
- Licences, transformations, exclusions, and patient/site identity handling.

## 5. Evaluation protocol

- Outer leave-one-hospital-out and nested leave-one-source-hospital-out design.
- Deterministic mask bank and exact cross-method mask identity.
- Zero-shot, unlabelled-target, and labelled-target tracks as separate studies.
- Primary macro-site worst-policy balanced log loss; secondary Brier, AUROC,
  selective prediction, and degradation curves.
- Paired patient-within-site bootstrap and its conditional inferential scope.
- Freeze, one-opening rule, recovery history, and evidence classes.

## 6. Methods

- Classical reference models and calibration.
- Observed-set neural encoder and prior-separated objective.
- Site-by-mask DRO and MIRRAMS Equation 9 reproduction.
- TabPFN, TabICL, TabM, RealMLP, and FT-Transformer comparators.
- Label-shift baselines and compatibility-gated abstention.
- Support-aware routing and ShiftGuard, clearly labelled post-outcome or
  development-only where applicable.

Every method name, version, licence, device, target-information capability, and
fidelity status comes from `configs/research/method_registry_v3.yaml`.

## 7. Results

### 7.1 Locked heart benchmark

Generate all values from `artifacts/reports/heart-outer-v5/`. Lead with the
primary estimand, paired contrasts, hospital heterogeneity, and the failure of
the preselected candidate to establish superiority.

### 7.2 Independent readmission task

Generate from `artifacts/reports/readmission-outer-v3/`. Present robustness and
AUROC trade-offs without calling the admission-source split a hospital shift.

### 7.3 Development and sensitivity analyses

Generate from `artifacts/reports/heart-research-development-v1/` and preserve
the post-outcome status. Include ten-seed stability, routing non-superiority,
and multiplicity-aware intervals.

### 7.4 Adaptation and mechanism failures

Report complete abstention, negative-transfer prevention, synthetic MNAR and
concept-shift failures, and eICU demo smoke status. A safety gate that abstains
is not evidence of successful adaptation.

## 8. Discussion

- Interpret disagreement between proper scores and discrimination.
- Discuss why simple averaging was more stable than learned routing here.
- Explain which results are benchmark observations versus architecture
  hypotheses.
- State that untouched, adequately powered external hospitals are required for
  method confirmation.

## 9. Limitations and ethics

Use `docs/LIMITATIONS.md` and `docs/REPORTING_CHECKLIST.md`. Include referral
bias, temporal age, small hospital count, class imbalance, pretraining-overlap
uncertainty, subgroup power, endpoint limits, and non-clinical use.

## 10. Reproducibility statement

Reference the exact commit, lockfile, hardware manifest, dataset manifests,
method registry, prediction/metric contracts, report sidecars, failed-run
history, remote gate attestation, and release inventory. Add a DOI only after an
actual immutable deposit exists.
