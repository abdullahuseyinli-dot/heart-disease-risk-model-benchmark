# HeartShift data card

## UCI Heart Disease case study

- Source: UCI Heart Disease, DOI `10.24432/C52P4X`, CC BY 4.0.
- Frozen raw archive SHA-256:
  `B17CD273DA9CE1CAA4710FCE80227EA454D4DBF9FCBC8E6A9121672751563ADC`.
- Canonical table: 920 records from Cleveland (303), Hungary (294),
  Switzerland (123), and VA Long Beach (200).
- Endpoint: `num > 0`, an angiographically defined disease-status label in
  historical referred cohorts. It is not prospective population risk.
- Unit: one source record. The public files do not provide a reliable cross-site
  patient identifier, so undetectable patient overlap cannot be ruled out.
- Predictors: the 13 conventional UCI Heart variables. Hospital identity is kept
  for splitting and auditing and is never a model input.
- Missingness: natural source missingness is preserved. Counterfactual policies
  may only hide an observed value; they can never reveal a naturally absent value.
- Split: deterministic nested leave-one-hospital-out. Every outer hospital is
  unavailable to preprocessing, fitting, early stopping, selection, calibration,
  and thresholding until the frozen one-time evaluation.

## UCI Diabetes 130-US-hospitals method-validation task

- Source: UCI Diabetes 130-US Hospitals, DOI `10.24432/C5230J`, CC BY 4.0.
- Frozen raw archive SHA-256:
  `F82AC129DA2DDD2299391FF6FBAE3A6A58B3EDCF59AC9D7BD480C00FE453112A`.
- Canonical table: 101,766 encounters from 71,518 patients.
- Primary compatibility endpoint: any recorded readmission (`readmitted != NO`),
  matching the audited TableShift implementation. This is distinct from 30-day
  readmission: 46,902 versus 11,357 positive encounters, respectively.
- Domain: emergency-room admission source (`admission_source_id == 7`) versus
  other admission sources. Admission source is an environment variable, not a
  prediction feature.
- Hardened split: patient-grouped source train/validation/ID test and target
  validation/OOD test. Patients spanning source and target domains are quarantined.
  All evaluated partitions are patient-disjoint.
- TableShift reproduction: an additional exact encounter-level reproduction is
  retained for audit. It has patient overlap (including 4,983 patients shared by
  train and OOD test), so it is not the confirmatory split.
- Interpretation: this is independent evidence for a general shift/missingness
  method on a different endpoint. It is not external validation of heart disease.

## Data handling and exclusions

Raw archives, extracted files, raw-line hashes, manifests, canonical hashes, split
tables, and quarantined records are retained. No row is deleted as cleanup. Values
are parsed deterministically; preprocessing statistics and category vocabularies
are fitted inside source folds. The datasets are historical and may encode
selection, access, documentation, treatment, sex, race, age, and institutional
biases. They are unsuitable for deployment decisions without contemporary local
validation, governance, and clinical review.
