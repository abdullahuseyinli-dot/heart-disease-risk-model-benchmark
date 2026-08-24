# Locked outer evidence audit

`heartshift-audit-outer` performs a post-run reconstruction from immutable
sample-level predictions. It does not fit a model or choose a method.

For heart classical and modern runs, the audit verifies:

- the endpoint is absent from both unlabelled ensemble and seed artifacts;
- labelled artifacts equal their endpoint-free predecessors after only the
  canonical endpoint column is removed;
- site, record fingerprint, and binary endpoint equal the canonical table;
- every score is finite and bounded, prediction keys are unique, configured
  seeds are complete, and the three-seed ensemble reconstructs exactly;
- every stored metric reconstructs with the repository metric implementation.

For the readmission run, it additionally reconstructs each of nine seed shards,
checks the top-level concatenation, and proves refit/test patient sets are
disjoint. The only permitted representation normalization is an all-missing
`epochs` column encoded as object `None` in classical shards and floating `NaN`
after heterogeneous concatenation.

For neural heart runs, the audit additionally verifies source-calibration patients
exclude the held-out hospital, calibration rows are patient-unique per policy,
research-only UDA scores are separated from deployment-gated scores, rejected
cells expose no deployment score, and all reported DG/UDA metrics reconstruct.

The audit writes `independent_validation.json` followed by `evidence_audit.json`.
The latter hashes every other file in the run recursively. A recovery or release
lock validates both the audit hash and the complete nested artifact tree; missing,
modified, or newly inserted run files therefore invalidate the lock.
