# Architecture

HeartShift is organized around an evidence pipeline rather than a notebook
pipeline. Each transition has an explicit contract, and no model-selection
component can read a locked outer label.

```text
official source -> raw manifest -> create-only canonicalization -> split contract
       -> source-only selection/freeze -> locked prediction rows
       -> independent aggregation/audit -> report sidecar -> release gate
```

## Trust boundaries

| Boundary | Enforced invariant |
| --- | --- |
| Acquisition | HTTPS is opt-in; expected bytes and SHA-256 are fixed before download; an existing path is verified and never replaced. |
| Canonicalization | Raw evidence is read-only input. Every derived destination is preflighted and published create-only. |
| Partitioning | Hospital/source identity is retained for splitting and audit but excluded from disease-prediction features. |
| Selection | Preprocessing, model choice, early stopping, calibration, thresholds, and auditors use source folds only. |
| Adaptation | Zero-shot, unlabelled-target, and labelled-target tracks are separate. Target labels never enter an unlabelled track. |
| Reporting | Aggregates are reconstructed from immutable rows keyed by `sample_id`; report sidecars bind every input and headline table by hash. |
| Release | The exact Git commit is scanned. A final inventory is impossible until an independently completed remote-CI attestation passes. |

## Package map

| Module | Responsibility |
| --- | --- |
| `heartshift.data` | Source parsing, deterministic canonicalization, split construction, and manifest-driven acquisition. |
| `heartshift.models` | Classical comparators, observed-set encoders, PS-MaskDRO, ShiftGuard, and the support-aware router. |
| `heartshift.evaluation` | Inner selection and fixed outer execution with prediction-level retention. |
| `heartshift.reporting` | Estimands, uncertainty, subgroup summaries, and independent reconstruction. |
| `heartshift.contracts` | Strict JSON loading, Draft 2020-12 validation, canonical serialization, and record self-hashes. |
| `heartshift.registry` | Typed method capabilities, versions, licences, provenance, devices, and evidence eligibility. |
| `heartshift.release` | Candidate-tree scanning, gate bindings, two-stage attestations, and full-tree inventories. |

The unified `heartshift` command exposes contract, registry, data, validation,
and release operations. Historical entry points remain available so preserved
run commands continue to resolve.

## Extension contract

A new benchmark method is admissible only when all of the following are true:

1. it has a unique entry in `configs/research/method_registry_v3.yaml`;
2. its exact implementation version and licence are recorded;
3. its target-information capability is declared before execution;
4. it emits the versioned prediction-table columns and immutable `sample_id`;
5. all tuning occurs inside source-only folds;
6. unavailable dependencies or checkpoints cause a visible omission, not a
   substitute method with the same name; and
7. its report states whether the evidence is locked, post-outcome,
   development-only, or smoke-only.

A new dataset must first receive a schema-valid manifest under
`manifests/datasets/`, including the official release, licence, redistribution
policy, acquisition mode, storage path, byte count, and SHA-256. Protected data
remain external and terms-gated; a public demo cannot be promoted to external
validation evidence.

## Failure behavior

Contract violations, duplicate YAML or JSON keys, ambiguous YAML aliases,
checksum mismatches, path collisions, existing output paths, unavailable
credentials, and incomplete gates fail closed. Partial downloads and historical
failed runs remain visible. The system does not silently retry a consumed outer
evaluation, rewrite an evidence file, or infer that missing evidence passed.

## Version boundary

The Python wheel contains software and type information only. Raw data, run
evidence, and results remain outside the distribution archive. Git and Git LFS
carry the repository evidence; a release inventory records both Git blob IDs and
LFS object identifiers for one exact candidate commit.
