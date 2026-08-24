# Contributing

HeartShift accepts changes that improve reproducibility, evaluation coverage,
documentation, or software quality without changing the meaning of existing
evidence.

## Development setup

```powershell
uv sync --locked --extra dev
uv run pytest -q
uv run ruff check src tests tools
uv run ruff format --check src tests tools
uv run mypy src/heartshift
uv run python tools/validate_repository.py
uv build
uv run python tools/validate_distribution.py
```

Changes that depend on the full prediction archive must also pass:

```powershell
git lfs pull
git lfs fsck
uv run heartshift-validate --repo-root .
```

## Research invariants

- The endpoint is historical angiographic disease status (`num > 0`), not
  prospective population risk.
- Hospital identity is required for splitting and auditing and is never a
  disease-prediction feature.
- Outer leave-one-hospital-out targets are locked. Preprocessing, selection,
  early stopping, calibration, threshold choice, and auditors use source data
  only.
- Zero-shot generalization, unlabelled-target adaptation, and labelled-target
  adaptation remain separate experiments and result tables.
- Raw archives, extracted sources, checksums, split manifests, configurations,
  sample-level predictions, failures, and prior runs are immutable evidence.
- Synthetic MNAR and concept-shift failures remain visible. No result supports
  unrestricted MNAR robustness.
- Every aggregate must be reproducible from predictions joined by `sample_id`.
- Existing outer outcomes cannot be reused for a newly confirmatory claim.

## Adding a method

1. Implement the estimator under `src/heartshift/models/` without introducing
   target-label access.
2. Add a versioned configuration under `configs/`.
3. Register provenance, dependency, and execution status in
   `configs/research/method_registry_v2.yaml`.
4. Add synthetic tests for missingness, mask identity, determinism, and leakage.
5. Use the existing prediction schema and reporting interfaces.
6. Record unavailable dependencies or incompatible protocols as explicit
   omissions rather than substituting a different method under the same name.

## Adding data

New datasets require an official source URL, version, citation, license,
retrieval date, file sizes, SHA-256 hashes, schema, endpoint definition,
exclusions, and redistribution status. Raw inputs are never overwritten.
Patient or subject identifiers must be partitioned before row-level sampling.

## Pull requests

Keep changes focused and include:

- the scientific or engineering reason for the change;
- tests covering new behavior;
- any affected evidence class or claim boundary;
- dependency and license changes;
- commands used for validation.

Do not force-push over published evidence, delete historical failures, or commit
credentials, local environments, generated caches, or protected data.
