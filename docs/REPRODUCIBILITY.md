# Reproducibility runbook

Run from the repository root with Python 3.11. Commands below use the Windows
environment created for this study; replace `.venv\Scripts\python.exe` with the
equivalent interpreter on another platform.

## Environment and gates

```powershell
uv sync --frozen --extra models --extra modern --extra dev
.venv\Scripts\python.exe -m ruff check src tests tools
.venv\Scripts\python.exe -m mypy src/heartshift
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m heartshift.cli.validate --repo-root .
.venv\Scripts\python.exe tools\validate_repository.py
```

TabPFN v3 additionally requires the owner to accept the upstream licence and place
their credential in TabPFN's user cache. Never put it in the repository or command
history.

## Source-only development sequence

The `*-v1` sweeps are developmental. Fixed-candidate confirmation uses three
seeds and may start only after its referenced developmental selection exists.

```powershell
.venv\Scripts\python.exe -m heartshift.cli.psmask --repo-root . --config configs/benchmark/psmask_inner_v1.yaml --phase inner --run-name psmask-inner-v1
.venv\Scripts\python.exe -m heartshift.cli.psmask --repo-root . --config configs/benchmark/psmask_factorial_inner_v1.yaml --phase inner --run-name psmask-factorial-inner-v1
.venv\Scripts\python.exe -m heartshift.cli.psmask --repo-root . --config configs/benchmark/mirrams_inner_v1.yaml --phase inner --run-name mirrams-inner-v1
.venv\Scripts\python.exe -m heartshift.cli.benchmark --repo-root . --config configs/benchmark/classical_inner_confirm_v1.yaml --phase inner --run-name classical-inner-confirm-v1
.venv\Scripts\python.exe -m heartshift.cli.benchmark --repo-root . --config configs/benchmark/modern_inner_confirm_v1.yaml --phase inner --run-name modern-inner-confirm-v1
.venv\Scripts\python.exe -m heartshift.cli.benchmark --repo-root . --config configs/benchmark/modern_2026_inner_v1.yaml --phase inner --run-name modern-2026-inner-v1
.venv\Scripts\python.exe -m heartshift.cli.benchmark --repo-root . --config configs/benchmark/tabpfn_v3_inner_confirm_v1.yaml --phase inner --run-name tabpfn-v3-inner-confirm-v1
.venv\Scripts\python.exe -m heartshift.cli.psmask --repo-root . --config configs/benchmark/psmask_inner_confirm_v1.yaml --phase inner --run-name psmask-inner-confirm-v1
.venv\Scripts\python.exe -m heartshift.cli.psmask --repo-root . --config configs/benchmark/mirrams_inner_confirm_v1.yaml --phase inner --run-name mirrams-inner-confirm-v1
.venv\Scripts\python.exe -m heartshift.cli.synthetic --repo-root . --config configs/synthetic/mechanism_v2.yaml --run-name synthetic-mechanism-v2
.venv\Scripts\python.exe -m heartshift.cli.synthetic --repo-root . --config configs/synthetic/mechanism_v3.yaml --run-name synthetic-mechanism-v3
.venv\Scripts\python.exe -m heartshift.cli.readmission --repo-root . --config configs/independent/readmission_inner_v1.yaml --phase inner --run-name readmission-inner-v1
```

Failures and partial prediction shards remain in their run directories. Do not
overwrite or delete them. A failed synthetic gate closes all outer evaluations.
Protocol v2 failed and remains closed. Protocol v3 may authorize a newly versioned
freeze only if every gate in its machine-readable acceptance record passes; the
historical v2 freeze and outer commands below are not authorization to bypass it.

The protocol-v1 PS-MaskDRO gate failed and remains immutable at
`artifacts/runs/psmask-inner-confirm-v1/acceptance_gate_v1.json`. Protocol v2 does
not reinterpret that gate as passed. Its source-only, pre-outer deterministic
pivot is retained at
`artifacts/runs/psmask-inner-confirm-v1/pivot_selection_v2.json`. The freeze
command recomputes both records exactly and refuses to proceed if either changed.
The protocol-v3 freeze also reconstructs the synthetic gate from the round-trip
CSV, verifies its nested evidence hashes, and includes the independently audited
readmission source-only run.

## Freeze

Commit the complete method, configuration, tests, and source evidence so the
worktree is clean, then run:

```powershell
.venv\Scripts\python.exe -m heartshift.cli.freeze --repo-root . --config configs/release/freeze_v3.yaml --output artifacts/locks/heartshift_candidate_v3.json
```

The lock hashes the freeze, outer, synthetic, independent, and reporting
configurations; the Python method tree; `pyproject.toml`; `uv.lock`; and every
source prediction, selection, metric, fit-summary, manifest, resolved-config,
evidence-audit, and independent-validation artifact that exists. It separately
hashes the failed-v1, pivot-v2, and synthetic-v3 gate records. Outer CLIs
independently verify these hashes, nested audit hashes, and the Git commit. They
also enforce one frozen run name, so the same locked test cannot silently be rerun
under another directory.

## One-time locked evaluations

Run each command once, in this order, only after the freeze succeeds:

```powershell
.venv\Scripts\python.exe -m heartshift.cli.benchmark --repo-root . --config configs/benchmark/classical_outer_v3.yaml --phase outer --inner-run-dir artifacts/runs/classical-inner-confirm-v1 --confirmation RUN_LOCKED_OUTER_ONCE
.venv\Scripts\python.exe -m heartshift.cli.benchmark --repo-root . --config configs/benchmark/modern_outer_v3.yaml --phase outer --inner-run-dir artifacts/runs/modern-inner-confirm-v1 --confirmation RUN_LOCKED_OUTER_ONCE
.venv\Scripts\python.exe -m heartshift.cli.benchmark --repo-root . --config configs/benchmark/modern_2026_outer_v3.yaml --phase outer --inner-run-dir artifacts/runs/modern-2026-inner-v1 --confirmation RUN_LOCKED_OUTER_ONCE
.venv\Scripts\python.exe -m heartshift.cli.benchmark --repo-root . --config configs/benchmark/tabpfn_v3_outer_v3.yaml --phase outer --inner-run-dir artifacts/runs/tabpfn-v3-inner-confirm-v1 --confirmation RUN_LOCKED_OUTER_ONCE
.venv\Scripts\python.exe -m heartshift.cli.psmask --repo-root . --config configs/benchmark/mirrams_outer_v3.yaml --phase outer --confirmation RUN_LOCKED_OUTER_ONCE
.venv\Scripts\python.exe -m heartshift.cli.psmask --repo-root . --config configs/benchmark/psmask_outer_v3.yaml --phase outer --confirmation RUN_LOCKED_OUTER_ONCE
.venv\Scripts\python.exe -m heartshift.cli.readmission --repo-root . --config configs/independent/readmission_outer_v3.yaml --phase outer --confirmation RUN_LOCKED_READMISSION_TEST_ONCE
```

Each evaluator fixes every probability and adaptation decision before loading the
corresponding test endpoint.

## Prediction-derived reports

```powershell
.venv\Scripts\python.exe -m heartshift.cli.report_heart --repo-root . --config configs/reporting/heart_outer_v3.yaml --output-dir artifacts/reports/heart-outer-v3
.venv\Scripts\python.exe -m heartshift.cli.report_readmission --repo-root . --config configs/reporting/readmission_outer_v3.yaml --output-dir artifacts/reports/readmission-outer-v3
```

Reported aggregates are regenerated from sample-level prediction Parquets. Heart
intervals are paired patient bootstraps within the four observed sites; readmission
intervals cluster all encounters by patient. Neither supports random-effects
inference over future hospitals.

Classical and modern heart runs emit both raw and `source_oof_platt` methods. The
calibrator CSV is fixed from hospital-held-out source predictions before outer
labels are loaded. These are sensitivity results, not target recalibration.

## Protocol-v4 neural-only mechanical recovery

The v3 MIRRAMS command failed in its first non-adaptable shard before the heart
endpoint-loading block. Preserve `artifacts/runs/mirrams-outer-v3` and do not run
either neural evaluator under a v3 name. The completed v3 baseline/readmission
runs and failed MIRRAMS run are bound into the recovery lock. See
`docs/OUTER_V3_FAILURE_AND_V4_RECOVERY.md` for the scope and disclosure.

First reconstruct the completed evidence:

```powershell
.venv\Scripts\python.exe -m heartshift.cli.audit_outer --repo-root . --run-dir artifacts/runs/classical-outer-v3 --kind heart-classical --workers 8
.venv\Scripts\python.exe -m heartshift.cli.audit_outer --repo-root . --run-dir artifacts/runs/modern-v2-outer-v3 --kind heart-classical --workers 8
.venv\Scripts\python.exe -m heartshift.cli.audit_outer --repo-root . --run-dir artifacts/runs/modern-2026-outer-v3 --kind heart-classical --workers 8
.venv\Scripts\python.exe -m heartshift.cli.audit_outer --repo-root . --run-dir artifacts/runs/tabpfn-v3-outer-v3 --kind heart-classical --workers 8
.venv\Scripts\python.exe -m heartshift.cli.audit_outer --repo-root . --run-dir artifacts/runs/readmission-outer-v3 --kind readmission --workers 8
```

After committing the mechanical fix, recovery configurations, audits, and all
preserved v3 evidence with a clean worktree, generate the recovery lock:

```powershell
.venv\Scripts\python.exe -m heartshift.cli.freeze --repo-root . --config configs/release/freeze_v4_neural_recovery.yaml --output artifacts/locks/heartshift_candidate_v4_neural_recovery.json
```

Run each unopened neural target once, without code, configuration, documentation,
or commit changes between the freeze and both commands:

```powershell
.venv\Scripts\python.exe -m heartshift.cli.psmask --repo-root . --config configs/benchmark/mirrams_outer_v4.yaml --phase outer --confirmation RUN_LOCKED_OUTER_ONCE
.venv\Scripts\python.exe -m heartshift.cli.psmask --repo-root . --config configs/benchmark/psmask_outer_v4.yaml --phase outer --confirmation RUN_LOCKED_OUTER_ONCE
```

Then audit the neural runs and build the mixed-provenance report from immutable v3
baseline plus v4 neural predictions:

```powershell
.venv\Scripts\python.exe -m heartshift.cli.audit_outer --repo-root . --run-dir artifacts/runs/mirrams-outer-v4 --kind heart-neural
.venv\Scripts\python.exe -m heartshift.cli.audit_outer --repo-root . --run-dir artifacts/runs/psmask-outer-v4 --kind heart-neural
.venv\Scripts\python.exe -m heartshift.cli.report_heart --repo-root . --config configs/reporting/heart_outer_v4.yaml --output-dir artifacts/reports/heart-outer-v4
```

The completed v3 outcomes were known before the recovery lock, so this must be
reported as a mechanical recovery of two still-unopened neural evaluations, not
as a fresh joint preregistration of the already completed comparisons.

## Protocol-v5 exact-aggregation recovery

MIRRAMS v4 failed after its endpoint join because a broad shard glob also selected
the endpoint-free files. Preserve that run; do not retrain or overwrite it. Commit
the exact-filename fix, no-refit finalizer, v4 failure evidence, and v5 configs,
then freeze:

```powershell
.venv\Scripts\python.exe -m heartshift.cli.freeze --repo-root . --config configs/release/freeze_v5_aggregation_recovery.yaml --output artifacts/locks/heartshift_candidate_v5_aggregation_recovery.json
```

Finalize the already-fixed MIRRAMS shards without fitting a model, then run the
still-unopened PS-MaskDRO target once:

```powershell
.venv\Scripts\python.exe -m heartshift.cli.finalize_neural_outer --repo-root . --config configs/benchmark/mirrams_outer_v5_finalize.yaml --confirmation FINALIZE_LOCKED_SHARDS_ONCE
.venv\Scripts\python.exe -m heartshift.cli.psmask --repo-root . --config configs/benchmark/psmask_outer_v5.yaml --phase outer --confirmation RUN_LOCKED_OUTER_ONCE
```

Audit and report only after both commands finish:

```powershell
.venv\Scripts\python.exe -m heartshift.cli.audit_outer --repo-root . --run-dir artifacts/runs/mirrams-outer-v5-finalized --kind heart-neural
.venv\Scripts\python.exe -m heartshift.cli.audit_outer --repo-root . --run-dir artifacts/runs/psmask-outer-v5 --kind heart-neural
.venv\Scripts\python.exe -m heartshift.cli.report_heart --repo-root . --config configs/reporting/heart_outer_v5.yaml --output-dir artifacts/reports/heart-outer-v5
.venv\Scripts\python.exe -m heartshift.cli.audit_heart_report --repo-root . --config configs/reporting/heart_outer_v5.yaml --report-dir artifacts/reports/heart-outer-v5
.venv\Scripts\python.exe -m heartshift.cli.plot_publication --repo-root . --config configs/reporting/publication_figures_v2.yaml --output-dir artifacts/figures/heartshift-v5-r2
```
