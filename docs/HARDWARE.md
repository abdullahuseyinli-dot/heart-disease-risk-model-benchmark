# Hardware and execution profiles

Hardware is part of reproducibility metadata, not evidence that one method is
more accurate. Runtime comparisons are valid only within a shared environment,
workload, precision, warm-up, and timing protocol.

## Recorded research workstation

The privacy-safe machine record is
`manifests/environment/confirmatory_machine_20260824.json`. Its values were
observed on 2026-08-24 and are hash-bound to
`artifacts/runs/classical-outer-v3/run_manifest.json`.

| Component | Recorded value |
| --- | --- |
| CPU | Intel Core Ultra 9 285H; 16 physical cores and 16 logical processors reported by Windows |
| RAM | 68,137,205,760 physical bytes (63.46 GiB) |
| GPU | NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU |
| GPU memory | 12,227 MiB reported by `nvidia-smi` |
| Compute capability | 12.0 |
| Driver | 596.72 |
| Driver-reported CUDA compatibility | 13.2 |
| Executed PyTorch runtime | PyTorch 2.13.0+cu130 with CUDA runtime 13.0 |
| Python | 3.11.9 |
| Recorded platform | Windows-10-10.0.26200-SP0 |

The driver compatibility version and the CUDA runtime compiled into PyTorch are
different quantities; they are intentionally reported separately. Hostname,
username, hardware serial numbers, and credentials are excluded.

## Dependency profiles

Use one neural profile at a time:

```powershell
# Portable CI and CPU neural tests
uv sync --locked --extra dev --extra reporting --extra neural-cpu

# Local CUDA research runs
uv sync --locked --extra dev --extra reporting --extra neural-cuda

# Optional comparator families
uv sync --locked --extra classical --extra foundation-models --extra neural-cuda
```

`neural-cpu` and `neural-cuda` are resolver conflicts by design. This prevents
an environment from silently selecting a different PyTorch build. The legacy
`models` and `modern` extras remain aliases for earlier installation guidance;
new work should use the explicit profiles.

## Resource-aware execution

The 12 GiB GPU can run the small-table neural methods used here, but method
budgets remain configuration-controlled. Increasing batch size, seeds, or
ensembles because more memory is available would create a different experiment
and requires a new source-only protocol. GPU availability does not permit a
locked target to be reopened.

For comparable timing studies, record at minimum: commit, lockfile hash, Python,
package versions, device and driver, precision, thread settings, batch size,
warm-up, repetitions, input shape, and whether model/checkpoint loading is timed.
