# Installation and verification

HeartShift separates lightweight software checks from audits that require the
complete Git LFS evidence store. Both paths use the locked `uv` environment and
run without downloading models or research data.

## Source-only checkout

Skip automatic LFS downloads when cloning for code review or development.

### PowerShell

```powershell
$env:GIT_LFS_SKIP_SMUDGE = "1"
git clone https://github.com/abdullahuseyinli-dot/heart-disease-risk-model-benchmark.git
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
Set-Location heart-disease-risk-model-benchmark
uv sync --locked --extra dev --extra reporting --extra neural-cpu
```

### POSIX shell

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone \
  https://github.com/abdullahuseyinli-dot/heart-disease-risk-model-benchmark.git
cd heart-disease-risk-model-benchmark
uv sync --locked --extra dev --extra reporting --extra neural-cpu
```

Run the portable quality gate:

```powershell
uv run pytest -q -m "not full_evidence" --cov=heartshift `
  --cov-config=configs/coverage/source-only.coveragerc --cov-report=term-missing
uv run ruff check src tests tools
uv run ruff format --check src tests tools
uv run mypy
uv run heartshift --version
uv run heartshift contracts validate --repo-root .
uv run python tools/validate_repository.py
uv build
uv run python tools/validate_distribution.py
uv run python tools/smoke_install_distribution.py
```

These commands exercise package behavior, static checks, public documentation,
licensing markers, the archived compact benchmark record, and the wheel/sdist
boundary. Contract validation in this profile checks schemas and self-hashes;
it does not assert that large bound prediction objects are materialized. The
source-only profile excludes only the full-LFS validator and publication-figure
module and enforces a 54% floor; the complete-evidence profile measures those
modules and retains the 55% floor.

## Evidence checkout

Fetch only report-level objects for manuscript-table inspection:

```powershell
git lfs pull --include="artifacts/reports/**"
```

Fetch and verify the complete preserved evidence store:

```powershell
git lfs pull
git lfs fsck
uv run heartshift validate --repo-root .
uv run heartshift contracts validate --repo-root . --verify-bindings
uv run pytest -q --cov=heartshift --cov-report=term-missing
```

The complete checkout is several gigabytes. Missing LFS objects are a failed
evidence audit, not permission to replace a pointer or regenerate an outcome.

## Optional model implementations

The default development environment is sufficient for the quality gate. Install
the optional benchmark implementations only when reproducing a method that
requires them:

```powershell
uv sync --locked --extra dev --extra reporting --extra classical `
  --extra foundation-models --extra neural-cuda
```

Some upstream models have additional terms or authenticated model stores. Keep
credentials in the provider's user-level credential store, never in the
repository, configuration files, command history, or run artifacts.
Benchmark execution does not automatically retrieve TabICL checkpoints.

## Historical reconstruction

Exact historical commands and recovery records are retained in the frozen
[reproducibility record](REPRODUCIBILITY.md). Their paths and pre-execution
status language reflect the environment in which the evidence was produced.
Current result interpretation is defined by the [project status](PROJECT_STATUS.md),
[benchmark card](BENCHMARK_CARD.md), and report-specific evidence status.

The [acquisition runbook](DATA_ACQUISITION_RUNBOOK.md) documents offline
verification and explicit create-only downloads. The [hardware guide](HARDWARE.md)
separates CPU and CUDA dependency profiles.
