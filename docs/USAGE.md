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
uv sync --locked --extra dev
```

### POSIX shell

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone \
  https://github.com/abdullahuseyinli-dot/heart-disease-risk-model-benchmark.git
cd heart-disease-risk-model-benchmark
uv sync --locked --extra dev
```

Run the portable quality gate:

```powershell
uv run pytest -q --cov=heartshift --cov-report=term-missing --cov-fail-under=45
uv run ruff check src tests tools
uv run ruff format --check src tests tools
uv run mypy src/heartshift
uv run heartshift-validate --help
uv run python tools/validate_repository.py
uv build
uv run python tools/validate_distribution.py
uv run python tools/smoke_install_distribution.py
```

These commands exercise package behavior, static checks, public documentation,
licensing markers, the archived compact benchmark record, and the wheel/sdist
boundary. They do not assert that large prediction objects are present.

## Evidence checkout

Fetch only report-level objects for manuscript-table inspection:

```powershell
git lfs pull --include="artifacts/reports/**"
```

Fetch and verify the complete preserved evidence store:

```powershell
git lfs pull
git lfs fsck
uv run heartshift-validate --repo-root .
```

The complete checkout is several gigabytes. Missing LFS objects are a failed
evidence audit, not permission to replace a pointer or regenerate an outcome.

## Optional model implementations

The default development environment is sufficient for the quality gate. Install
the optional benchmark implementations only when reproducing a method that
requires them:

```powershell
uv sync --locked --extra dev --extra models --extra modern
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
