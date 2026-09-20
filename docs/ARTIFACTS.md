# Artifact storage and verification

HeartShift separates source code from large research evidence so routine
development does not require downloading every prediction shard.

## Storage tiers

| Tier | Contents | Delivery |
| --- | --- | --- |
| Source | Package code, tests, configurations, documentation, compact tables, and manifests | Git checkout with LFS smudging disabled |
| Core evidence | Canonical data, split manifests, report tables, locks, audits, and headline prediction sets | Git plus Git LFS |
| Full evidence | Individual-seed predictions, development shards, failures, and large reconstruction inputs | Versioned release archives with a SHA-256 inventory |

The tracked tree currently contains several gigabytes of Git LFS objects. GitHub
source archives may contain LFS pointers unless repository archive settings
include the LFS objects. A release archive and its checksum inventory are the
preferred citation boundary for full evidence.

## Source-only verification

Clone without automatically downloading the multi-gigabyte LFS store:

```powershell
$env:GIT_LFS_SKIP_SMUDGE = "1"
git clone https://github.com/abdullahuseyinli-dot/heart-disease-risk-model-benchmark.git
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
Set-Location heart-disease-risk-model-benchmark
```

```powershell
uv sync --locked --extra dev --extra reporting --extra neural-cpu
uv run pytest -q --cov=heartshift --cov-report=term-missing
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

This gate checks package behavior, repository links, compact legacy evidence,
licensing markers, and public-path policy.

## Full evidence verification

For report-level evidence only, use a targeted pull:

```powershell
git lfs pull --include="artifacts/reports/**"
```

For the complete evidence audit:

```powershell
git lfs install
git lfs pull
git lfs fsck
uv run heartshift validate --repo-root .
```

The full validator checks canonical data, split isolation, study contracts, and
the hashes required by the current reports. Report-specific audits reconstruct
aggregates directly from sample-level predictions.

## Immutability

Raw archives, extracted source files, manifests, locks, sample-level
predictions, failed runs, and prior dry runs are evidence rather than cache.
They are never rewritten for presentation. Historical run manifests may contain
workstation paths captured at execution time; those strings are provenance, not
portable inputs. Current split and command-provenance generators write
repository-relative paths and redact external workstation roots.

Release bundles must record:

- exact Git commit and version;
- every archive filename, byte size, and SHA-256;
- dataset licenses and citations;
- evidence class and claim scope;
- required versus optional artifacts;
- validation command and outcome.

No DOI or release checksum should be added until the corresponding immutable
deposit exists.

The exact-candidate procedure is specified in the
[release evidence gate](RELEASE_EVIDENCE_GATE.md). A candidate attestation stays
`pending_remote_ci`; only a separately bound successful remote run permits a
completed report and full-tree inventory.
