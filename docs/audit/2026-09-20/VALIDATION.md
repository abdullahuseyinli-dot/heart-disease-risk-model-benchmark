# Verification record

This change preserves and documents existing evidence. It does not refit a model
or rerun a consumed outer evaluation. Commands below were run in the locked
Python 3.11.9 development environment on Windows.

| Check | Result |
| --- | --- |
| Baseline source suite | 148 passed, 3 full-evidence tests excluded; 54.14% source-profile coverage. |
| Full local suite after adding archive checks | 156 passed, including all three evidence-marked tests; 55.36% full-profile coverage. |
| Focused audit checks after adding the workflow-pin guard | All 6 passed; includes failure visibility, unclassified trials, CRLF/LF portability, corruption, complete figure outputs, paths with spaces, and malformed pins. |
| Ruff lint and format | Passed across source, tests, and tools; 128 Python files formatted. |
| Mypy | Passed for 122 source/test files. |
| Unified data, study, and contract validation | Passed, with declared LFS bindings materialized. |
| Distribution | Wheel and source archive built; content validation and clean-wheel installation smoke test passed. |
| Recovery inventory | 152 source entries reconciled; recovered output and notebook-extract hashes verified. |
| Figure provenance | Input-table hashes, complete plotted-data exports, and PNG/SVG output hashes verified. |
| Visual inspection | Both new figures inspected as PNGs; label/legend spacing corrected and final images rechecked. |

The source tree now contains 157 tests. The full local run preceded the last
workflow-pin test; that test and the five other audit tests then passed together.
Remote workflows rerun the applicable complete suite on the pushed candidate.
The [PR checks](https://github.com/abdullahuseyinli-dot/heart-disease-risk-model-benchmark/pull/2/checks)
and [workflow runs](https://github.com/abdullahuseyinli-dot/heart-disease-risk-model-benchmark/actions)
are the live commit-specific records; this document does not claim a remote
result before the corresponding run completes.

## Reproduce

```powershell
uv sync --locked --extra dev --extra reporting --extra neural-cpu
uv run pytest -q -m "not full_evidence" --cov=heartshift --cov-config=configs/coverage/source-only.coveragerc
uv run ruff check src tests tools
uv run ruff format --check src tests tools
uv run mypy
uv run python tools/validate_repository.py
uv run python tools/build_research_atlas.py --check
uv run python tools/plot_research_overview.py --check
uv build
uv run python tools/validate_distribution.py
uv run python tools/smoke_install_distribution.py
```

The full-evidence route is documented in [USAGE.md](../../USAGE.md). Local
validation fetched the declared report bindings, the complete observed-backbone
source audit, and the PS-MaskDRO prediction table used by the publication-figure
test. It did not download every historical LFS shard or claim a full local
`git lfs fsck`; the repository-wide LFS workflow performs that separate check.

## Disclosed verification failures and fixes

An initial partial-LFS check failed because the readmission prediction path was
still a 134-byte pointer rather than its 210,394,026-byte object. Fetching the
declared LFS object resolved the failure. No evidence binding was relaxed.

Development checks also caught long/mixed-format lines, a dynamic-module typing
annotation, a figure legend overlapping its caption, and the inherited 41-character
checkout pin. These were corrected before publication. The new recovery check
explicitly handles Git's declared CRLF-to-LF conversion for old tracked text,
while new recovered files retain exact original bytes. Numeric corruption still
fails the check.

Validation log files remain under the ignored local `.audit/verification/`
directory. The current [audit](REPOSITORY_AUDIT.md), manifests, and figure
sidecars are the compact public evidence for this preservation work.
