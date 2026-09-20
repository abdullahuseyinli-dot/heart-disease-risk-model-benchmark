# Versioning and citation boundary

HeartShift uses semantic versions for the software and immutable identifiers for
research evidence. A software version does not reset an experiment's evidence
status or make a consumed evaluation confirmatory again.

## Current candidate

The package, `heartshift.__version__`, `CITATION.cff`, and `.zenodo.json` declare
`0.1.0`. Until the exact-candidate remote gate passes and an annotated `v0.1.0`
tag and release are created, this is release-candidate metadata rather than a
published release. No DOI is currently claimed.

The preservation tag `legacy-v1-development-consumed` points to
`551d4706b02538e9a096b4a4b5b544f0484091a5`, the exact commit documented by the
legacy validity audit. The September 2026 reconciliation found the tag absent
locally and on GitHub and restored it on 2026-09-20. Its annotation explicitly
records the restoration date; it does not claim to reproduce missing original
tag metadata. It is not software version 1.0.0 and must not be moved or deleted.
See the [reconciliation record](audit/2026-09-20/REPOSITORY_AUDIT.md).

## Version changes

- Patch: compatible software, documentation, validator, or packaging fixes that
  do not alter a frozen experiment or evidence value.
- Minor: new backward-compatible benchmark capabilities, contracts, methods, or
  datasets with explicit evidence status.
- Major: incompatible public Python/CLI contracts or benchmark-specification
  changes.

Every version change must align `pyproject.toml`, `src/heartshift/__init__.py`,
`CITATION.cff`, `.zenodo.json`, the changelog, release workflow, and tag. The
repository validator enforces the machine-readable metadata agreement.

## Evidence versions

Dataset, split, method, run, table, report, freeze, and release contracts carry
their own schema or protocol versions. Never replace an earlier contract in
place merely to align it with a software release. Create a new versioned record,
preserve the old record and its hashes, and explain supersession.

## Archival citation

After the release inventory passes, release assets should include the completed
remote attestation and inventory. If a later Zenodo deposit succeeds, verify the
deposit's files and checksums before adding its DOI to citation metadata. A
reserved, imagined, or failed deposit is not cited.
