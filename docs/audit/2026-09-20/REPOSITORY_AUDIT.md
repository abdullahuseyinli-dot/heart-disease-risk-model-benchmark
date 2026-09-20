# Local work and Git reconciliation

Audit date: 2026-09-20. Baseline candidate: `d67941271b80961b0a24d7e459a8918e3e2510e4`.
This is a preservation and representation audit, not a new model evaluation.

## Findings

| Finding at the start of this audit | Evidence | Resolution in this change |
| --- | --- | --- |
| Contract/release hardening was pushed but absent from `main` | `main` at `1b8def7`; open [PR #2](https://github.com/abdullahuseyinli-dot/heart-disease-risk-model-benchmark/pull/2) at `d679412`, two commits ahead | Continue the existing review branch and include this audit. Publication to a branch is distinct from integration into `main`. |
| No additional uncommitted research or ignored non-cache evidence in the three located checkouts | Clean initial worktrees; all tracked blobs from both older clones exist in the fetched remote history | Preserve those copies; do not duplicate obsolete Git histories. |
| The final research branch was already represented on `main` | `research/shiftguard-v1-development` and `main` had identical trees; PR #1 was merged | Do not describe the older research branches as missing experiments. |
| Coursework had substantial local-only detail | 152 retained files; 27 byte-identical counterparts in the candidate checkout | Recover 121 original output files and index the four source documents. |
| Notebook development was only partly visible through the exported scripts | 41 main, 1 Dask, and 26 edge code cells; execution counts not in cell order | Add reviewable source extracts and per-cell hashes; retain original notebooks locally. |
| The documented legacy tag was absent locally and remotely | Empty tag lists; validity audit identifies commit `551d4706b02538e9a096b4a4b5b544f0484091a5` | Restore an annotated reference to that exact commit, explicitly dated 2026-09-20; do not recreate or backdate missing original tag metadata. |
| Ideas and trials were scattered across directories | 56 run directories, 12 report directories, one separate failure package | Add an explicitly classified ledger, machine-readable inventory, and research atlas. |
| Ten-seed crosswalk pointed to the combined development report | The 0.600458 result is in the historical seed-extension stability report | Correct the claim-to-evidence link. |
| Archived `throughput_rps` was liable to overinterpretation | Original edge notebook computes `1000 / median(latency_ms)` | Explain that this is a latency-derived rate proxy; retain the raw field and value. |
| Release workflow had a malformed action pin | Checkout reference was 41 hex characters; official v6.0.2 resolves to `de0fac2e4500dabe0009e67214ff5f5447ce83dd` | Correct the pin and validate action-reference shape in the repository check. |
| Contributor instructions lagged the release branch | Missing reporting/neural extras and old method registry version | Align source/full-evidence commands and the v3 registry with the current CLI. |
| Exact-candidate scan failed on Windows Git newline conversion | Committed release policy used LF; the checkout had CRLF | Mark the policy byte-stable and materialize the existing Git blob. Policy values and the committed policy are unchanged; original checkout bytes and the failure record remain local. |

The [Git snapshot](git_snapshot.json) records exact remote OIDs, initial clone
states, historical blob comparisons, and scope limits. It is a dated observation,
not a live branch-status display. The preserved backup bundle advertised head
`37cb2ec`; it was identified without restoring its superseded history into the
active repository.

The local preservation tag was restored after the required data, contract,
test, lint, typing, structure, and evidence checks passed. The annotation states
that it is a new restoration record for historical development evidence. It is
not a new software release or a claim about original tag creation time.

## Coursework preservation

| Disposition | Files | Meaning |
| --- | ---: | --- |
| Already tracked, byte-identical | 27 | Linked to their existing repository paths; includes two pairs of duplicate image filenames. |
| Recovered original outputs | 121 | 64 PNGs, 28 CSVs, 28 text tables, and one JSON file; 4,365,570 bytes copied without changing the originals. |
| Source documents retained locally and indexed | 4 | Three notebooks and the 54-page group report. Notebook source cells are represented through explicitly transformed extracts. |

The 121 recovered files are artifacts, not 121 independent experiments. The
original Dask summary and edge samples/summary also have existing versions with
equivalent contents but different text bytes. Their original bytes are preserved
in the recovered archive rather than replacing the existing records.

[coursework_manifest.json](coursework_manifest.json) maps every source file to its
disposition, size, SHA-256, and repository destination. The original four source
documents remain in the local coursework directory; their hashes provide identity
but do not make their full contents downloadable from Git. The
[notebook source index](notebook_source_index.json) records cell types, execution
counts, output counts, source hashes, and path-redaction flags. Extracts remove
outputs and notebook metadata and label original cell order explicitly.

The [coursework provenance guide](../../legacy/COURSEWORK_PROVENANCE.md) maps each
trial to recovered tables, plots, and current interpretation. No existing raw
data, predictions, configurations, failures, or reports were overwritten.

## Representation and scientific scope

The [research atlas](../../RESEARCH_ATLAS.md) separates project-developed
hypotheses from attributed baselines. Its conclusion is that HeartShift has a
substantial research and engineering contribution, while a general architectural
novelty or clinical-validity claim remains unsupported. Negative results are
first-class entries: the joint-axis source gate, synthetic v2, real-data
adaptation abstention, ShiftGuard v7, and support-aware routing.

The two new figures read frozen CSV tables. The seed plot includes all ten
variants and both familywise intervals for the highlighted V4 comparison. The
ShiftGuard plot shows all five full revisions and states that the mechanism
banks differ. Neither graph changes an estimand, fits a model, or resets the
evidence class.

## Verification boundaries

The source suite at baseline passed 148 tests, with three full-evidence tests
excluded. After fetching the required LFS bindings and source audit shards, the
unified data/study/contract validator passed. The initial partial-LFS validation
correctly failed on the still-unmaterialized readmission prediction pointer;
that failure was resolved by retrieving the declared object, not by weakening
validation.

Current change verification is recorded in [VALIDATION.md](VALIDATION.md).
Remote CI is tied to the commit it actually tested. A historical successful run
does not attest a later commit, and a Git LFS pointer alone is not the underlying
prediction data. This audit does not issue a software release, DOI, or new
scientific acceptance decision.

The local search covered the accessible Desktop, Downloads, Documents/Codex,
OneDrive, Atlas staging directories, and a filename search of the attached D
drive. Installed environments and caches were excluded from evidence comparison.
Historical manifests refer to a different workstation, whose untracked files
cannot be audited from this machine. The result is therefore scoped to the
located local copies and the fetched GitHub history.
