# Release evidence gate

HeartShift uses a two-stage release boundary so a mutable working tree or a CI
job cannot attest to evidence it has not independently observed.

## Stage A: candidate attestation

For one full 40-character Git commit, CI records individually hashed evidence
for the candidate tree, contracts, repository checks, tests and coverage, lint,
formatting, strict typing, build, distribution boundary, clean-wheel smoke test,
dependency licences, and full-history secret scan. The assembler accepts only
records bound to that exact commit and verifies each evidence file again. The
release policy must itself be a regular file in that commit and match the
checked-out bytes exactly; a dirty or external policy cannot weaken the scan.

If every local gate passes, the report status is `pending_remote_ci`, not
`pass`. Missing status records are `not_run`; they are never inferred from an
earlier workflow or a developer workstation.

```powershell
uv run heartshift release scan `
  --candidate $commit `
  --policy configs/release/release_gate_policy_v1.json `
  --output .audit/release/$commit/candidate-tree.json

uv run heartshift release assemble `
  --candidate $commit --version 0.1.0 `
  --status-directory .audit/release/$commit `
  --output .audit/release/$commit/candidate-attestation.json
```

## Stage B: completed remote attestation

After the exact candidate's remote CI run succeeds, its externally retrieved
workflow evidence is hash-bound as the `remote_ci` gate. A completed report
requires an HTTPS workflow URL, every Stage A gate, the remote-CI gate, matching
candidate commits, valid self-hashes, and matching evidence hashes. Only this
mode can have status `pass`.

The remote evidence is the GitHub Actions workflow-run API response. The gate
checks its run ID, canonical workflow URL, repository, completed/successful
state, and `head_sha` against the candidate; a generic log or locally written
success flag is not accepted. A passing gate always requires a non-empty,
hash-bound evidence file. The declared release version is also read from the
candidate's own `pyproject.toml` rather than trusted from a command argument.

For a successful run, retrieve and bind the API response before assembling the
completed report (all output paths are create-only):

```powershell
$runId = "123456789"
$workflowUrl = "https://github.com/abdullahuseyinli-dot/heart-disease-risk-model-benchmark/actions/runs/$runId"
$remoteEvidence = ".audit/release/$commit/github-workflow-run-$runId.json"
gh api "repos/abdullahuseyinli-dot/heart-disease-risk-model-benchmark/actions/runs/$runId" |
  Set-Content -LiteralPath $remoteEvidence -Encoding utf8NoBOM

uv run heartshift release record-gate `
  --candidate $commit --gate remote_ci --status pass `
  --evidence $remoteEvidence `
  --output ".audit/release/$commit/remote_ci.status.json"

uv run heartshift release assemble `
  --candidate $commit --version 0.1.0 `
  --status-directory ".audit/release/$commit" `
  --output ".audit/release/$commit/completed-attestation.json" `
  --mode completed_remote_attestation --workflow-url $workflowUrl
```

The final release inventory then enumerates every Git tree entry, mode, Git
object ID, Git blob size, and any Git LFS SHA-256 object ID and logical size. It
also binds required documentation, licences, cards, dataset manifests, method
registry, paper files, and the completed gate report.

## Tag and archive rule

Create `v0.1.0` only after the completed remote attestation and inventory pass
for the same commit and the tag exactly matches the package version. Attach both
JSON records to the release. Do not mint or advertise a DOI until a real
immutable archive deposit exists and its checksum has been verified.

Tags, releases, failed gate records, and prior candidates are preserved. A
failed release attempt is not deleted or rewritten; a corrected candidate uses
a new commit and a new evidence directory.
