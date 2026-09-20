# Data acquisition and reconstruction runbook

Dataset manifests under `manifests/datasets/` are the release-facing source of
truth. They bind official releases, licences, claim scope, expected byte counts,
SHA-256 digests, acquisition modes, and repository-relative destinations.

## Verify preserved sources

Verification never uses the network:

```powershell
uv run heartshift data verify --manifest manifests/datasets/uci_heart_v1.json
uv run heartshift data verify --manifest manifests/datasets/uci_diabetes_readmission_v1.json
uv run heartshift data verify --manifest manifests/datasets/eicu_crd_demo_v2.0.1.json
```

Any missing file or byte/hash mismatch is a failed gate. Do not repair a
mismatch by overwriting it; preserve the mismatch and investigate provenance.

## Acquire an absent public source

Network access requires an explicit flag. A manifest fixes the expected content
before the request begins:

```powershell
uv run heartshift data acquire `
  --manifest manifests/datasets/uci_heart_v1.json `
  --allow-network `
  --receipt .audit/acquisition/uci-heart-receipt.json
```

All destinations are checked before the first request. Downloads use a unique
same-directory partial file, are checked for exact bytes and SHA-256, and are
published with create-only semantics. A failed partial remains available for
forensic review. Existing valid files are retained; existing invalid files
cause failure.

An offline environment (`HEARTSHIFT_OFFLINE=1`) blocks acquisition even if the
flag is present. Terms-gated artifacts cannot be downloaded by this command.
Credentials belong in provider-managed user storage and never in this
repository, a manifest, command history, or receipt.

## Canonical preparation

Preparation commands now preflight every declared output and refuse to replace
canonical tables, profiles, raw manifests, or split manifests:

```powershell
uv run heartshift-prepare --repo-root .
uv run heartshift-prepare-readmission --repo-root .
uv run heartshift-prepare-eicu-demo --repo-root .
```

The tracked outputs already exist, so running these commands in the release
checkout is expected to stop with a create-only error. Reconstruction should use
a fresh destination or a fresh clone with no derived outputs; it must never
delete the preserved release evidence to make a command succeed.

## Licence boundaries

UCI Heart and UCI Diabetes Readmission are CC BY 4.0 and require attribution.
The eICU demo and its adapted database retain ODbL 1.0 terms. The demo is a
pipeline smoke test only. Credentialed eICU/GOSSIS data are not included and
would require their own terms-gated manifest and an untouched external protocol.
