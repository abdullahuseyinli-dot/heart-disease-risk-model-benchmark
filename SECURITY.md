# Security policy

## Supported code

HeartShift does not currently claim a stable software release. Security fixes
target the current development branch. Historical tags, immutable research
evidence, raw data, and consumed-outcome artifacts remain preserved for audit
and should not be interpreted as supported software versions.

| Code line | Security support |
| --- | --- |
| Current development branch | Yes |
| Historical and legacy tags | No |

## Reporting a vulnerability

Do not disclose a suspected vulnerability, credential, secret, exploit, or
sensitive record in a public issue, discussion, pull request, log, or artifact.

Use the repository's private **Report a vulnerability** function on GitHub when
it is available. If private reporting is not available, open a public issue that
only requests a private communication channel; include no vulnerability details
or sensitive material in that issue.

A useful private report includes:

- the affected commit and component;
- the operating system and dependency versions;
- minimal reproduction steps or a proof of concept without real patient data;
- the expected and observed behavior;
- the potential confidentiality, integrity, or availability impact; and
- any known mitigation.

Never attach credentials, private keys, protected health information, or raw
clinical rows. Use synthetic fixtures and redact local paths, tokens, and account
identifiers. The maintainer will coordinate disclosure after the issue and an
appropriate remediation have been assessed; no response-time or bounty promise
is made by this policy.

## Scope

Software vulnerabilities, dependency compromise, secret exposure, unsafe file
handling, and evidence-integrity bypasses are in scope. Scientific disagreements,
benchmark corrections, and documentation errors should use the normal issue
tracker unless reporting them would expose sensitive information.

HeartShift is research software, not a medical device. Reports about clinical
deployment, diagnostic use, or patient-specific decisions cannot make such use
safe; those uses are outside the project's intended scope.
