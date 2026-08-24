# Manuscript evidence index

The repository keeps manuscript-facing navigation separate from executable
research code. Machine-readable results remain under `artifacts/reports/`; this
directory does not duplicate raw data or prediction arrays.

The current evidence supports a benchmark and mechanism-safety paper centered on:

1. hospital and measurement-policy evaluation with source-only selection;
2. disagreement between robust proper scores and natural-policy discrimination;
3. adaptation abstention that prevents negative transfer;
4. cross-task heterogeneity between the heart and readmission evaluations; and
5. the negative routing result and post-outcome seed-stability analysis.

Primary sources include current reports and frozen study records:

- `docs/RESEARCH_PROTOCOL.md`
- `docs/METHOD_SPECIFICATION.md`
- `docs/HEART_OUTER_V5_RESULT.md`
- `docs/READMISSION_OUTER_V3_RESULT.md`
- `docs/HEART_RESEARCH_DEVELOPMENT_RESULT.md`
- `docs/LIMITATIONS.md`
- `docs/LITERATURE_MATRIX.md`
- `docs/REPORTING_CHECKLIST.md`

Headline tables and figures must be regenerated from their bound report
directories. Narrative text cannot replace a machine-readable comparison or
change an artifact's evidence status.

The protocol, literature matrix, and limitations files preserve the wording
bound to the executed studies. Current claim scope is summarized in
`docs/PROJECT_STATUS.md` and `docs/PUBLICATION_ROUTE.md`.
