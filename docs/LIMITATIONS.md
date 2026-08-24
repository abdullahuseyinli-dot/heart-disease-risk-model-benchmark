# Limitations and prohibited claims

1. UCI Heart contains only four small, historical, referred cohorts. It cannot
   establish contemporary clinical utility, prospective risk, or generalization
   to a population of hospitals.
2. The binary endpoint collapses ordinal angiographic disease status and is not a
   longitudinal cardiovascular-event outcome.
3. The public source does not support a reliable patient-cluster audit across the
   four heart files. Duplicate-record analysis cannot exclude all identity overlap.
4. Site is entangled with country, era, case mix, acquisition, and missingness.
   Leave-one-site-out performance cannot identify which mechanism caused a change.
5. Synthetic deletion policies approximate plausible acquisition shifts; they do
   not reproduce every clinical workflow or missing-data mechanism.
6. No method can claim unrestricted robustness to MNAR missingness. The synthetic
   outcome-dependent regime is a falsification stress test, not a training promise.
7. Unlabelled prevalence adaptation relies on stable class-conditional evidence.
   Diagnostic acceptance is not proof of that assumption, and rejection disables
   automatic adaptation rather than repairing the shift.
8. The 130-US-hospitals experiment validates a method on readmission, not heart
   disease. Its admission-source domain is a context proxy because hospital IDs
   are unavailable. The TableShift-compatible endpoint is any readmission, not
   specifically 30-day readmission.
9. Foundation-model comparisons may be affected by unknown pretraining-data
   overlap. TabPFN version, checkpoint access, and results are therefore reported
   separately and cannot establish uncontaminated superiority.
10. Multiple ablations and metrics create multiplicity. The protocol designates a
    primary estimand; secondary findings and bootstrap probabilities are not
    treated as independent confirmatory tests.
11. Four observed hospitals do not support a random-effects confidence interval
    over future hospitals. Patient bootstrap intervals are conditional on these
    sites and keep site-specific estimates visible.
12. Excellent discrimination would still not establish calibration, net benefit,
    fairness, causal validity, robustness to changing care, or safety in use.
13. Protocol v2 was selected after the registered protocol-v1 joint-DRO gate
    failed. Although no outer labels were accessed, the v2 source comparison is
    selection evidence rather than independent confirmation. The failed gate,
    selection rule, candidate set, and pivot output must be reported together.
14. Source-OOF Platt scaling assumes that a monotone map learned on natural
    source-hospital folds transports to the target hospital and to imposed mask
    policies. That assumption can fail. Raw scores remain visible and calibration
    must not be described as target calibration.
15. The benchmark compares many methods. A strong outer rank after source-driven
    selection is hypothesis-generating unless it also survives the prespecified
    paired contrasts, mechanism tests, and independent task.
16. The final heart report has mixed protocol provenance: baseline predictions
    completed under the v3 lock; MIRRAMS uses no-refit v5 finalization of fixed v4
    shards after a post-endpoint aggregation failure; and PS-MaskDRO is run under
    the v5 exact-filename fix. The completed v3 results were known at v4 recovery,
    and the MIRRAMS endpoint was joined before v5 recovery. Both preserved failures
    and this sequence must accompany the report.

The appropriate fallback is a transparent benchmark or negative-results paper if
the proposed method fails its synthetic gates or does not improve robust outcomes
against equally tuned controls.
