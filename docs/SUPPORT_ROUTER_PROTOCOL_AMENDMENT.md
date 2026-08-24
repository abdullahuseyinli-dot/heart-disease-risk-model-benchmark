# Support-router protocol amendment

## Decision

The full `support_router_v2` source meta-evaluation was cancelled before it ran.
Its premise that a rotating third source hospital was untouched was too strong.
The prediction for a row in that third hospital excluded the hospital, but OOF
predictions used to train the router could come from base fits that included it.
That is valid cross-fitted stacking data, not a fully nested evaluation of the
whole two-level learner.

The failed v2 smoke and its configuration remain preserved. They may test code
paths, but they cannot support an out-of-hospital performance claim.

## Corrected v3 route

For each historical UCI outer hospital, v3 performs the following operations:

1. choose attention versus equal-budget DeepSets separately for the
   prior-separated and joint-policy-DRO expert families using source inner scores;
2. choose the classical anchor and its hyperparameter using source inner scores;
3. align ten-seed expert ensembles under identical mask codes and all thirteen
   feature-level mask bits;
4. compare three prespecified router inputs: support only, support plus all expert
   logits, and support plus anchor-relative expert logits;
5. rotate the source early-stopping hospital, train ten seeds per rotation, and
   select the router input mode using source validation balanced log loss;
6. persist predictions from every router checkpoint and every fixed comparator
   before reading the historical target endpoint;
7. report patient-level predictions, cell metrics, source selections, routing
   weights, fitted-model variation, conditional patient bootstrap intervals, and
   an exploratory hospital/patient bootstrap.

Fixed comparisons include each expert, equal logit averaging, the source-selected
best expert, and a source-selected convex logit blend. The router must improve
macro site-worst policy balanced log loss over the strongest of them, retain
natural AUROC within 0.01 of that same comparator, and avoid collapse to a mean
maximum expert weight of 0.95 or higher.

## Interpretation boundary

The current run is outcome-isolated in execution, but the UCI outcomes were
already consumed by earlier releases. Passing v3 is therefore a development
result, not confirmation; failing it stops the router. Four hospitals are also
too few for stable future-hospital inference. The only confirmation route is a
frozen, authorized external multi-hospital dataset with hospital-level splitting.

Configuration: `configs/research/support_router_outer_v3.yaml`.

## Completed result

The full v3 run is complete. The router's macro hospital worst-policy balanced
log loss is 0.619081 versus 0.615985 for the strongest fixed comparator, equal
logit averaging. The robust-improvement gate therefore failed. Natural AUROC
noninferiority and the no-collapse check passed. The conditional percentile
interval for equal blend minus router is [-0.006138, 0.000090], and the
exploratory four-hospital hierarchical interval is [-0.008206, 0.001316].

The result does not advance the router. Full evidence and the subsequent exact
ten-seed stability analysis are documented in
`docs/HEART_RESEARCH_DEVELOPMENT_RESULT.md`.
