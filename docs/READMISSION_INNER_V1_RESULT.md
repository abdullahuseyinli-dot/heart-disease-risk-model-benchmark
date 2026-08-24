# Independent readmission source-only selection result

Status: complete source-only inner evidence; independent tests remain locked

Run directory: `artifacts/runs/readmission-inner-v1`

## Evidence integrity

The registered run completed 42 fits across 9 experiment tracks and retained 42
prediction shards. The consolidated evidence contains 1,586,046 predictions for
3,433 validation encounters from 2,888 patients, 462 policy/replicate metric
rows, 137 neural history rows, and one deterministic selection for each track.

Every metric recomputed exactly from patient-level predictions. The selection
rule was independently reconstructed from fit summaries, all prediction scores
were finite and bounded, all configuration hashes matched, and no `id_test` or
`ood_test` row occurred in the evidence. Validation patients have zero overlap
with either test split. Exact hashes for 50 files, including every shard, are in
`evidence_audit.json`.

## Selected source-validation comparison

| Experiment | Macro policy balanced log loss | Worst policy balanced log loss | Natural balanced log loss | Natural AUROC |
|---|---:|---:|---:|---:|
| PS-MaskDRO-ANE | 0.6643 | 0.6757 | 0.6618 | 0.6408 |
| PS-MaskDRO | 0.6662 | 0.6784 | 0.6628 | 0.6403 |
| Prior separation | 0.6717 | 0.6884 | 0.6660 | 0.6309 |
| Pooled ERM | 0.6813 | 0.7083 | 0.6730 | 0.6882 |
| CatBoost, environment/class balanced | 0.6893 | 0.7787 | 0.6591 | 0.6584 |
| MIRRAMS Equation 9 | 0.6896 | 0.7106 | 0.6822 | 0.6855 |
| Pooled logistic | 0.7147 | 0.8007 | 0.6807 | 0.7035 |
| LightGBM, environment/class balanced | 0.7156 | 0.8386 | 0.6686 | 0.6570 |
| Logistic, environment/class balanced | 0.7491 | 0.8787 | 0.6965 | 0.6365 |

Relative to pooled ERM, PS-MaskDRO-ANE improves macro and worst-policy balanced
log loss by 0.0170 and 0.0327 respectively. ANE improves over base PS-MaskDRO by
0.00192 macro and 0.00275 worst-policy balanced log loss, while natural AUROC
changes by +0.00046.

The robustness gain does not dominate every metric. Relative to pooled ERM,
PS-MaskDRO-ANE loses 0.0475 natural AUROC. Pooled logistic has the highest
natural AUROC, and CatBoost has the lowest natural balanced log loss, among these
source-validation selections. These trade-offs must remain visible in the
independent outer report rather than collapsing the task to one favourable
score.

## Interpretation

This is encouraging cross-dataset source-selection evidence: the structured
missingness methods rank best on the declared policy-balanced proper-score
criterion, and ANE gives a small consistent improvement over base PS-MaskDRO.
It is not independent test performance because the same source validation split
selects configurations. The patient-disjoint `id_test` and distribution-shifted
`ood_test` labels remain unopened and may be evaluated only after the complete
candidate is frozen.
