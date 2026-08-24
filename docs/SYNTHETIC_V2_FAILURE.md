# Protocol-v2 synthetic gate failure

Status: immutable failed source/synthetic evidence; outer evaluation closed

## Result

`synthetic-mechanism-v2` completed all 63 registered
experiment-by-scenario-by-seed cells and wrote 31,500 sample-level target
predictions. Four of six checks passed. The gate failed because:

- label-plus-MAR diagnostic acceptance was 0.0, below the required 0.5;
- conditional-shift diagnostic acceptance was 0.6666666666666666, above the
  allowed 0.5.

Prevalence estimation and adapted log loss passed their declared label/MAR
checks, and MNAR acceptance stayed below its threshold. Those successes do not
override the failed gate. Exact files and hashes are in
`artifacts/runs/synthetic-mechanism-v2/evidence_audit.json`.

## Mechanism diagnosis

The v2 diagnostic compares the target scalar evidence distribution with mixtures
of source class-conditional scalar evidence. It attempted to align acquisition by
applying masks sampled from the unlabelled target to already-incomplete source
records. This operation can delete a source value but cannot reveal a value that
was naturally missing in the source. If source and target natural observed masks
are (M_s) and (M_t), the transported source mask is their intersection,
(M_s \land M_t), not (M_t). Consequently, a change in MAR acquisition policy
changes the observed-data class-conditionals even when the latent complete-data
class-conditionals are stable. The v2 `label_mar` scenario therefore does not
satisfy ordinary observed-data label shift after feasible mask transport.

The test also has two statistical weaknesses exposed by the run:

1. three mask replicates of the same source patients are treated as independent;
2. a one-dimensional score can lack power when conditional shift remains close
   to a different mixture of source score distributions.

Changing only the significance level or acceptance threshold would conceal these
problems and is prohibited.

## Protocol-v3 route registered after this failure

The next mechanism study separates four questions:

1. `label_only`: stable acquisition and class-conditionals, changed prevalence;
   adaptation should be accepted, estimate prevalence, and improve log loss;
2. `mar_policy_shift`: changed ignorable acquisition; zero-shot mask robustness
   is evaluated, while unsupported prevalence adaptation may abstain;
3. `conditional_only`: stable acquisition with changed class-conditionals;
   automatic label-shift adaptation should be rejected;
4. `mnar_outcome_only`: target-only outcome-dependent missingness;
   automatic adaptation should be rejected.

The diagnostic will use unique-patient source predictions and a composite
multiview kernel mean discrepancy over calibrated evidence, always-observed core
features, and observed-mask indicators. Prevalence is re-estimated inside the
bootstrap. A mask-support audit remains a separate hard guard. Raw equal-prior,
always-adapted research, and automatically deployed probabilities remain separate
columns.

This design is informed by the failed v2 experiment and is not represented as an
independent preregistration. It follows the distribution-matching view of label
shift, the need for calibrated scores, multivariate shift testing, and explicit
missingness-shift treatment described in:

- Garg et al., [A Unified View of Label Shift Estimation](https://proceedings.neurips.cc/paper/2020/hash/219e052492f4008818b8adb6366c7ed6-Abstract.html), NeurIPS 2020;
- Gerber et al., [Kernel-Based Tests for Likelihood-Free Hypothesis Testing](https://proceedings.neurips.cc/paper_files/paper/2023/hash/32c6d65ec2591dfcfb3f0e345a51f585-Abstract-Conference.html), NeurIPS 2023;
- Rabanser et al., [Failing Loudly](https://proceedings.neurips.cc/paper/2019/hash/846c260d715e5b854ffad5f70a516c88-Abstract.html), NeurIPS 2019;
- Zhou et al., [Domain Adaptation under Missingness Shift](https://proceedings.mlr.press/v206/zhou23b.html), AISTATS 2023;
- Miller and Futoma, [Label Shift Estimators for Non-Ignorable Missing Data](https://arxiv.org/abs/2310.18261), 2023.

No heart outer label was opened during this diagnosis or the exploratory
acquisition-conditioned MMD prototype.
