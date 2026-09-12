# Research Task: Test MLP certificate repair across MiniGrid P0–P4 with and without `tau_K + L_lip`

## Research question
Does the current continuous–discrete MiniGrid reach-avoid verifier remain effective when the certificate is an MLP that is iteratively repaired from verifier counterexamples, and does the `tau_K + L_lip` term materially improve final certifiability or robustness across the existing P0–P4 layouts?

## Hypothesis
The prior MiniGrid P0–P4 success should extend to an MLP certificate if repair is driven by genuine verifier violations rather than only sampled training loss. We expect counterexample-guided repair to close localized violations that remain after ordinary certificate training. The `tau_K + L_lip` variant should be more conservative but more reliable near partition boundaries and discontinuities in the boxwise policy; the no-`L_lip` variant may train more easily but should show smaller certified margins or additional verifier failures on at least some layouts.

## Experiment / implementation request
Run one controlled research study on the existing continuous–discrete MiniGrid P0–P4 benchmark suite using the verifier definition that previously succeeded on these layouts. Do not redesign the environment or certificate semantics. The goal is to test whether an MLP certificate plus verifier-guided repair is sufficient and whether `tau_K + L_lip` is scientifically necessary.

For each of P0, P1, P2, P3, and P4:

1. Reuse the existing successful policy/environment pipeline and the same boxwise/discontinuous policy abstraction already used for these layouts. Do not retrain or alter the policy unless regression checks show the stored policy no longer solves the task.
2. Train an MLP reach-avoid certificate using the project’s current theoretically intended conditions and verifier.
3. Verify the certificate globally over the intended state/box domain.
4. If verification fails, perform **counterexample-guided repair**: collect the verifier’s violating boxes/states, add targeted training constraints/samples derived from those violations, continue certificate optimization, and re-run the exact same verifier. Iterate until certification succeeds or a clearly documented stopping criterion is met.
5. Run a matched two-variant comparison:
   - Variant A: verifier/constraints using `tau_K + L_lip`.
   - Variant B: the corresponding version without `L_lip` (use the project’s exact baseline interpretation and record the implemented formulas).
6. Keep the policy, box partition, certificate architecture, optimizer, repair rule, training budget per repair round, verifier tolerances, and random seeds matched across the two variants as closely as possible.
7. Use the same certificate architecture across P0–P4 initially. Only increase capacity for a layout if diagnostics show clear representational underfitting; if changed, preserve and report the failed fixed-architecture result first.
8. Do not introduce new theoretical conditions or production-style engineering. Reuse implementation optimizations from the existing codebase or LOGRASM only when they improve numerical stability, bounds, batching, or runtime without changing the theory.

The key scientific object is the repair trajectory: whether formal violations shrink and disappear under targeted repair, and whether the two verifier variants differ in repair difficulty, certified margin, or final outcome.

## Controls
- Use the existing P0–P4 layouts exactly as defined in the project.
- Same policy and same boxwise abstraction for both verifier variants on a given layout.
- Same MLP architecture and initialization scheme for matched runs.
- Same optimizer, learning rate, per-round training budget, repair sampling rule, and maximum number of repair rounds.
- Same verifier tolerances and partition unless a documented verifier soundness requirement forces a change.
- Minimum 3 random seeds per layout/variant; prefer 5 if computationally practical.
- Before interpreting any new failure, run the existing previously successful MiniGrid verifier/certificate setup on at least P0 and P4 as regression controls to confirm that the environment and verifier implementation have not been broken.
- Preserve initial, pre-repair certificate results separately from repaired results; do not overwrite them.

## Expected measurements
For every layout, verifier variant, seed, and repair round, record:

- Neural/boxwise policy reach-avoid success over at least 1000 rollouts, including goal, unsafe/collision, and timeout rates.
- Certificate training loss decomposed by theoretical condition.
- Formal verifier pass/fail status.
- Number of violating boxes/counterexamples by certificate condition.
- Minimum certified margin for each key condition, especially decrease/progress and unsafe/barrier conditions.
- `tau_K`, `L_lip`, and the combined contribution to the bound for Variant A; record the exact corresponding quantity for Variant B.
- Number of repair rounds to certification, examples added per round, and total optimization steps.
- Wall-clock training and verification time if readily available.
- Whether the location/type of verifier violations migrates, shrinks, persists, or reappears across repair rounds.
- Final certificate success rate across seeds for each P0–P4 layout and each verifier variant.

Produce a summary table with rows P0–P4 and columns for each variant: seed-level certification rate, median repair rounds, initial violating-box count, final violating-box count, minimum final margin, and total verification/training effort.

## Completion criteria
The task is complete when all five layouts have been attempted under both verifier variants and the repair process is documented sufficiently to support one of the following conclusions:

**Positive result:** MLP certificate repair yields formally verified certificates on most or all P0–P4 layouts, with reproducible seed-level evidence and a clear comparison of `tau_K + L_lip` versus the no-`L_lip` baseline.

**Informative negative result:** one or more layouts remain uncertified after the predefined repair budget, but the remaining violations are localized and diagnosed as representational limitation, verifier conservatism, policy/abstraction issue, or optimization failure. Preserve the best margins and counterexamples rather than changing the layouts until they pass.

The experiment must specifically answer two questions: (1) does verifier-guided repair materially improve certification over the initial MLP certificate, and (2) does `tau_K + L_lip` change certification rate, required repair effort, or certified margins across P0–P4?

## Artifacts / results to preserve
- Exact P0–P4 environment/config definitions used.
- Policy and boxwise-policy checkpoints/representations used for each layout.
- Initial and final MLP certificate checkpoints for every seed and verifier variant.
- Exact certificate architecture, optimizer config, verifier formulas, tolerances, and partition settings.
- Counterexample/violating-box sets for every repair round.
- Machine-readable per-round metrics in CSV or JSON.
- Verification logs and minimum-margin summaries.
- Plots showing violation count and minimum margin versus repair round for each layout/variant.
- Representative certificate visualizations and spatial plots of violating boxes before and after repair.
- A concise `RESULTS.md` containing the P0–P4 comparison table, repair curves, failures/debugging changes, and the scientific conclusion about MLP repair and `tau_K + L_lip`.
