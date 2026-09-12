# Research Task: Stress-test neural reach-avoid certification on a hard double-integrator layout

## Research question
Can the current end-to-end pipeline — neural policy -> boxwise policy abstraction -> neural reach-avoid certificate — still produce a valid certificate on a materially harder double-integrator obstacle geometry, and does the verifier term `tau_K + L_lip` materially affect certifiability, robustness, or sample/optimization efficiency?

## Hypothesis
The pipeline should remain viable on a hard but feasible layout if the policy first learns a robust collision-free route and the certificate learner uses the implementation optimizations already proven useful in the current codebase. We expect the `tau_K + L_lip` verifier to be at least as reliable as the version without that term, and potentially to eliminate false/fragile certificates near box boundaries or obstacle-adjacent regions. If both variants certify equally well, the result is still valuable because it suggests the extra Lipschitz margin may be unnecessary on this class of instances.

## Experiment / implementation request
Conduct one controlled experiment on a **single hard double-integrator reach-avoid layout** that is substantially more difficult than the simple layouts already solved. Use a diagonal start/goal arrangement and multiple rectangular obstacles that force a nontrivial multi-turn route; include at least one central blocking rectangle and additional obstacle(s) positioned so the policy cannot solve the task by a near-straight trajectory. Keep the layout feasible with visible clearance rather than creating a pathological narrow passage.

Start from the existing successful double-integrator pipeline and preserve its theory. You may reuse implementation-level ideas from LOGRASM or prior successful experiments (bound tightening, stable optimization, network sizing, batching, verifier engineering), but do not import LOGRASM theory or change the certificate semantics.

For the chosen layout:

1. Train a neural policy to high reach-avoid success.
2. Convert that policy to the same style of boxwise/discontinuous policy abstraction used in the current work. Start with the box resolution that worked previously; refine only if verification diagnostics show the abstraction is the bottleneck.
3. Train an MLP reach-avoid certificate using the current theoretically intended conditions.
4. Run a **matched ablation** with two verifier variants while holding everything else fixed as much as possible:
   - Variant A: verifier/constraints with `tau_K + L_lip`.
   - Variant B: the corresponding verifier without the `L_lip` term (or the exact current baseline interpretation of “without `tau_K + L_lip`” used in the project; record the precise formulas implemented).
5. Use identical environment geometry, policy, boxwise abstraction, certificate architecture, training budget, optimizer settings, seeds, and verification tolerances across the two variants unless a variant becomes numerically impossible. Any deviation must be recorded.
6. If certificate learning initially fails, debug scientifically: identify whether the failure comes from policy quality, abstraction error, certificate optimization, or the verifier margin. Make only targeted changes and preserve each failed configuration/results rather than overwriting them.

Do not move on to MiniGrid P0-P4 in this run. The purpose of this iteration is to resolve whether the harder continuous-state/discrete-action double-integrator case works and whether the `tau_K + L_lip` term is doing meaningful work.

## Controls
- Same hard layout for both verifier variants.
- Same trained neural policy and same boxwise policy abstraction for both variants.
- Same certificate architecture and initialization scheme.
- Same optimizer, learning-rate schedule, training-step budget, verifier tolerances, and box partition unless a documented diagnostic forces a change.
- Use multiple random seeds for certificate learning (prefer 5 if affordable; minimum 3) so a single lucky initialization is not treated as evidence.
- Preserve the already-solved simple double-integrator layout as a regression control: rerun the final verifier implementation on it if any verifier code is changed, to ensure the new experiment did not break the known success case.

## Expected measurements
Report, for each verifier variant and each seed:

- Neural-policy reach-avoid success over at least 1000 rollouts, including goal-hit, obstacle-hit, boundary/unsafe-hit, and timeout rates.
- Boxwise-policy reach-avoid success over at least 1000 rollouts and the drop relative to the neural policy.
- Number of boxes / partition resolution and any refinement performed.
- Certificate training success/failure, training steps, wall-clock time if readily available, and final loss decomposition by certificate condition.
- Formal verifier outcome and, if failed, the number/type/location of violating boxes or counterexamples.
- Minimum verification margins for the key certificate conditions, especially the decrease condition and obstacle/unsafe constraints.
- For the `tau_K + L_lip` ablation, record `tau_K`, `L_lip`, their contribution to the bound/margin, and whether the final certification decision differs between variants.
- Empirical trajectories overlaid on the obstacle layout for both neural and boxwise policies.
- If certification succeeds, evaluate at least 1000 fresh rollouts from the full allowed initial set as an empirical sanity check; do not present this as a replacement for formal verification.

## Completion criteria
This task is complete when one of the following scientifically useful outcomes is reached and fully documented:

**Success outcome:** the hard layout is solved by the neural and boxwise policies and at least one verifier variant produces a valid neural certificate, with a clean matched comparison between the two verifier variants.

**Informative failure outcome:** after targeted debugging, certification still fails, but the failure is localized with evidence to one of policy quality, box abstraction, certificate optimization, or verifier conservatism; the best achieved margins/counterexamples and all configurations needed to reproduce the failure are preserved. Do not hide a negative result by repeatedly changing the environment until something certifies.

In either case, the experiment must answer the main comparison question: whether adding `tau_K + L_lip` changes certificate validity, margin, robustness, or optimization behavior on the hard layout.

## Artifacts / results to preserve
Preserve enough information for exact reproduction and later paper use:

- Exact environment/layout definition and a rendered figure.
- Neural-policy checkpoint and training config.
- Boxwise-policy representation, partition definition, and any refinement log.
- Certificate checkpoints for every reported seed/variant.
- Exact verifier formulas/configs for both ablation variants.
- Per-seed metrics in machine-readable CSV/JSON.
- Verification logs, violating boxes/counterexamples, and minimum-margin summaries.
- Trajectory plots and certificate visualizations/slices that make failures or margins interpretable.
- A concise `RESULTS.md` stating what was attempted, what changed during debugging, the final comparison table, and the scientific conclusion.
