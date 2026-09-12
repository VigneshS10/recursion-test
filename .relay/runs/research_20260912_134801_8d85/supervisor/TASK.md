# Research Task: Compare fixed box resolutions against verifier-guided adaptive refinement

## Research question
Can verifier-guided adaptive refinement of the boxwise policy abstraction achieve the same or better reach-avoid certifiability than a fixed D20 partition while using substantially fewer boxes and less verification effort, especially around obstacle boundaries and policy-discontinuity regions?

## Hypothesis
A uniform D20 partition spends resolution in easy regions while remaining too coarse exactly where certification is difficult: near obstacle boundaries, action-switching surfaces, and verifier counterexamples. Refining only boxes implicated by abstraction disagreement or formal verifier violations should improve certified margins and reduce failure rates more efficiently than globally shrinking the box size. If adaptive refinement does not outperform matched uniform refinements, then fixed-resolution D20 is simpler and scientifically preferable.

## Experiment / implementation request
Conduct one controlled experiment on a single nontrivial reach-avoid layout for which the neural-policy -> boxwise-policy -> neural-certificate pipeline is already known to work or nearly work. Prefer the hard double-integrator layout from the current research trajectory if it is available; otherwise use the strongest existing solved continuous-state/discrete-action layout with rectangular obstacles.

The goal is not to invent new certificate theory. Keep the current policy, certificate semantics, verifier equations, and `tau_K + L_lip` treatment fixed. Vary only the abstraction/refinement strategy.

Evaluate these conditions from the same trained neural policy:

1. **Uniform coarse control:** a deliberately coarse partition, approximately D40 if that convention means larger boxes/coarser resolution in the current implementation. If the D-number convention differs, choose a clearly coarser uniform resolution than D20 and record the exact box dimensions.
2. **Uniform D20 reference:** the current standard D20 abstraction.
3. **Uniform fine control:** a clearly finer uniform partition, approximately D10 if that convention means smaller boxes/finer resolution. Again, use exact box dimensions rather than relying on the label.
4. **Adaptive refinement:** start from the same coarse partition as condition 1 and iteratively split only boxes selected by verifier/abstraction diagnostics until certification succeeds or a predefined resource budget is exhausted.

For adaptive refinement, use a simple, theory-preserving selection rule rather than adding a new learned component. At each round:

- Construct the boxwise policy with the current abstraction rule.
- Train or reuse the same certificate procedure and run the exact formal verifier.
- Mark candidate boxes for refinement if they satisfy at least one of the following, in priority order:
  1. They contain a formal verifier violation/counterexample.
  2. They are immediately predecessor/neighbor boxes whose transition image reaches a violating box.
  3. They show high neural-policy/boxwise-policy action disagreement on a fixed diagnostic state sample.
  4. They straddle an obstacle boundary or known action-switching boundary and are implicated in a small verification margin.
- Split selected boxes geometrically using the simplest existing compatible subdivision (for example, bisect each state dimension or use the project's existing refinement operation). Do not change obstacle geometry or clip boxes in a way that changes the verified domain.
- Rebuild only what must change, rerun verification, and preserve the full refinement history.
- Stop when a valid certificate is obtained or when the adaptive method reaches the same total box count or verification-cost budget as the uniform fine control.

Do not tune the adaptive rule separately after seeing the test outcome. Choose thresholds/priorities once from existing diagnostics, record them, and use them for all refinement rounds.

If the certificate optimization itself is stochastic, use at least 3 matched seeds per condition. Reuse identical initializations where meaningful. If certificate training failure obscures the abstraction comparison, first separate verifier failure caused by a genuinely invalid abstraction from optimizer failure by checking the best achieved margins/counterexamples and rerunning the same abstraction with matched seeds.

## Controls
- Same environment, initial set, goal set, unsafe set, dynamics, and horizon across all conditions.
- Same trained neural policy checkpoint.
- Same boxwise action-assignment rule; only partition resolution/refinement differs.
- Same certificate architecture, verifier equations, `tau_K + L_lip` formulation, tolerances, optimizer, and nominal training budget.
- Same diagnostic state sample for action-disagreement measurements.
- Same rollout initial-state distribution and rollout count.
- Adaptive refinement begins from exactly the same coarse partition as the uniform coarse control.
- Adaptive refinement must stop no later than the resource ceiling defined by the uniform fine control (box count and, where measurable, verification effort), so efficiency claims are meaningful.
- Record exact box dimensions/counts; do not rely only on D10/D20/D40 names because their convention may differ across code paths.

## Expected measurements
For every condition and seed, record:

- Exact partition geometry, total box count, and distribution of box sizes.
- Neural-policy reach-avoid success over at least 1000 rollouts as the fixed baseline.
- Boxwise-policy reach-avoid success over at least 1000 rollouts, including goal, unsafe/collision, and timeout rates.
- Global neural-vs-boxwise action disagreement and disagreement restricted to obstacle-adjacent / refined regions.
- Formal verifier pass/fail and minimum margin for every certificate condition.
- Number, type, and spatial location of violating boxes/counterexamples.
- Certificate training steps and success/failure per seed.
- Verification runtime and training runtime when readily available.
- Peak memory if already easy to measure; do not add substantial engineering just for profiling.
- For the adaptive condition at each refinement round: number of boxes selected, reason each was selected, boxes before/after splitting, total boxes, verifier violation count, minimum margin, boxwise success, and cumulative verification/training effort.

Produce efficiency curves rather than only final points:

- verifier minimum margin vs total box count,
- violating-box count vs total box count,
- boxwise-policy success vs total box count,
- verification runtime vs total box count,
- for adaptive refinement, cumulative refined boxes spatially overlaid on the environment.

The key comparison is the **Pareto tradeoff between certifiability and abstraction cost**. Report whether adaptive refinement reaches certification with fewer boxes and/or lower total verification effort than uniform refinement.

## Completion criteria
The task is complete when all four abstraction conditions have been evaluated under matched settings and one of the following conclusions is supported:

**Adaptive-refinement win:** starting from the coarse partition, targeted refinement reaches a valid certificate with materially fewer boxes or lower total verification cost than the uniform fine partition while maintaining comparable empirical policy fidelity.

**Uniform-resolution win:** adaptive refinement provides no meaningful efficiency/certification benefit over D20 or the uniform fine partition, establishing that the simpler fixed-resolution abstraction is preferable for this setting.

**Structural limitation:** neither adaptive nor uniform refinement certifies within the matched resource budget, but the experiment localizes the failure to policy abstraction, certificate expressivity/optimization, or verifier conservatism and preserves the evidence needed to decide the next research step.

A result is not complete if only the final pass/fail status is reported. The study must reveal how certification margin and violations evolve as resolution is spent and whether targeted allocation of boxes is more efficient than uniform allocation.

## Artifacts / results to preserve
- Exact environment/layout definition and neural-policy checkpoint/config.
- Initial coarse, D20, fine, and every adaptive partition definition.
- Boxwise-policy representations for each condition/refinement round.
- Exact adaptive selection rule, thresholds, split rule, and resource budget.
- Per-round list of selected boxes with selection reasons.
- Certificate checkpoints for every reported seed/condition.
- Machine-readable per-condition and per-round metrics (CSV/JSON).
- Formal verification logs, counterexamples, violating boxes, and minimum-margin summaries.
- Plots of certification/violation metrics versus box count and runtime.
- Spatial plots showing where adaptive refinement concentrates boxes relative to obstacles, trajectories, action-disagreement regions, and verifier violations.
- A concise `RESULTS.md` with the final Pareto comparison and a scientific conclusion on whether fixed D20 should remain the default or be replaced/supplemented by adaptive refinement.
