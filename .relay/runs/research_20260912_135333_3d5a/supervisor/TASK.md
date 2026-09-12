# Research Task: Quantify verifier conservatism from global versus local Lipschitz bounds

## Research question
How much of the current reach-avoid certificate verifier's conservatism is caused by using a single global Lipschitz constant in the `tau_K + L_lip` margin, and can sound box-local Lipschitz bounds substantially improve certifiability without changing the certificate theory, policy, abstraction, or verified domain?

## Hypothesis
A global Lipschitz constant is dominated by the steepest part of the certificate network and therefore over-penalizes many boxes whose local certificate variation is much smaller. Replacing that global contribution with sound per-box or region-local upper bounds should increase verification margins and reduce counterexamples, especially away from high-gradient regions, while preserving soundness. If local bounds yield little improvement, then the dominant bottleneck is likely abstraction error, certificate shape, or the `tau_K` term rather than Lipschitz over-approximation.

## Experiment / implementation request
Run one controlled verifier study on a single difficult reach-avoid instance for which the current neural-policy -> boxwise-policy -> neural-certificate pipeline has a trained certificate that is either narrowly certifiable or fails due to small verifier margins. Prefer the hard double-integrator obstacle layout used in the current research trajectory; if that artifact is unavailable, use the strongest existing continuous-state/discrete-action reach-avoid instance with a nontrivial MLP certificate.

Do not retrain the policy, redesign the partition, alter certificate semantics, or introduce new theoretical certificate conditions. The independent variable is only how the sound Lipschitz contribution used by the existing verifier is bounded.

Evaluate the exact same fixed policy, boxwise abstraction, MLP certificate checkpoint(s), verifier equations, partition, and tolerances under the following conditions:

1. **Global-bound baseline.** Use the current project implementation of the global `L_lip` bound exactly as-is. Record how it is computed and its numerical value.
2. **Layerwise/global tightened control.** If the implementation currently uses a crude product-of-norms bound and there is an already-supported sound tightening that does not change theory (for example, an existing bound-propagation routine in the project), evaluate that as a control. Do not add a large new bounding framework solely for this condition.
3. **Box-local sound Lipschitz bound.** For every verifier box, compute a sound upper bound on the certificate's Lipschitz constant over that box or its exact verifier-relevant neighborhood. Use an implementation-compatible method such as interval/linear bound propagation, bounded Jacobian norms, or another already-available sound network bound. The method must return an upper bound, not an empirical gradient estimate.
4. **Diagnostic oracle only (not for certification).** Estimate the observed maximum gradient/Jacobian norm by dense/random sampling inside each box. This is only to measure bound looseness and must never be used to claim formal verification.

Use the local bound only in the same location where `L_lip` currently appears in the verifier expression. Do not alter `tau_K`, box transition construction, certificate thresholds, unsafe/goal definitions, or any other term.

If the verifier uses a neighborhood larger than the current state box because of transition uncertainty, compute the local bound over the full sound region required by the existing proof, not merely the nominal box. Document that region precisely.

Run the study on at least one fixed certificate checkpoint. If certificate optimization stochasticity matters and multiple successful/borderline checkpoints already exist, evaluate 3 matched checkpoints/seeds; do not spend the main budget retraining certificates just to manufacture a comparison.

For each box and each sound bounding method, preserve the Lipschitz upper bound, the resulting complete verifier margin, and pass/fail status. Localize exactly which boxes change status between global and local bounds.

As a follow-up diagnostic within the same task, decompose the verifier penalty into `tau_K` and the Lipschitz contribution for every violating or near-violating box. This should answer whether localizing `L_lip` is enough or whether `tau_K` remains the dominant term after tightening.

Do not combine this task with adaptive partition refinement, counterexample-guided certificate repair, policy retraining, or new layouts. Those would confound the source of improvement.

## Controls
- Same environment, initial/goal/unsafe sets, dynamics, horizon, and verified domain.
- Same neural policy checkpoint.
- Same boxwise policy and exact partition.
- Same MLP certificate checkpoint for all bound variants in a matched comparison.
- Same verifier equations, `tau_K`, tolerances, arithmetic precision, and transition over-approximation.
- Only the sound computation of `L_lip` may change.
- Any sampled/empirical Jacobian estimate is diagnostic only and must not be treated as a proof bound.
- If local bounds are cached or approximated for speed, verify that the cached value is still a sound upper bound for every covered box.

## Expected measurements
For every box and every bound variant, record:

- Global or local `L_lip` value used by the verifier.
- `tau_K` value and the separate numerical contribution of `tau_K` and `L_lip` to the verifier margin.
- Final verifier margin for each relevant certificate condition.
- Pass/fail status and violation type.
- Whether a box changes from fail to pass, pass to fail, or remains unchanged relative to the global baseline.
- Sound-bound / empirical-gradient ratio as a measure of bound looseness, clearly labeled diagnostic.
- Box location, obstacle distance, action region, and whether it lies near an action-switching or obstacle boundary when those labels are already available.

Aggregate and report:

- Number and fraction of boxes verified under each method.
- Total number of formal violations under each method.
- Minimum, median, and lower-tail verification margins.
- Distribution of global-to-local Lipschitz tightening factors.
- Fraction of baseline violations resolved purely by local bounds.
- Among remaining violations, fraction dominated numerically by `tau_K` versus the Lipschitz contribution.
- Verification runtime and bound-computation runtime.
- If the certificate becomes globally valid only under local bounds, explicitly identify the boxes responsible for the change and verify that every local bound used there is sound under the proof's required region.

Produce at least these figures/tables:

1. Spatial heatmap of per-box local `L_lip` and baseline violating boxes.
2. Spatial heatmap of verifier margin improvement from global to local bounds.
3. Scatter plot of bound-tightening factor versus verifier-margin improvement.
4. Table comparing global baseline, any existing tightened-global control, and local bounds: verified-box fraction, total violations, minimum margin, and runtime.
5. For remaining violations, a decomposition plot/table showing `tau_K` versus `L_lip` contribution.

## Completion criteria
The task is complete when the same fixed certificate has been evaluated under the global and sound local Lipschitz-bound variants and the experiment supports one of the following conclusions:

**Local-bound improvement:** sound local bounds materially reduce verifier conservatism, resolve a substantial fraction of violations or convert the certificate from failing to formally valid, and do so without changing any other component of the pipeline.

**Global bound is adequate:** local sound bounds are much tighter numerically but produce little change in verification outcome, showing that `L_lip` is not the dominant bottleneck.

**Different bottleneck identified:** after local tightening, the unresolved margin is dominated by `tau_K`, abstraction/transition uncertainty, or another fixed verifier term. Quantify this rather than simply reporting failure.

A result is not complete if it reports only aggregate pass/fail. It must provide per-box evidence that isolates how much margin is recovered by tighter Lipschitz bounding and what term limits the verifier afterward.

## Artifacts / results to preserve
- Exact environment, policy, boxwise abstraction, partition, and certificate checkpoint identifiers.
- Exact mathematical formulas used for the global and local Lipschitz bounds.
- Implementation/configuration of the sound local bound method and the verifier region over which each bound is valid.
- Per-box machine-readable data containing global/local bounds, empirical diagnostic gradient, `tau_K`, verifier margins, condition labels, and pass/fail status.
- Formal verification logs for every sound-bound variant.
- Runtime measurements for bound computation and verification.
- Spatial heatmaps and comparison plots described above.
- Any boxes whose status changes, together with enough data to independently inspect their local bounds and margins.
- A concise `RESULTS.md` stating whether global `L_lip` is a significant source of conservatism and what the next bottleneck is if local bounds are insufficient.
