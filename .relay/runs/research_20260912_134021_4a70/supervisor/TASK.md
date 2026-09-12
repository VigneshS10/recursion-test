# Research Task: Stress-test D20 under obstacle-misaligned random box partitions

## Research question
Does the current D20 boxwise abstraction remain reliable when partition boxes are not obstacle-aware and can straddle both free space and obstacle regions, or is its current success dependent on obstacle-aligned partitioning?

## Hypothesis
D20 will degrade materially when boxes mix obstacle and free-space states because a single abstract box must summarize qualitatively different reach-avoid behavior. We expect either (i) policy-abstraction fidelity to drop, (ii) verification to become substantially more conservative, or (iii) mixed boxes to dominate the remaining counterexamples. If D20 remains robust under randomized misaligned partitions, that is strong evidence that its utility is not an artifact of hand-aligned geometry.

## Experiment / implementation request
Conduct one controlled stress test of the existing D20 boxwise-policy abstraction on a reach-avoid environment/layout that has already been solved successfully with the current obstacle-aware D20 setup. Do not redesign the certificate theory or train a new environment merely to make the experiment easier.

The independent variable is **partition placement**, not policy quality or certificate architecture.

1. Select one existing solved layout with nontrivial rectangular obstacle geometry and a stored/high-quality neural policy.
2. Reproduce the current obstacle-aware D20 abstraction and its downstream reach-avoid evaluation as the reference condition.
3. Construct randomized D20 partitions with the **same nominal box size / resolution and approximately the same box count**, but randomize grid origin/offset or box placement so that some boxes deliberately intersect both obstacle and free space. Do not clip or realign boxes to obstacle boundaries for the randomized condition.
4. Generate at least 20 randomized partition realizations if computationally practical; minimum 10. Use fixed recorded seeds.
5. For every partition, explicitly classify boxes as free-only, obstacle-only, or mixed free/obstacle. Record the fraction and spatial distribution of mixed boxes.
6. Build the boxwise policy using the existing D20 rule exactly as currently defined. Do not silently introduce an obstacle-special-case to rescue mixed boxes. If the current implementation is undefined on mixed boxes, first document that failure precisely; then, only as a diagnostic secondary analysis, test the smallest principled interpretation already compatible with the method and label it clearly as such.
7. Evaluate the resulting boxwise policy empirically and with the same downstream certificate/verifier pipeline used in the successful reference experiment. Keep the neural policy, environment, verifier definition, certificate architecture, optimization budget, and tolerances fixed.
8. If verification fails, localize violations relative to mixed boxes: measure whether violating boxes are themselves mixed or lie one transition away from a mixed box. Preserve counterexamples instead of changing the random partition until one passes.
9. As a diagnostic control, take the worst randomized partition and perform **targeted obstacle-boundary refinement only** while leaving the rest of the partition unchanged. Test whether this restores abstraction fidelity and/or certifiability. This refinement is a diagnostic of the cause, not a replacement for the randomized-condition result.

The scientific objective is to determine whether obstacle alignment is an implicit requirement of D20 and, if so, quantify how strongly mixed boxes predict failure.

## Controls
- Same environment/layout and initial/goal/unsafe definitions for all conditions.
- Same trained neural policy for all partitions.
- Same nominal D20 box dimensions/resolution and comparable total box count.
- Same D20 action-assignment/abstraction rule.
- Same rollout initial-state distribution and evaluation budget.
- Same certificate architecture, optimizer, training budget, verifier equations, and tolerances wherever downstream certification is attempted.
- Reference control: the existing obstacle-aware/aligned D20 partition.
- Randomized partitions differ only by placement/alignment, with seeds recorded before evaluation.
- Diagnostic refinement is applied only after the randomized result is recorded and only to the worst partition.

## Expected measurements
For the reference partition and every randomized partition, record:

- Total number of boxes and box dimensions.
- Number/fraction of free-only, obstacle-only, and mixed free/obstacle boxes.
- Neural-policy reach-avoid success over at least 1000 rollouts as a fixed baseline.
- Boxwise-policy reach-avoid success over at least 1000 rollouts, including goal, obstacle/unsafe, and timeout rates.
- Drop in success from neural policy to D20 boxwise policy.
- Action disagreement between neural and boxwise policies on a common state sample, reported globally and separately for states in/near mixed boxes.
- If available, a box-level abstraction error/confidence statistic under the existing D20 implementation.
- Formal certificate/verifier pass or fail.
- Number and type of verifier violations / counterexamples and their minimum margins.
- Fraction of verifier violations located in mixed boxes, adjacent to mixed boxes, and elsewhere.
- Correlation or rank association between mixed-box fraction and key outcomes (boxwise success drop, violating-box count, minimum verifier margin).
- For the worst randomized partition, measurements before and after targeted obstacle-boundary refinement, including how many boxes were added/changed.

Produce scatter/summary plots relating mixed-box fraction to abstraction fidelity and verification outcomes. Spatially plot at least the reference partition, a representative randomized partition, and the worst randomized partition with mixed boxes and verifier violations overlaid.

## Completion criteria
This task is complete when the obstacle-aware reference and the full randomized-partition set have been evaluated, and the evidence supports one of these conclusions:

**Robustness result:** D20 retains comparable boxwise-policy fidelity and certifiability despite substantial obstacle/free mixing, with no strong relationship between mixed boxes and failures.

**Limitation result:** randomized misalignment produces a reproducible degradation, and mixed boxes predict policy-abstraction errors, verifier violations, or failed certification. The targeted-refinement diagnostic should then establish whether obstacle-boundary refinement substantially repairs the worst case.

**Informative implementation limitation:** D20 is mathematically or operationally undefined for mixed boxes under the current rule. In that case, document the exact assumption being violated and demonstrate how frequently random partitions violate it; do not hide the issue by automatically realigning boxes.

A single favorable random partition is not sufficient. Completion requires a distribution over randomized partitions and a quantitative comparison to the obstacle-aware reference.

## Artifacts / results to preserve
- Exact environment/layout and stored neural-policy checkpoint/config used.
- Reference obstacle-aware D20 partition.
- Every randomized partition definition and RNG seed.
- Per-box labels identifying free-only, obstacle-only, and mixed boxes.
- Boxwise policy representations for all evaluated partitions.
- Machine-readable per-partition metrics (CSV/JSON).
- Certificate/verifier logs, minimum margins, and counterexamples for each partition.
- Spatial figures showing partitions, obstacles, mixed boxes, trajectories, and verifier violations.
- Scatter/summary plots linking mixed-box prevalence to abstraction and verification metrics.
- Worst-case partition before and after targeted obstacle-boundary refinement.
- A concise `RESULTS.md` stating whether obstacle alignment is an implicit D20 requirement, the magnitude of any degradation, and the implications for future abstraction/refinement design.
