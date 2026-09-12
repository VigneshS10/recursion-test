# Research Task: ReLU vs tanh reliability on XOR under a small training budget

## Research question
For a tiny MLP trained on the four-point XOR dataset under a deliberately limited optimization budget, does `tanh` reach a perfect XOR solution more reliably across random seeds than ReLU?

## Hypothesis
With the same architecture and optimizer budget, `tanh` will achieve 100% XOR accuracy on a larger fraction of random seeds than ReLU. This is a deliberately small experiment whose main purpose is to exercise the full research relay mechanics while still producing a real, falsifiable result.

## Experiment / implementation request
Implement a minimal, self-contained experiment in Python using PyTorch if available.

Dataset:
- Inputs: `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)`.
- Labels: `0, 1, 1, 0`.
- Train on all four points; there is no train/test split because the research question is optimization reliability, not generalization.

Model:
- MLP: `2 -> 4 -> 1`.
- Compare exactly two hidden activations: ReLU and `tanh`.
- Use `BCEWithLogitsLoss`; do not apply sigmoid before the training loss.
- Keep all architecture details identical except activation.

Training:
- Use Adam with learning rate `0.01`.
- Train for exactly 200 optimization steps per seed.
- Use full-batch training on all four XOR points.
- Run 20 seeds, seeds `0` through `19`.
- For each seed, reset the RNG to that seed immediately before constructing each activation's model so the comparison is paired as closely as practical.
- Do not tune learning rate, width, optimizer, initialization, or step budget separately by activation after seeing results.

For every activation and seed, record:
- final BCE loss,
- final sigmoid probabilities and binary predictions on all four points,
- final classification accuracy using threshold `0.5`,
- whether the run achieved exactly 100% accuracy by step 200,
- first optimization step at which 100% accuracy was reached, if ever.

Define step counting explicitly: evaluate the untrained model as step 0 if desired for diagnostics, but `first_success_step` must refer to the first completed optimizer update after which accuracy is 100%; therefore valid reported success steps are 1 through 200.

Preserve loss at every step or at least every 10 steps. If success is transient, do not replace final accuracy with best-seen accuracy; report both the first-success step and the final outcome faithfully.

## Controls
- Same XOR dataset and target encoding.
- Same architecture width, output layer, initialization procedure, and parameter shapes.
- Same optimizer, learning rate, number of updates, and full-batch update rule.
- Same 20 seeds, paired across activations.
- No activation-specific tuning.
- CPU execution is sufficient; do not add unnecessary engineering.

## Expected measurements
Produce a compact summary comparing ReLU and tanh:
- number and fraction of seeds reaching 100% accuracy by step 200,
- number and fraction ending at 100% accuracy at step 200,
- mean and median final loss,
- mean final accuracy,
- median first-success step among successful seeds,
- paired per-seed result table.

Because the comparison is paired, also report the seed-level win/tie counts for final success: seeds where only tanh succeeds, only ReLU succeeds, both succeed, or neither succeeds. This is the primary reliability comparison and avoids over-interpreting tiny differences in average loss.

Include one simple plot of loss versus training step aggregated over seeds if convenient. The numeric table and machine-readable metrics are authoritative; plotting is secondary.

## Completion criteria
The task is complete when all 20 paired seeds have been run for both activations and the results clearly determine whether the hypothesis is supported, rejected, or unresolved under this exact training budget.

Interpret the hypothesis narrowly:
- **supported** if tanh reaches 100% accuracy by step 200 on more paired seeds than ReLU;
- **rejected** if ReLU does so on more paired seeds than tanh;
- **unresolved/tied** if both succeed on the same number of seeds.

Do not change the experiment because one activation performs poorly. A negative or tied result is valid.

## Artifacts / results to preserve
Under `.relay/runs/research_20260912_141551_b985/executor/`, preserve:
- `RESULT.md` with methods, aggregate summary, paired seed-level comparison, interpretation, and hypothesis decision.
- `metrics.json` with machine-readable aggregate and per-seed results.
- the exact experiment source code, e.g. `xor_experiment.py`.
- any small plot(s) if generated.
- `handoff.json` containing this run ID, the copied task nonce, a newly generated evidence nonce, status, and result/artifact references.

Keep the implementation minimal and reproducible. This is primarily a relay-mechanics test, so avoid unrelated refactoring or production engineering.
