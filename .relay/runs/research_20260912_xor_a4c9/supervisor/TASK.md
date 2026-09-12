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
- Use a sigmoid-compatible binary output objective, preferably `BCEWithLogitsLoss` with no explicit sigmoid during training.
- Keep all architecture details identical except activation.

Training:
- Use Adam with learning rate `0.01`.
- Train for exactly 200 optimization steps per seed.
- Use full-batch training on all four XOR points.
- Run 20 seeds, seeds `0` through `19`.
- For each seed, use the same seed for the ReLU and tanh conditions before model initialization so the comparison is paired as closely as practical.
- Do not tune learning rate, width, optimizer, or step budget separately by activation after seeing results.

For every activation and seed, record:
- final BCE loss,
- final predictions/probabilities on the four points,
- final classification accuracy,
- whether the run achieved exactly 100% accuracy,
- first training step at which 100% accuracy was reached, if ever.

Also preserve the loss trajectory at a lightweight cadence (for example every 10 steps) so obvious optimization differences can be inspected.

## Controls
- Same XOR dataset and target encoding.
- Same architecture width and output layer.
- Same optimizer, learning rate, number of steps, and full-batch update rule.
- Same 20 seeds, paired across activations.
- No activation-specific tuning.
- CPU execution is sufficient; do not add unnecessary engineering.

## Expected measurements
Produce a compact summary comparing ReLU and tanh:
- number and fraction of seeds reaching 100% accuracy by step 200,
- mean and median final loss,
- mean final accuracy,
- median first-success step among successful seeds,
- paired per-seed result table.

Include one simple plot of loss versus training step aggregated over seeds if convenient, and optionally a per-seed success plot. The numeric table is authoritative; plotting is secondary.

## Completion criteria
The task is complete when all 20 paired seeds have been run for both activations and the results clearly determine whether the hypothesis is supported, rejected, or unresolved under this exact training budget.

Do not change the experiment because one activation performs poorly. A negative or tied result is valid.

## Artifacts / results to preserve
Under this run's executor directory, preserve:
- `RESULT.md` with methods, result table/summary, interpretation, and whether the hypothesis was supported/rejected/unresolved.
- `metrics.json` with machine-readable aggregate and per-seed results.
- the experiment source code, e.g. `xor_experiment.py`.
- any small plot(s) if generated.
- `handoff.json` containing the run ID, copied task nonce, a newly generated evidence nonce, status, and result/artifact references.

Keep the implementation minimal and reproducible. This is primarily a relay-mechanics test, so avoid unrelated refactoring or production engineering.
