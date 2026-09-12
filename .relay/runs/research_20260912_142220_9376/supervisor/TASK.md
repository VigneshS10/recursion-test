# Research Task: Test whether the XOR activation effect is robust to initialization

## Research question
Does the apparent reliability advantage of `tanh` over ReLU on the tiny XOR MLP persist when the experiment is replicated over many more paired seeds and under both shared He-style and shared Xavier-style initialization?

## Hypothesis
If the activation itself is the main driver of the previous result, `tanh` should retain a higher probability of reaching perfect XOR accuracy than ReLU under both reasonable shared initialization schemes. If the advantage disappears or reverses under Xavier initialization, then the previous result was substantially initialization-dependent.

## Experiment / implementation request
Continue the exact XOR experiment from the previous run. Keep the dataset, architecture, loss, optimizer, learning rate, full-batch training, and 200-step budget unchanged.

Use:
- XOR inputs `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)` with labels `0,1,1,0`.
- MLP `2 -> 4 -> 1`.
- Hidden activations: exactly ReLU and `tanh`.
- Stable BCE-with-logits loss.
- Adam, learning rate `0.01`.
- Exactly 200 updates.
- Exactly 200 paired seeds, seeds `0..199`.

Cross activation with exactly two initialization conditions:

1. **Shared He-style initialization:** use the same rule as the previous run, with identical initial weights for ReLU and `tanh` within each seed.
2. **Shared Xavier-style initialization:** use a standard Xavier/Glorot-style scale, again with identical initial weights for ReLU and `tanh` within each seed.

For each initialization scheme and seed, both activations must start from exactly the same weights and biases. Do not tune learning rate, width, optimizer, initialization scale, or training budget separately by activation after seeing results.

For every activation × initialization × seed combination, record:
- final BCE loss,
- final classification accuracy,
- whether 100% accuracy was reached by step 200,
- whether the run ended at 100% accuracy,
- first success step if any,
- final probabilities on all four XOR points.

Classify final failed runs by accuracy, at minimum 50%, 75%, and 100%. Preserve lightweight loss trajectories only for a fixed predeclared subset of seeds (for example seeds `0..9`) to keep evidence compact.

## Controls
- Same XOR dataset and target encoding.
- Same architecture and optimizer.
- Same exact 200-step budget.
- Same seeds `0..199` for every condition.
- ReLU and `tanh` paired from identical initial parameters within each seed and initialization scheme.
- Initialization scheme is the only added independent variable beyond activation.
- No post-hoc activation-specific tuning.

## Expected measurements
For each of the four conditions (ReLU-He, tanh-He, ReLU-Xavier, tanh-Xavier), report:
- success count and fraction by step 200,
- mean and median final BCE loss,
- mean final accuracy,
- median first-success step among successful runs,
- distribution of final accuracies.

For each initialization scheme, report paired discordant counts:
- only `tanh` succeeds,
- only ReLU succeeds,
- both succeed,
- neither succeeds.

Also report the activation success-rate difference (`tanh - ReLU`) separately under He and Xavier initialization. Include a simple paired statistical comparison for binary outcomes, preferably an exact McNemar test or an equivalent exact binomial test on discordant pairs, and state the test definition clearly.

Produce a compact four-condition table and paired-analysis table sufficient to determine whether the previous 20-seed result replicates and whether the activation effect is stable across initialization schemes.

## Completion criteria
The task is complete when all 200 seeds have been run for all four conditions and the evidence supports one of these conclusions:

- **Robust activation effect:** `tanh` has a reproducibly higher success probability under both initialization schemes.
- **Initialization-dependent effect:** the ReLU-versus-`tanh` reliability difference materially changes, disappears, or reverses between He and Xavier initialization.
- **No clear effect:** larger-sample paired results do not support a stable difference under either scheme.

Do not add new activations, widths, optimizers, or budgets in this run. This experiment should resolve the initialization ambiguity before broadening the study.

## Artifacts / results to preserve
Under this run's executor directory, preserve:
- `RESULT.md` with methods, four-condition summary, paired analysis, interpretation, and hypothesis decision.
- `metrics.json` or compact CSV/JSON with aggregate and per-seed results.
- the exact source code used.
- any small summary plot if convenient.
- `handoff.json` with this run ID, copied task nonce, fresh evidence nonce, status, and artifact references.

Keep the implementation minimal and reproducible. This remains a relay-mechanics research test, so avoid unrelated refactoring or production engineering.
