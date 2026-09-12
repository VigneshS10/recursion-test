# Research Task: Measure how the XOR activation advantage changes with training budget

## Research question
Does tanh's higher XOR success probability persist across training budgets, or does ReLU catch up when given more optimization steps?

## Hypothesis
The tanh advantage observed at 200 updates is primarily a finite-budget reliability effect: tanh should outperform ReLU at short and medium budgets, while the gap may shrink at longer budgets if ReLU eventually reaches successful solutions from many of the seeds that fail by step 200. If the gap remains large even at long budgets, then the activation difference reflects a more persistent optimization reliability effect rather than only slower convergence.

## Experiment / implementation request
Continue the exact XOR setup from the previous run. Keep the dataset, architecture, loss, optimizer, learning rate, initialization schemes, and paired seeds unchanged. Vary only the optimization budget.

Use:
- XOR inputs `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)` with labels `0,1,1,0`.
- MLP `2 -> 4 -> 1`.
- Hidden activations: exactly ReLU and `tanh`.
- Stable BCE-with-logits loss.
- Adam with learning rate `0.01`.
- Shared He-style and shared Xavier-style initialization exactly as in the previous run.
- Paired seeds `0..199`.

Run each activation × initialization × seed condition once up to **2000 updates**, and record success at the following fixed checkpoints without restarting training:

- 25
- 50
- 100
- 200
- 500
- 1000
- 2000 updates

For every run, record:
- first update at which 100% XOR accuracy is reached, if ever,
- whether 100% accuracy has been reached by each checkpoint,
- final accuracy and loss at each checkpoint,
- final probabilities on the four XOR points at the final 2000-step checkpoint.

Do not alter learning rate, optimizer, architecture, width, or initialization between budgets. Do not run separate independently initialized models for each checkpoint; the budget curve must come from one continuous trajectory per seed/condition so differences across checkpoints are directly interpretable.

## Controls
- Same exact 200 seeds for all conditions.
- Same paired initial parameters for ReLU and tanh within each seed and initialization scheme.
- Same He and Xavier formulas as the previous run.
- Same optimizer and learning rate throughout all 2000 updates.
- Same success definition: exactly 100% classification accuracy on all four XOR points.
- No post-hoc activation-specific tuning.

## Expected measurements
For every checkpoint and each of the four conditions (ReLU-He, tanh-He, ReLU-Xavier, tanh-Xavier), report:
- number and fraction of seeds that have reached 100% accuracy,
- mean classification accuracy,
- mean and median BCE loss,
- paired discordant success counts between tanh and ReLU within each initialization scheme.

For each checkpoint, report the activation success-rate difference (`tanh - ReLU`) separately for He and Xavier initialization and the exact paired McNemar/binomial p-value on discordant pairs.

Also report:
- empirical distribution of first-success steps for each condition,
- fraction of seeds that never reach 100% accuracy by 2000 updates,
- among seeds failing at step 200, how many subsequently recover by 500, 1000, or 2000 steps.

The key output should be a compact success-rate-versus-budget table. A simple line plot is optional if plotting is convenient; the numeric table is authoritative.

## Completion criteria
The task is complete when all 200 paired seeds have been followed continuously to 2000 updates under all four conditions and the evidence supports one of these conclusions:

- **Persistent tanh advantage:** tanh remains substantially more reliable than ReLU even at 1000-2000 updates.
- **Finite-budget advantage:** tanh is clearly better at short/medium budgets, but ReLU substantially catches up at longer budgets.
- **Complex/crossing behavior:** the relative advantage changes non-monotonically or differs materially between He and Xavier initialization.

Do not add new activations, widths, optimizers, or learning rates in this run. The goal is to isolate training-budget dependence of the already-established activation effect.

## Artifacts / results to preserve
Under this run's executor directory, preserve:
- `RESULT.md` with methods, checkpoint tables, paired analysis, interpretation, and hypothesis decision.
- `metrics.json` or compact CSV/JSON with aggregate and per-seed checkpoint results.
- the exact source code used.
- any small summary plot if convenient.
- `handoff.json` with this run ID, copied task nonce, fresh evidence nonce, status, and artifact references.

Keep the implementation minimal and reproducible. This remains a relay-mechanics research test, so avoid unrelated engineering.
