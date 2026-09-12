# Research Task: Diagnose why ReLU fails on XOR seeds that tanh solves

## Research question
Are the persistent ReLU failures in the tiny XOR MLP explained by dead/inactive hidden units and vanishing parameter updates, or do failed runs retain active units and gradients yet remain trapped in a different optimization basin?

## Hypothesis
A substantial fraction of failed ReLU seeds will show a characteristic loss of useful hidden-unit activity: one or more hidden units will be inactive on all four XOR inputs for most or all of training, or the network will collapse to too few distinct activation patterns to represent XOR effectively. These failed runs should also show smaller effective gradient/update norms than successful ReLU runs. If failed runs instead maintain comparable hidden activity and gradients, then the failure is not primarily a dead-ReLU mechanism and is more likely due to a different optimization-basin geometry.

## Experiment / implementation request
Use the exact XOR setup already established in the previous runs. Do not change architecture, optimizer, learning rate, dataset, initialization formulas, or seed set.

Use:
- XOR inputs `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)` with labels `0,1,1,0`.
- MLP `2 -> 4 -> 1`.
- ReLU and `tanh` hidden activations.
- Stable BCE-with-logits loss.
- Adam, learning rate `0.01`.
- Shared He-style and shared Xavier-style initialization.
- Paired seeds `0..199`.
- Train continuously for 2,000 updates as before.

The main analysis should focus on ReLU, stratified by whether the run ever reaches 100% XOR accuracy by step 2,000. Use tanh as a reference condition for the same paired seeds rather than as the primary object of diagnosis.

Instrument the hidden layer and optimization trajectory at fixed checkpoints: initialization (step 0), 25, 50, 100, 200, 500, 1,000, and 2,000 updates.

For every seed/checkpoint, record at minimum:

1. **Hidden activation pattern** on all four XOR inputs for each of the four hidden units.
2. **Dead-unit count** for ReLU: number of hidden units whose pre-activation is `<= 0` on all four XOR inputs and therefore output zero on all examples.
3. **Always-active count** for ReLU: hidden units whose pre-activation is `> 0` on all four points.
4. **Number of distinct binary ReLU activation signatures** across the four XOR inputs.
5. Hidden-unit output variance across the four XOR points.
6. Gradient norm for each parameter block (`W1`, `b1`, `W2`, `b2`) before the optimizer update at the checkpoint.
7. Adam update norm for each parameter block at the checkpoint, or equivalently parameter displacement per step if easier to record faithfully.
8. Cumulative parameter displacement from initialization for `W1` and `W2`.
9. Current loss and accuracy.

For tanh, record corresponding hidden-output variance, gradient/update norms, and cumulative parameter displacement; a dead-unit count is not meaningful for tanh and should not be invented.

Do not add new activations or interventions in this run. This is a diagnostic study, not a rescue experiment.

## Controls
- Same exact paired seeds and initial parameters already used in the prior study.
- Same He and Xavier initialization formulas.
- Same optimizer and learning rate.
- Same continuous 2,000-update trajectories.
- Same success definition: 100% classification accuracy on all four XOR points.
- No retraining with altered hyperparameters after examining failures.

## Expected measurements
For ReLU under each initialization scheme, compare successful and failed seeds using:

- distribution of dead-unit count at each checkpoint,
- fraction of runs with at least one dead hidden unit,
- distribution of distinct activation-signature count,
- hidden-output variance,
- gradient norms and update norms,
- cumulative parameter displacement,
- loss and accuracy trajectories.

Report simple effect sizes or transparent comparisons rather than elaborate statistics. At minimum include:

- median/mean dead-unit count for successful vs failed ReLU runs at each checkpoint,
- probability of eventual failure conditioned on dead-unit count at initialization and at step 25,
- probability of eventual failure conditioned on number of distinct activation signatures,
- median first-layer gradient norm for successful vs failed runs at early checkpoints,
- corresponding tanh reference summaries on the same seed groups.

Also inspect the subset of discordant paired seeds where tanh succeeds by 2,000 steps and ReLU does not. Determine whether their ReLU trajectories show systematically different hidden activity from seeds where both activations succeed.

A compact table plus one or two simple diagnostic plots is sufficient if plotting is available; numeric JSON/CSV remains authoritative.

## Completion criteria
The task is complete when all 200 seeds under both initialization schemes have been instrumented and the evidence supports one of these conclusions:

- **Dead/inactive-unit mechanism supported:** ReLU failure is strongly associated with dead units, low activation-pattern diversity, and/or near-zero early gradients/updates.
- **Mechanism only partial:** dead/inactive units explain a subset of failures, but many failed seeds retain substantial hidden activity and gradients.
- **Dead-unit hypothesis rejected:** failed ReLU runs generally retain active hidden units and comparable gradient/update magnitudes, implying a more subtle optimization-basin mechanism.

Do not add rescue interventions in this run. If a clear mechanism is identified, the following iteration can test a targeted intervention.

## Artifacts / results to preserve
Under this run's executor directory, preserve:
- `RESULT.md` with the successful-vs-failed diagnostic comparison and mechanism conclusion.
- `metrics.json` or compact CSV/JSON with per-seed checkpoint diagnostics.
- the exact source code used.
- any small diagnostic plot(s) if generated.
- `handoff.json` with this run ID, copied task nonce, fresh evidence nonce, status, and artifact references.

Keep the implementation minimal and reproducible. This remains a relay-mechanics research test, so avoid unrelated engineering.
