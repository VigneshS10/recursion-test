# Research Task: Causally test whether preventing dead ReLU units rescues XOR optimization

## Research question
Does replacing standard ReLU with a small-leak LeakyReLU substantially increase XOR success probability by preventing hidden units from becoming permanently inactive, or does the large failure rate persist despite preserving gradient flow on the negative side?

## Hypothesis
If dead/inactive hidden units are a major causal mechanism behind the persistent ReLU failures, then a small negative slope should materially improve optimization reliability while reducing hidden-unit collapse. LeakyReLU should therefore achieve a higher perfect-XOR success rate than standard ReLU under the same paired initializations and should show greater hidden-pattern diversity, larger first-layer movement, and fewer effectively inactive units. If the success rate remains close to ReLU despite preserved gradient flow, then the residual optimization-basin geometry is likely more important than dead units themselves.

## Experiment / implementation request
Continue the exact established XOR setup and change only the hidden activation.

Use:
- XOR inputs `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)` with labels `0,1,1,0`.
- MLP `2 -> 4 -> 1`.
- Adam with learning rate `0.01`.
- Stable BCE-with-logits loss.
- Full-batch training.
- Shared He-style and shared Xavier-style initialization exactly as in prior runs.
- Paired seeds `0..199`.
- Continuous training for 2,000 updates.

Compare exactly:
1. standard ReLU;
2. LeakyReLU with fixed negative slope `0.01`.

For each seed and initialization scheme, standard ReLU and LeakyReLU must start from exactly identical weights and biases. Do not retune learning rate, width, optimizer, initialization, or training budget for either activation.

Instrument checkpoints at steps 0, 25, 50, 100, 200, 500, 1,000, and 2,000.

For every run/checkpoint, record:
- loss and classification accuracy;
- first update reaching 100% accuracy, if any;
- hidden pre-activations and hidden outputs on all four XOR points;
- number of distinct binary sign signatures across the four inputs;
- hidden-output variance;
- gradient norms for `W1`, `b1`, `W2`, `b2`;
- Adam update norms for the same blocks;
- cumulative displacement from initialization for `W1` and `W2`.

For standard ReLU, preserve the same dead-unit definition used previously: a hidden unit is dead when its pre-activation is `<= 0` on all four XOR inputs.

For LeakyReLU, do not call such units "dead" because gradients remain nonzero. Instead record the analogous count of units whose pre-activation is `<= 0` on all four inputs, labeled clearly as `all_negative_unit_count`. This allows a matched structural comparison without conflating it with zero-gradient death.

The primary endpoint is perfect-XOR success by 2,000 updates. Also report success at 25, 50, 100, 200, 500, and 1,000 updates so the rescue timing is visible.

## Controls
- Same 200 seeds for every condition.
- Identical initial parameters for ReLU and LeakyReLU within each seed and initialization scheme.
- Same He and Xavier initialization formulas as previous runs.
- Same optimizer, learning rate, architecture, dataset, loss, and continuous 2,000-step budget.
- No post-hoc tuning of LeakyReLU slope or other hyperparameters.
- Do not add tanh as a primary condition; prior runs have already established the tanh reference. It may be quoted from prior evidence for context, but this run should focus on the causal ReLU rescue question.

## Expected measurements
For both initialization schemes, report:
- ReLU and LeakyReLU success fraction at every checkpoint;
- paired counts: only LeakyReLU succeeds, only ReLU succeeds, both, neither;
- activation success-rate difference and exact paired McNemar/binomial p-value at step 2,000, and preferably at step 200 as a secondary checkpoint;
- distribution of first-success steps;
- number of seeds failing at 2,000 updates.

Mechanistically compare successful and failed runs using:
- ReLU dead-unit count versus LeakyReLU all-negative-unit count;
- number of distinct activation/sign signatures;
- hidden-output variance;
- early W1 gradient and update norms;
- W1/W2 cumulative displacement.

Most importantly, analyze the exact seeds that failed with standard ReLU in the previous mechanism study. Report what fraction of those same seeds are rescued by LeakyReLU under He and Xavier initialization, and whether rescued seeds are enriched for cases that previously had two or more dead ReLU units by step 25.

## Completion criteria
The task is complete when all 200 paired seeds have been evaluated under ReLU and LeakyReLU for both initialization schemes and the evidence supports one of these conclusions:

- **Causal dead-unit mechanism strongly supported:** LeakyReLU materially raises success probability, especially rescuing seeds with early dead-unit collapse under ReLU, while maintaining gradient flow and hidden diversity.
- **Partial causal support:** LeakyReLU improves reliability but leaves a substantial residual failure rate, consistent with dead units explaining only part of the optimization problem.
- **Dead-unit causal explanation weakened:** LeakyReLU produces little or no reliability improvement despite eliminating zero-gradient death, pointing to optimization-basin geometry or another mechanism.

Do not vary the negative slope or add other interventions in this run. A single fixed rescue intervention is enough to answer the causal question cleanly.

## Artifacts / results to preserve
Under this run's executor directory, preserve:
- `RESULT.md` with methods, paired rescue results, mechanism comparison, and conclusion;
- `metrics.json` or compact CSV/JSON with per-seed checkpoint measurements;
- the exact source code used;
- any small summary plot if convenient;
- `handoff.json` with this run ID, copied task nonce, fresh evidence nonce, status, and artifact references.

Keep the implementation minimal and reproducible. This remains a relay-mechanics research test, so avoid unrelated engineering.
