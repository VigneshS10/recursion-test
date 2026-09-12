# Research Task: Measure the LeakyReLU negative-slope dose response on XOR

## Research question
Is the large LeakyReLU rescue primarily a binary consequence of making the negative-side derivative nonzero, or does XOR optimization reliability improve gradually as the negative slope becomes larger?

## Hypothesis
If exact zero-gradient dead-unit trapping is the dominant mechanism, then even a very small nonzero negative slope should recover a large fraction of the reliability gap between standard ReLU and `LeakyReLU(0.01)`. If rescue instead scales strongly with leak magnitude, then improved conditioning and optimization geometry contribute materially beyond merely preventing exactly dead gradients.

## Experiment / implementation request
Continue the exact established XOR setup. Keep dataset, architecture, loss, optimizer, learning rate, initialization schemes, seed set, and training budget unchanged.

Use:
- XOR inputs `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)` with labels `0,1,1,0`.
- MLP `2 -> 4 -> 1`.
- Stable BCE-with-logits loss.
- Full-batch Adam with learning rate `0.01`.
- Shared He-style and shared Xavier-style initialization exactly as before.
- Paired seeds `0..199`.
- Continuous training for 2,000 updates.

Compare exactly these negative slopes, using the same piecewise-linear activation `f(x)=x` for `x>0` and `f(x)=alpha*x` otherwise:

- `alpha = 0` (standard ReLU control),
- `alpha = 1e-4`,
- `alpha = 1e-3`,
- `alpha = 1e-2` (the successful prior LeakyReLU condition),
- `alpha = 1e-1`.

For each initialization scheme and seed, all slope conditions must start from exactly identical weights and biases. Do not tune optimizer, learning rate, width, budget, or initialization by slope.

Instrument checkpoints at 0, 25, 50, 100, 200, 500, 1,000, and 2,000 updates.

For every run/checkpoint record:
- loss and classification accuracy;
- first update reaching 100% XOR accuracy, if any;
- number of units whose preactivation is `<= 0` on all four XOR inputs;
- number of distinct binary sign signatures across the four inputs;
- hidden-output variance;
- first-layer gradient norm and Adam update norm;
- cumulative `W1` displacement from initialization.

Use neutral terminology such as `all_negative_unit_count` for `alpha > 0`; reserve `dead_unit_count` for `alpha = 0`, where the derivative is truly zero.

## Controls
- Same 200 seeds for every slope and initialization scheme.
- Identical initial parameters across all slope values within each seed/scheme.
- Same He/Xavier formulas, dataset, architecture, optimizer, learning rate, and continuous 2,000-step trajectory.
- No post-hoc selection or retuning of slope values.
- Do not add tanh or other activation families in this run; the goal is to isolate the negative-slope dose response.

## Expected measurements
For each slope under He and Xavier initialization, report:
- success fraction at 25, 50, 100, 200, 500, 1,000, and 2,000 updates;
- mean/median final loss and mean final accuracy;
- median first-success step among successful runs;
- number of seeds never reaching 100% by 2,000;
- mean all-negative/dead-unit count, signature diversity, early W1 gradient/update norm, and cumulative W1 displacement.

At 2,000 updates, report paired success comparisons between each nonzero slope and the `alpha=0` ReLU control using exact discordant counts and exact McNemar/binomial p-values.

Also compute two targeted mechanism summaries:

1. **Rescue fraction among prior ReLU failures** for each nonzero slope.
2. Among seeds with at least two dead ReLU units at step 25, the fraction rescued by each slope.

The key output should be a compact success-rate-versus-log-slope table. If useful, include the fraction of the full `alpha=0.01` rescue already achieved at `1e-4` and `1e-3`.

## Completion criteria
The task is complete when all 200 paired seeds have been evaluated for all five slopes under both initialization schemes and the evidence supports one of these conclusions:

- **Near-binary dead-gradient effect:** very small nonzero slopes recover most of the LeakyReLU benefit, showing that avoiding exact zero gradients is the dominant causal factor.
- **Dose-dependent effect:** reliability improves substantially and systematically with slope magnitude, indicating that conditioning/optimization geometry matters beyond exact deadness.
- **Non-monotone/complex effect:** intermediate slopes outperform both tiny and large slopes, or the pattern differs materially between He and Xavier initialization.

Do not broaden the study until this mechanism distinction is resolved.

## Artifacts / results to preserve
Under the next run's executor directory, preserve:
- `RESULT.md` with methods, slope-response tables, mechanism interpretation, and hypothesis decision;
- `metrics.json` or compact CSV/JSON with per-seed checkpoint results;
- the exact source code used;
- any small summary plot if convenient;
- `handoff.json` with the next run ID, copied task nonce, a **fresh evidence nonce not reused from any earlier run**, status, and artifact references.

Keep the implementation minimal and reproducible. This remains a relay-mechanics research test, so avoid unrelated engineering.
