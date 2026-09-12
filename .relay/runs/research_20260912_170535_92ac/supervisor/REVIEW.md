# Review: XOR training-budget study

RUN_ID: research_20260912_170535_92ac

## What the executor actually did

The executor reran the committed XOR training-budget experiment exactly as requested using the NumPy implementation because PyTorch was unavailable. It evaluated 200 paired seeds (`0..199`) for ReLU and `tanh`, under both shared He-style and shared Xavier-style initialization, using the same `2 -> 4 -> 1` MLP, stable BCE-with-logits loss, full-batch Adam with learning rate `0.01`, and one uninterrupted 2,000-update trajectory per condition. Success was evaluated at checkpoints 25, 50, 100, 200, 500, 1,000, and 2,000 updates.

The source code confirms that the checkpoints come from a single continuous trajectory rather than separately restarted runs, that ReLU and `tanh` use identical initial parameters within each seed/initialization pair, and that first success is checked after every completed optimizer update.

## Observed evidence

The main result is that additional optimization time does not make ReLU catch up materially.

At 2,000 updates:

- He-ReLU: 107/200 successful (`0.535`).
- He-tanh: 189/200 successful (`0.945`).
- Xavier-ReLU: 106/200 successful (`0.530`).
- Xavier-tanh: 192/200 successful (`0.960`).

The tanh-minus-ReLU success-rate gap therefore remains about +0.41 under He and +0.43 under Xavier at 2,000 steps.

The paired exact McNemar/binomial tests remain overwhelmingly significant at long budgets:

- He at 2,000: only-tanh 88, only-ReLU 6, p ≈ `8.81e-20`.
- Xavier at 2,000: only-tanh 90, only-ReLU 4, p ≈ `3.22e-22`.

ReLU effectively plateaus after 200–500 updates. Under He it moves from 103/200 at step 200 to only 107/200 by step 500 and never improves afterward. Under Xavier it moves from 104/200 at step 200 to 106/200 by step 500 and then stops improving. By contrast, tanh continues to recover additional seeds after step 200 and reaches 94.5–96.0% success by step 2,000.

Among seeds failing at step 200, only 4/97 He-ReLU failures and 2/96 Xavier-ReLU failures ever recover by step 2,000, whereas 14/25 He-tanh failures and 3/11 Xavier-tanh failures recover.

There is one short-budget exception: under Xavier initialization, ReLU initially leads at 25 updates (14.5% vs 5.5% success; paired p ≈ 0.0029), and the curves are nearly tied at 50 updates. The ordering reverses decisively by 100 updates and then remains strongly in tanh's favor.

## Hypothesis decision

The original hypothesis proposed that tanh's advantage might mainly be a finite-budget effect and that ReLU could substantially catch up with more optimization. That hypothesis is **rejected** for this exact XOR setup.

The evidence instead supports a **persistent optimization-reliability advantage for tanh**: extending training from 200 to 2,000 steps barely changes ReLU's success fraction, while tanh remains much more reliable under both initialization schemes.

## Methodological assessment

The experiment is well controlled for the stated question:

- same dataset, architecture, optimizer, learning rate, and step budget;
- same 200 seeds across all conditions;
- identical paired initial parameters within each seed and initialization scheme;
- single uninterrupted trajectories across checkpoints;
- exact paired binary comparison at every checkpoint.

The NumPy fallback is acceptable because the task is a tiny deterministic MLP optimization study and the implementation explicitly reproduces Adam and BCE-with-logits. The main limitation is not correctness but scope: the result applies only to this tiny XOR network and setup.

A small code-quality oddity is that an unused `evaluate()` helper refers to `p[4]` even though `p` contains only four numeric arrays. It is never called, so it does not affect the reported experiment, but it should be removed or fixed in future code to avoid confusion.

## Interpretation

The plateau is the scientifically important result. ReLU failure is not predominantly a matter of simply needing more iterations. The failed ReLU seeds appear to enter states from which Adam at the fixed learning rate rarely escapes, while tanh continues to optimize successfully from most of the same paired initializations.

That points to a mechanistic follow-up rather than another budget extension. The most plausible simple mechanism to test is whether failed ReLU runs are associated with dead or permanently inactive hidden units, insufficient active hidden-unit diversity on the four XOR inputs, or zero/near-zero gradients. If so, the activation reliability gap can be explained directly. If not, then the problem is a more subtle optimization-basin effect.

## Why the next task is highest value

The initialization and budget ambiguities are now resolved. Repeating more seeds or extending beyond 2,000 steps would likely add little. The highest-value next step is therefore to diagnose **why** ReLU fails on roughly half the seeds.

The next task should instrument the same paired trajectories and compare successful versus failed ReLU seeds using hidden-unit activation patterns, number of active/dead units, gradient norms, and parameter-motion statistics. It should use the already-established 200 seeds and both initialization schemes, without introducing new architectures or optimizers. This keeps the relay experiment small while converting the observed performance difference into a mechanistic explanation.
