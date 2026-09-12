# Review: LeakyReLU rescue on XOR

## What the executor actually did

The executor ran the requested paired causal intervention on the established XOR setup using seeds `0..199`, both shared He and shared Xavier initialization, a `2 -> 4 -> 1` MLP, full-batch Adam with learning rate `0.01`, stable BCE-with-logits, and continuous training to 2,000 updates. The only intervention was replacing standard ReLU with `LeakyReLU(0.01)` while holding the initial parameters fixed within each seed/initialization pair.

The implementation also preserved the requested mechanistic diagnostics at steps 0, 25, 50, 100, 200, 500, 1,000, and 2,000: sign-pattern diversity, hidden variance, gradient/update norms, parameter displacement, ReLU dead-unit count, and the analogous all-negative-unit count for LeakyReLU.

## Observed evidence

The intervention produced a large and consistent reliability gain.

At 2,000 updates:

- He initialization: ReLU succeeded on 53.5% of seeds; LeakyReLU succeeded on 92.0%, a +38.5 percentage-point gain.
- Xavier initialization: ReLU succeeded on 53.0%; LeakyReLU succeeded on 93.5%, a +40.5 point gain.

The paired contrasts were especially strong:

- He at 2,000: 77 seeds were Leaky-only successes, 0 were ReLU-only successes, 107 both, 16 neither; exact paired p = `1.32e-23`.
- Xavier at 2,000: 81 Leaky-only, 0 ReLU-only, 106 both, 13 neither; exact paired p = `8.27e-25`.

The rescue already appears by 200 updates, where LeakyReLU improves success by +24.0 points under He and +27.0 under Xavier.

Mechanistically, standard ReLU remains heavily affected by dead units: at step 2,000 the any-dead fraction is 0.835 under He and 0.860 under Xavier. The matched Leaky condition has substantially fewer all-negative units by the end, despite starting from the same parameters, and continues to improve because negative-side derivatives remain nonzero.

Most importantly, among seeds that standard ReLU failed:

- He: LeakyReLU rescued 77/93 = 82.8%.
- Xavier: LeakyReLU rescued 81/94 = 86.2%.

Rescued seeds were also enriched for the exact pathology identified in the previous diagnostic. Among rescued seeds, 68.8% (He) and 76.5% (Xavier) had at least two dead ReLU units at step 25, versus 31.3% and 46.2% among the residual non-rescued failures.

## Hypothesis decision

**Strongly supported, with a residual mechanism remaining.**

The intervention directly changes the negative-side gradient from zero to a small nonzero value while holding the paired initialization, optimizer, architecture, data, and training budget fixed. The very large rescue rate, together with the enrichment of early dead-unit collapse among rescued seeds, provides strong causal evidence that zero-gradient dead-unit trapping is a major driver of standard ReLU failure in this tiny XOR problem.

It is not the whole story. Even with LeakyReLU, 16 He seeds and 13 Xavier seeds remain failures at 2,000 steps. Therefore a residual optimization-basin or representation/conditioning effect remains.

## Methodological issues and unexpected findings

1. The result is based on the NumPy fallback rather than PyTorch. This is acceptable for the relay-mechanics study because the executor preserved the exact source and matched conditions, but the conclusion is specific to this implementation.
2. The executor reused the previous run's `evidence_nonce` (`a2725fe3b43846959a9161e603965e99`) rather than generating a fresh one. The committed evidence itself is readable and internally consistent, so this does not invalidate the scientific result, but it is a relay-protocol hygiene problem because evidence nonces should uniquely identify each executor handoff.
3. The strongest unexpected finding is that the causal intervention is nearly monotone in paired outcomes: at step 2,000 there are zero seeds under either initialization where ReLU succeeds and LeakyReLU fails. That is stronger than merely observing a mean success-rate increase.

## Interpretation

The sequence of experiments now supports a coherent mechanism:

1. tanh is much more reliable than ReLU on this fixed XOR setup;
2. the gap persists under He/Xavier initialization and long training budgets;
3. failed ReLU runs are enriched for dead units, reduced activation-pattern diversity, and smaller first-layer movement;
4. restoring a small nonzero negative-side derivative rescues most failed ReLU seeds.

This is strong evidence that dead-unit trapping is a major causal mechanism rather than a passive correlate.

The remaining scientific uncertainty is now narrower: is the rescue essentially a binary effect of making the negative derivative nonzero, or does rescue quality depend materially on the magnitude of the leak? That distinction matters because it separates a pure dead-gradient mechanism from a broader conditioning/optimization effect.

## Why the next task is highest value

The highest-value next experiment is a small **negative-slope dose-response** study on exactly the same paired seeds and setup. Compare ReLU (`slope=0`) against a few fixed LeakyReLU slopes spanning orders of magnitude. If even an extremely tiny nonzero slope produces most of the rescue, that strongly supports the dead-gradient explanation. If success increases progressively with slope, then improved conditioning and optimization geometry matter in addition to merely avoiding exact zero gradients.

This is a much sharper next step than adding another activation family or changing architecture, optimizer, or width, because it directly probes the causal mechanism established by the current result.
