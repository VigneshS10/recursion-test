# XOR Initialization-Robustness Study

RUN_ID: research_20260912_142220_9376

## Method

The exact prior XOR setup was replicated for seeds `0..199`: four-point XOR, MLP `2 -> 4 -> 1`, hidden activation ReLU versus `tanh`, stable BCE-with-logits, full-batch Adam (`lr=0.01`) for exactly 200 updates. For each seed and initialization scheme, both activations received identical copied weights and zero biases. The two schemes were shared He-style (`sqrt(2/fan_in)`) and shared Xavier-style (`sqrt(2/(fan_in+fan_out))`). Loss trajectories were retained for seeds `0..9` only. PyTorch was unavailable; the bundled Python NumPy 2.3.5 fallback was used.

## Four-condition results

| initialization | activation | seeds ending at 100% | fraction | mean final loss | median final loss | mean final accuracy | median first-success step | final accuracy counts (50/75/100%) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| He | ReLU | 103 / 200 | 0.515 | 0.317943 | 0.359640 | 0.86250 | 56 | 13 / 84 / 103 |
| He | tanh | 175 / 200 | 0.875 | 0.160894 | 0.111259 | 0.94875 | 70 | 16 / 9 / 175 |
| Xavier | ReLU | 104 / 200 | 0.520 | 0.300661 | 0.319437 | 0.86375 | 42 | 13 / 83 / 104 |
| Xavier | tanh | 189 / 200 | 0.945 | 0.115497 | 0.0898506 | 0.97375 | 68 | 10 / 1 / 189 |

All 800 activation/initialization/seed runs completed. Final accuracy categories are exactly 0.50, 0.75, or 1.00 for every run.

## Paired analysis

Counts are over the same 200 seeds within each initialization scheme. The exact two-sided McNemar test is implemented as an exact binomial test on the discordant pairs: with `n = only_tanh + only_relu`, test `Binomial(n, 0.5)` and compute `min(1, 2 * P[X <= min(only_tanh, only_relu)])`.

| initialization | only tanh | only ReLU | both | neither | tanh minus ReLU success rate | exact two-sided p |
|---|---:|---:|---:|---:|---:|---:|
| He | 84 | 12 | 91 | 13 | +0.360 | 1.8316192854e-14 |
| Xavier | 90 | 5 | 99 | 6 | +0.425 | 3.0931950486e-21 |

## Decision

**Robust activation effect supported.** tanh has a higher perfect-XOR success probability under both shared initialization schemes, with large paired discordant-count asymmetry. The advantage is not initialization-dependent in direction: it is 0.875 versus 0.515 under He and 0.945 versus 0.520 under Xavier. The previous 20-seed result (tanh 16/20 versus ReLU 11/20) replicates at larger sample size under both schemes. Xavier changes the magnitude of tanh's advantage but does not remove or reverse it.

This conclusion is limited to the specified tiny XOR optimization problem, architecture, initialization rules, optimizer, and 200-step budget; it is not a general claim about activation functions or generalization.

## Evidence

- `xor_init_experiment.py`: exact source used for all 800 runs.
- `metrics.json`: machine-readable summaries, paired rows for all 200 seeds per scheme, final probabilities, and stepwise losses for trace seeds `0..9`.
- No plot was generated because matplotlib was unavailable; the numeric tables and JSON are authoritative.

