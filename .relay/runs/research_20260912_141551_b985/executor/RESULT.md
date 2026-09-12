# XOR ReLU vs tanh Reliability

RUN_ID: research_20260912_141551_b985

## Method

- Dataset: four XOR points `(0,0),(0,1),(1,0),(1,1)` with labels `0,1,1,0`; full batch, no split.
- Model: `2 -> 4 -> 1` MLP; the only varied component is hidden activation (`ReLU` or `tanh`).
- Loss: stable BCE-with-logits equivalent; no sigmoid is applied before the loss.
- Optimizer: Adam, learning rate `0.01`, exactly 200 updates.
- Seeds: paired seeds `0..19`; RNG reset immediately before constructing each activation model.
- Step convention: `first_success_step` is the first completed update (1--200) with 100% accuracy. Final accuracy is measured after update 200 and is not replaced by best-seen accuracy.
- Runtime: bundled Python with NumPy 2.3.5. PyTorch and matplotlib were unavailable, so the experiment uses the minimal NumPy fallback and no secondary plot. Per-step losses are preserved in `metrics.json`.

## Aggregate results

| activation | reached 100% by step 200 | fraction | ended at 100% | mean final loss | median final loss | mean final accuracy | median first-success step |
|---|---:|---:|---:|---:|---:|---:|---:|
| ReLU | 11 / 20 | 0.55 | 11 / 20 | 0.331444 | 0.360002 | 0.8375 | 53 |
| tanh | 16 / 20 | 0.80 | 16 / 20 | 0.183978 | 0.131043 | 0.9250 | 90 |

Paired final-success counts: only tanh `6`, only ReLU `1`, both `10`, neither `3`.

## Paired seed-level results

`first` columns report the first completed update reaching 100%, or `—` if never reached.

| seed | ReLU loss | ReLU acc | ReLU first | tanh loss | tanh acc | tanh first |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.354474 | 1.00 | 190 | 0.372384 | 0.50 | — |
| 1 | 0.405139 | 0.75 | — | 0.200792 | 1.00 | 113 |
| 2 | 0.099984 | 1.00 | 12 | 0.099507 | 1.00 | 33 |
| 3 | 0.266149 | 1.00 | 128 | 0.139424 | 1.00 | 133 |
| 4 | 0.371335 | 0.75 | — | 0.108261 | 1.00 | 90 |
| 5 | 0.416393 | 1.00 | 176 | 0.088749 | 1.00 | 37 |
| 6 | 0.693147 | 0.50 | — | 0.108284 | 1.00 | 91 |
| 7 | 0.044290 | 1.00 | 30 | 0.071068 | 1.00 | 46 |
| 8 | 0.115075 | 1.00 | 61 | 0.081362 | 1.00 | 18 |
| 9 | 0.693147 | 0.50 | — | 0.377160 | 0.75 | — |
| 10 | 0.056462 | 1.00 | 38 | 0.058108 | 1.00 | 51 |
| 11 | 0.494938 | 0.75 | — | 0.232657 | 1.00 | 163 |
| 12 | 0.492102 | 0.75 | — | 0.359743 | 1.00 | 114 |
| 13 | 0.365531 | 0.75 | — | 0.151531 | 1.00 | 95 |
| 14 | 0.093398 | 1.00 | 33 | 0.099060 | 1.00 | 28 |
| 15 | 0.693147 | 0.50 | — | 0.368749 | 0.75 | — |
| 16 | 0.693147 | 0.50 | — | 0.389490 | 0.50 | — |
| 17 | 0.090802 | 1.00 | 44 | 0.189466 | 1.00 | 138 |
| 18 | 0.078943 | 1.00 | 59 | 0.122663 | 1.00 | 101 |
| 19 | 0.111275 | 1.00 | 53 | 0.061093 | 1.00 | 44 |

## Decision

**Hypothesis supported under this exact budget.** tanh reached and ended at 100% accuracy on 16/20 paired seeds (0.80), versus 11/20 (0.55) for ReLU. The paired comparison favors tanh on six seeds and ReLU on one. This is a narrow optimization-reliability result for this initialization, architecture, optimizer, and 200-step budget; it is not a generalization claim.

## Reproducibility and evidence

- Source: `xor_experiment.py` in this executor directory.
- Machine-readable results: `metrics.json`, including all 40 runs and every step's loss.
- No checkpoint, dataset, cache, or environment directory was generated.

