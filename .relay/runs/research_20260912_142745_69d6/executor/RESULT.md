# XOR Training-Budget Study

RUN_ID: research_20260912_142745_69d6

## Method

The prior XOR setup was followed continuously (no restart at checkpoints) for 200 paired seeds (`0..199`), both ReLU and `tanh`, and shared He/Xavier initializations. The dataset, `2 -> 4 -> 1` MLP, stable BCE-with-logits, full-batch Adam (`lr=0.01`), and initialization formulas were unchanged. Each trajectory ran 2,000 updates; metrics were captured after updates 25, 50, 100, 200, 500, 1,000, and 2,000. PyTorch was unavailable, so the bundled NumPy 2.3.5 implementation was used.

## Success rate versus budget

Each cell is `successful seeds / 200` (fraction).

| initialization | activation | 25 | 50 | 100 | 200 | 500 | 1000 | 2000 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| He | ReLU | 14 (0.070) | 43 (0.215) | 75 (0.375) | 103 (0.515) | 107 (0.535) | 107 (0.535) | 107 (0.535) |
| He | tanh | 14 (0.070) | 54 (0.270) | 125 (0.625) | 175 (0.875) | 187 (0.935) | 188 (0.940) | 189 (0.945) |
| Xavier | ReLU | 29 (0.145) | 60 (0.300) | 87 (0.435) | 104 (0.520) | 106 (0.530) | 106 (0.530) | 106 (0.530) |
| Xavier | tanh | 11 (0.055) | 55 (0.275) | 146 (0.730) | 189 (0.945) | 191 (0.955) | 192 (0.960) | 192 (0.960) |

Mean accuracy and mean/median BCE loss at each checkpoint are preserved in `metrics.json`. At 2,000 updates, the corresponding means are: He-ReLU accuracy 0.868, loss 0.210669; He-tanh 0.975, loss 0.020816; Xavier-ReLU 0.865, loss 0.212871; Xavier-tanh 0.983, loss 0.015148.

## Paired tanh-minus-ReLU analysis

| initialization | checkpoint | only tanh | only ReLU | both | neither | rate difference | exact two-sided p |
|---|---:|---:|---:|---:|---:|---:|---:|
| He | 25 | 11 | 11 | 3 | 175 | 0.000 | 1.000000 |
| He | 50 | 33 | 22 | 21 | 124 | 0.055 | 0.177001 |
| He | 100 | 73 | 23 | 52 | 52 | 0.250 | 3.110232e-07 |
| He | 200 | 84 | 12 | 91 | 13 | 0.360 | 1.831619e-14 |
| He | 500 | 86 | 6 | 101 | 7 | 0.400 | 3.090495e-19 |
| He | 1000 | 87 | 6 | 101 | 6 | 0.405 | 1.650465e-19 |
| He | 2000 | 88 | 6 | 101 | 5 | 0.410 | 8.807937e-20 |
| Xavier | 25 | 8 | 26 | 3 | -0.090 | 0.002935 |
| Xavier | 50 | 25 | 30 | 30 | -0.025 | 0.590053 |
| Xavier | 100 | 74 | 15 | 72 | 0.295 | 1.534206e-10 |
| Xavier | 200 | 90 | 5 | 99 | 6 | 0.425 | 3.093195e-21 |
| Xavier | 500 | 89 | 4 | 102 | 5 | 0.425 | 6.167247e-22 |
| Xavier | 1000 | 90 | 4 | 102 | 4 | 0.430 | 3.219068e-22 |
| Xavier | 2000 | 90 | 4 | 102 | 4 | 0.430 | 3.219068e-22 |

The p-values use the exact two-sided binomial test on discordant pairs, null probability 0.5 (the McNemar exact test).

## Recovery after step 200 and first-success tails

| initialization | activation | failed at 200 | recovered by 500 | by 1000 | by 2000 | never successful by 2000 |
|---|---|---:|---:|---:|---:|---:|
| He | ReLU | 97 | 4 | 4 | 4 | 93 |
| He | tanh | 25 | 12 | 13 | 14 | 11 |
| Xavier | ReLU | 96 | 2 | 2 | 2 | 94 |
| Xavier | tanh | 11 | 2 | 3 | 3 | 8 |

Median first-success steps among all successful-by-2,000 runs are He-ReLU 59, He-tanh 76, Xavier-ReLU 43, and Xavier-tanh 68; full first-success values are in `metrics.json`.

## Decision

**Persistent tanh advantage supported.** tanh is substantially more reliable at 200 updates and remains so through 2,000 updates under both initialization schemes. ReLU catches up only marginally after 200 (He 51.5% to 53.5%; Xavier 52.0% to 53.0%), while tanh reaches 94.5% (He) and 96.0% (Xavier). The short-budget Xavier result at 25 updates slightly favors ReLU, but the curves cross by 100 updates and the long-budget advantage is large and stable. This is limited to the specified XOR optimization setup and is not a general activation or generalization claim.

## Evidence

- `xor_budget_experiment.py`: exact continuous-trajectory source.
- `metrics.json`: all 800 runs, checkpoint metrics, final probabilities, first-success steps, and paired tests.
- No plot was generated; numeric tables and JSON are authoritative.

