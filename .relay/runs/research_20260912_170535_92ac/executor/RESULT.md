# XOR Training-Budget Study (Replication)

RUN_ID: research_20260912_170535_92ac

## Method

The committed continuous NumPy experiment was rerun unchanged: 200 paired seeds (`0..199`), ReLU and `tanh`, shared He/Xavier initialization, XOR `2 -> 4 -> 1` MLP, stable BCE-with-logits, full-batch Adam (`lr=0.01`), and one uninterrupted 2,000-update trajectory per condition. Checkpoints were 25, 50, 100, 200, 500, 1,000, and 2,000 updates. PyTorch was unavailable; bundled NumPy 2.3.5 was used.

## Success rate versus budget

Each cell is successful seeds / 200 (fraction).

| initialization | activation | 25 | 50 | 100 | 200 | 500 | 1000 | 2000 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| He | ReLU | 14 (0.070) | 43 (0.215) | 75 (0.375) | 103 (0.515) | 107 (0.535) | 107 (0.535) | 107 (0.535) |
| He | tanh | 14 (0.070) | 54 (0.270) | 125 (0.625) | 175 (0.875) | 187 (0.935) | 188 (0.940) | 189 (0.945) |
| Xavier | ReLU | 29 (0.145) | 60 (0.300) | 87 (0.435) | 104 (0.520) | 106 (0.530) | 106 (0.530) | 106 (0.530) |
| Xavier | tanh | 11 (0.055) | 55 (0.275) | 146 (0.730) | 189 (0.945) | 191 (0.955) | 192 (0.960) | 192 (0.960) |

At 2,000 updates, mean accuracy/loss were He-ReLU `0.868 / 0.210669`, He-tanh `0.975 / 0.020816`, Xavier-ReLU `0.865 / 0.212871`, and Xavier-tanh `0.983 / 0.015148`.

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
| Xavier | 25 | 8 | 26 | 3 | 163 | -0.090 | 0.002935 |
| Xavier | 50 | 25 | 30 | 30 | 115 | -0.025 | 0.590053 |
| Xavier | 100 | 74 | 15 | 72 | 39 | 0.295 | 1.534206e-10 |
| Xavier | 200 | 90 | 5 | 99 | 6 | 0.425 | 3.093195e-21 |
| Xavier | 500 | 89 | 4 | 102 | 5 | 0.425 | 6.167247e-22 |
| Xavier | 1000 | 90 | 4 | 102 | 4 | 0.430 | 3.219068e-22 |
| Xavier | 2000 | 90 | 4 | 102 | 4 | 0.430 | 3.219068e-22 |

P-values are exact two-sided binomial tests on discordant pairs (McNemar exact), null probability 0.5.

## Recovery after step 200

Among seeds failing at 200, recovery by 500/1,000/2,000 was: He-ReLU `4/4/4` of 97; He-tanh `12/13/14` of 25; Xavier-ReLU `2/2/2` of 96; Xavier-tanh `2/3/3` of 11. Never-successful-by-2,000 counts were respectively `93`, `11`, `94`, and `8`.

## Decision

**Persistent tanh advantage supported.** The advantage remains large at 1,000--2,000 updates in both initialization schemes; ReLU catches up only marginally after 200 updates. The short-budget Xavier checkpoint briefly favors ReLU, but curves cross by 100 updates and the long-budget ordering is stable. This conclusion is limited to the specified XOR optimization setup.

## Evidence

- `xor_budget_experiment.py`: exact continuous-trajectory source used.
- `metrics.json`: all 800 runs, all checkpoints, final probabilities, first-success steps, and paired tests.
- No plot was generated; numeric JSON/tables are authoritative.

