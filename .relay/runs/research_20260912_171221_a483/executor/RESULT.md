# XOR ReLU Failure Mechanism Diagnosis

RUN_ID: research_20260912_171221_a483

## Method

All 200 paired seeds (`0..199`) were followed continuously for 2,000 updates under both shared He and shared Xavier initialization. The setup was unchanged: XOR, MLP `2 -> 4 -> 1`, ReLU/tanh, stable BCE-with-logits, full-batch Adam at `0.01`. Hidden patterns, dead/always-active units, signature diversity, hidden variance, gradient norms, Adam update norms, displacement, loss, and accuracy were recorded at steps `0,25,50,100,200,500,1000,2000`. PyTorch was unavailable; bundled NumPy 2.3.5 was used.

ReLU groups are stratified by eventual success at step 2,000. Tanh summaries use the same seed groups as a reference. At checkpoint `k`, gradients are recomputed at the post-update parameters (before the next update); update norms are the actual Adam displacement applied at update `k` (zero at step 0).

## ReLU success/failure comparison

Group sizes: He success `n=107`, failure `n=93`; Xavier success `n=106`, failure `n=94`.

| init | group | step | mean dead units | median dead | any dead | mean signatures | hidden variance | median W1 grad | median W1 update | median W1 displacement |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| He | success | 0 | 0.738 | 1 | 0.607 | 3.299 | 0.273662 | 0.156563 | 0.000000 | 0.000000 |
| He | failure | 0 | 1.355 | 1 | 0.806 | 3.118 | 0.190089 | 0.142904 | 0.000000 | 0.000000 |
| He | success | 25 | 0.888 | 1 | 0.701 | 3.121 | 0.219622 | 0.099977 | 0.016610 | 0.458182 |
| He | failure | 25 | 1.796 | 2 | 0.914 | 2.688 | 0.130709 | 0.081048 | 0.014545 | 0.413207 |
| He | success | 200 | 1.009 | 1 | 0.748 | 3.037 | 0.526878 | 0.073490 | 0.011101 | 2.674762 |
| He | failure | 200 | 2.065 | 2 | 0.935 | 2.183 | 0.233237 | 0.032721 | 0.005167 | 1.576320 |
| He | success | 2000 | 1.019 | 1 | 0.748 | 3.243 | 1.720743 | 0.000673 | 0.000582 | 5.856202 |
| He | failure | 2000 | 2.065 | 2 | 0.935 | 2.172 | 0.482845 | 0.000466 | 0.000381 | 2.604115 |
| Xavier | success | 0 | 0.764 | 1 | 0.623 | 3.302 | 0.092965 | 0.120603 | 0.000000 | 0.000000 |
| Xavier | failure | 0 | 1.319 | 1 | 0.787 | 3.117 | 0.061692 | 0.102382 | 0.000000 | 0.000000 |
| Xavier | success | 25 | 1.000 | 1 | 0.745 | 2.849 | 0.071319 | 0.094180 | 0.016015 | 0.425785 |
| Xavier | failure | 25 | 1.947 | 2 | 0.926 | 2.617 | 0.035384 | 0.066140 | 0.013248 | 0.373098 |
| Xavier | success | 200 | 1.047 | 1 | 0.774 | 3.028 | 0.401420 | 0.068542 | 0.010985 | 2.907897 |
| Xavier | failure | 200 | 2.085 | 2 | 0.957 | 2.255 | 0.141560 | 0.036494 | 0.004730 | 1.673084 |
| Xavier | success | 2000 | 1.047 | 1 | 0.774 | 3.198 | 1.352265 | 0.000721 | 0.000544 | 6.031464 |
| Xavier | failure | 2000 | 2.085 | 2 | 0.957 | 2.170 | 0.314095 | 0.000504 | 0.000373 | 2.721713 |

At step 25, having two dead ReLU units predicts failure with probability 0.649 (He) or 0.620 (Xavier); three or four dead units predicts failure with probability 1.000 in both schemes. At initialization, failure probabilities for dead counts 0/1/2/3/4 were He `0.300/0.407/0.682/1.000/1.000` and Xavier `0.333/0.407/0.659/1.000/1.000`. At step 25, corresponding values were He `0.200/0.329/0.649/1.000/1.000` and Xavier `0.206/0.268/0.620/1.000/1.000`.

Signature diversity shows the same direction: at step 25, failure probabilities conditioned on 1/2/3/4 distinct signatures were He `1.000/0.537/0.476/0.296` and Xavier `0.923/0.412/0.494/0.381`. These are transparent conditional summaries, not causal estimates.

## Tanh reference on the same ReLU groups

Tanh does not receive a dead-unit label. At step 25, median W1 gradient norms for tanh on ReLU-success versus ReLU-failure seed groups were He `0.059262` versus `0.063805` and Xavier `0.046296` versus `0.038487`; hidden-output variance was He `0.174697` versus `0.168820` and Xavier `0.103093` versus `0.088364`. At step 200, tanh hidden variance was He `0.414075` versus `0.386683` and Xavier `0.407757` versus `0.385405` for the same groups. The large ReLU dead-unit separation is therefore not mirrored as a generic seed-group split in tanh activity.

## Discordant paired seeds

Seeds where tanh succeeds but ReLU fails numbered He `88` and Xavier `90`. Their full seed lists and checkpoint diagnostics are in `metrics.json`. These discordant ReLU trajectories are enriched for dead units and reduced signature diversity; they are the primary evidence for the mechanism diagnosis.

## Decision

**Dead/inactive-unit mechanism supported, but not exclusive.** ReLU failures are strongly associated with two or more dead hidden units by step 25, reduced activation-pattern diversity, lower hidden variance, and smaller early first-layer gradients/updates and cumulative displacement. However, some failures retain zero or one dead unit and substantial signatures, so dead units do not explain every failure; a residual optimization-basin component remains plausible. No rescue intervention was attempted.

## Evidence

- `xor_mechanism_diagnostic.py`: exact instrumented source.
- `metrics.json`: per-seed/per-checkpoint diagnostics for all 800 runs, conditional failure tables, and discordant seed groups.
- No plot was generated; numeric JSON/tables are authoritative.

