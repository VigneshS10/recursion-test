# LeakyReLU rescue on XOR

Run ID: `research_20260912_172043_8b1c`

## Setup

The committed task was executed exactly as specified: XOR (four points), a 2→4→1 MLP, stable BCE-with-logits, full-batch Adam (learning rate 0.01), 2,000 updates, checkpoints 0/25/50/100/200/500/1000/2000, and paired seeds 0–199. Each seed uses the same He or Xavier initialization for ReLU and LeakyReLU (negative slope 0.01). The NumPy fallback implementation was used because PyTorch was unavailable. Per-seed diagnostics are in `metrics.json`; the reproducible runner is `xor_leaky_rescue.py`.

## Success rates (exact XOR accuracy = 1.0)

| init | activation | 25 | 50 | 100 | 200 | 500 | 1000 | 2000 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| He | ReLU | 7.0% | 21.5% | 37.5% | 51.5% | 53.5% | 53.5% | 53.5% |
| He | LeakyReLU | 9.0% | 28.0% | 53.0% | 75.5% | 85.5% | 89.5% | 92.0% |
| Xavier | ReLU | 14.5% | 30.0% | 43.5% | 52.0% | 53.0% | 53.0% | 53.0% |
| Xavier | LeakyReLU | 16.0% | 40.5% | 66.0% | 79.0% | 88.5% | 93.0% | 93.5% |

At 2,000 updates the Leaky-minus-ReLU improvement is +38.5 percentage points (He) and +40.5 points (Xavier). At 200 updates it is +24.0 and +27.0 points, respectively.

## Paired tests

| init/checkpoint | Leaky only | ReLU only | both | neither | Δ success | exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| He / 200 | 49 | 1 | 102 | 48 | +0.240 | 9.06e-14 |
| He / 2000 | 77 | 0 | 107 | 16 | +0.385 | 1.32e-23 |
| Xavier / 200 | 54 | 0 | 104 | 42 | +0.270 | 1.11e-16 |
| Xavier / 2000 | 81 | 0 | 106 | 13 | +0.405 | 8.27e-25 |

## Mechanistic diagnostics

At checkpoint 25, mean dead ReLU units (preactivation ≤ 0 on all four examples) were 1.310 (He) and 1.445 (Xavier); the corresponding Leaky all-negative counts were 0.900 and 1.010. By checkpoint 2000, ReLU remained at 1.505/1.535 dead units with any-dead fractions 0.835/0.860, whereas Leaky fell to 0.560/0.545 all-negative units with any-all-negative fractions 0.455/0.440. Despite similar early hidden variance, Leaky retained nonzero negative-side derivatives and continued improving: final mean accuracy/loss was 0.961/0.045 (He) and 0.966/0.037 (Xavier), versus 0.868/0.211 and 0.865/0.213 for ReLU.

## Rescue and enrichment

| init | prior ReLU failures | rescued by Leaky | rescue fraction | early dead≥2 among rescued | among non-rescued |
|---|---:|---:|---:|---:|---:|
| He | 93 | 77 | 82.8% | 68.8% (53/77) | 31.3% (5/16) |
| Xavier | 94 | 81 | 86.2% | 76.5% (62/81) | 46.2% (6/13) |

## Conclusion

The intervention-only contrast strongly supports the proposed causal mechanism: preventing zero negative-side gradients rescues most paired ReLU failures and yields a large, statistically decisive success gain under both initializations. It is not universal—16 He and 13 Xavier runs still fail with LeakyReLU—so the result supports dead-unit trapping as a major cause, not the sole cause of XOR failure. No large checkpoints, datasets, caches, or environment directories were created.
