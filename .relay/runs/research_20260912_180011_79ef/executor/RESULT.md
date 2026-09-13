# XOR LeakyReLU negative-slope dose response

Run ID: `research_20260912_180011_79ef`

## Methods

The established XOR experiment was continued without changing the dataset, 2→4→1 architecture, stable BCE-with-logits loss, full-batch Adam (learning rate 0.01), He/Xavier initialization, 200 paired seeds (0–199), or 2,000-update budget. The activation was `f(x)=x` for `x>0` and `alpha*x` otherwise, with `alpha ∈ {0, 1e-4, 1e-3, 1e-2, 1e-1}`. Within every seed and initialization, all slopes received identical initial parameters. The NumPy fallback implementation is `xor_slope_dose_response.py`; all per-seed/checkpoint measurements are in `metrics.json`.

## Success fraction by checkpoint

Exact XOR accuracy of 1.0 (each cell is a fraction of 200 seeds):

| init | alpha | 25 | 50 | 100 | 200 | 500 | 1,000 | 2,000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| He | 0 | .070 | .215 | .375 | .515 | .535 | .535 | .535 |
| He | 1e-4 | .080 | .285 | .515 | .755 | .800 | .825 | .840 |
| He | 1e-3 | .080 | .285 | .515 | .755 | .825 | .850 | .860 |
| He | 1e-2 | .090 | .280 | .530 | .755 | .855 | .895 | .920 |
| He | 1e-1 | .075 | .245 | .550 | .835 | .910 | .920 | .925 |
| Xavier | 0 | .145 | .300 | .435 | .520 | .530 | .530 | .530 |
| Xavier | 1e-4 | .165 | .405 | .660 | .775 | .800 | .820 | .855 |
| Xavier | 1e-3 | .160 | .410 | .680 | .780 | .825 | .875 | .885 |
| Xavier | 1e-2 | .160 | .405 | .660 | .790 | .885 | .930 | .935 |
| Xavier | 1e-1 | .145 | .365 | .650 | .850 | .920 | .925 | .925 |

## Final summaries

| init / alpha | mean final loss | median final loss | mean final accuracy | median first-success step | never successful |
|---|---:|---:|---:|---:|---:|
| He / 0 | .21067 | .00248 | .8675 | 59.0 | 93 |
| He / 1e-4 | .08434 | .00121 | .9325 | 69.0 | 32 |
| He / 1e-3 | .07372 | .00119 | .9363 | 71.0 | 28 |
| He / 1e-2 | .04545 | .00114 | .9613 | 77.5 | 16 |
| He / 1e-1 | .04217 | .00078 | .9625 | 81.0 | 15 |
| Xavier / 0 | .21287 | .00191 | .8650 | 43.5 | 94 |
| Xavier / 1e-4 | .08611 | .00094 | .9413 | 54.0 | 29 |
| Xavier / 1e-3 | .06848 | .00093 | .9488 | 56.0 | 23 |
| Xavier / 1e-2 | .03661 | .00084 | .9663 | 60.0 | 13 |
| Xavier / 1e-1 | .04200 | .00068 | .9625 | 63.0 | 15 |

## Paired comparisons at 2,000 updates

Counts are slope-only / ReLU-only / both / neither; p is the exact two-sided McNemar/binomial p-value.

| init, alpha vs 0 | counts | Δ success | p |
|---|---|---:|---:|
| He, 1e-4 | 61 / 0 / 107 / 32 | +.305 | 8.67e-19 |
| He, 1e-3 | 65 / 0 / 107 / 28 | +.325 | 5.42e-20 |
| He, 1e-2 | 77 / 0 / 107 / 16 | +.385 | 1.32e-23 |
| He, 1e-1 | 80 / 2 / 105 / 13 | +.390 | 1.41e-21 |
| Xavier, 1e-4 | 65 / 0 / 106 / 29 | +.325 | 5.42e-20 |
| Xavier, 1e-3 | 71 / 0 / 106 / 23 | +.355 | 8.47e-22 |
| Xavier, 1e-2 | 81 / 0 / 106 / 13 | +.405 | 8.27e-25 |
| Xavier, 1e-1 | 81 / 2 / 104 / 13 | +.395 | 7.21e-22 |

## Mechanism summaries

Rescue among prior ReLU failures (93 He, 94 Xavier):

| alpha | He rescued | Xavier rescued |
|---:|---:|---:|
| 1e-4 | 65.6% (61/93) | 69.1% (65/94) |
| 1e-3 | 69.9% (65/93) | 75.5% (71/94) |
| 1e-2 | 82.8% (77/93) | 86.2% (81/94) |
| 1e-1 | 86.0% (80/93) | 86.2% (81/94) |

Among control seeds with at least two dead ReLU units at step 25, rescue fractions were He: 84.6%, 85.9%, 93.6%, 94.9% and Xavier: 81.1%, 86.3%, 93.7%, 93.7% for alpha 1e-4, 1e-3, 1e-2, 1e-1 respectively.

At step 25, mean all-negative-unit counts were approximately 0.91/1.015 (He/Xavier) at 1e-4 and 0.90/1.01 at 1e-2; early signature diversity and first-layer gradient/update norms were comparable across nonzero slopes. The main difference is continued optimization over time, not a qualitatively different initial representation.

## Decision

**Near-binary dead-gradient effect, with a measurable dose component.** A tiny slope of 1e-4 already achieves 79.2% (He) and 80.2% (Xavier) of the full 1e-2 final rescue gap relative to ReLU; 1e-3 reaches 84.4% and 87.7%. This establishes that avoiding exactly zero negative-side gradients is the dominant causal factor. The monotonic gains through 1e-2 and the small 1e-1 increment show a secondary dose/conditioning effect, with no materially different He/Xavier pattern and no non-monotone failure of the conclusion.
