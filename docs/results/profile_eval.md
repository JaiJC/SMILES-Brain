# Does the affinity model predict a molecule's receptor profile?

Scaffold split from `configs/affinity_baseline.yaml` (seed 0): 42674 train / 10668 test molecules, **no scaffold shared** (asserted); artifact verified to have been trained on exactly this training fold. 1206 test molecules have >=3 measured receptors. Mean per-receptor held-out R^2 is in `docs/results/affinity_baseline_report.csv`.

Within-molecule Spearman rho between predicted and measured profile across the receptors measured for that molecule; 95% CIs by resampling whole scaffolds (2000 draws). The baseline predicts every molecule with the training mean of each receptor, so it knows nothing about the molecule. A second baseline copies the profile of the nearest training analogues (kNN).

| subset | molecules | scaffolds | rho model | rho receptor-mean | model - receptor-mean | rho kNN (k=25) | model - kNN | top-1 model | top-1 receptor-mean | top-1 kNN | top-1 chance |
|---|---:|---:|---|---|---|---|---|---:|---:|---:|---:|
| all held-out molecules (>=3 receptors measured) | 1206 | 78 | +0.518 [+0.448, +0.592] | +0.153 [-0.006, +0.334] | +0.364 [+0.231, +0.494] | +0.490 [+0.405, +0.581] | +0.027 [-0.019, +0.075] | 0.609 | 0.417 | 0.594 | 0.310 |
| 3 receptors measured | 961 | 65 | +0.555 [+0.478, +0.630] | +0.186 [+0.002, +0.391] | +0.368 [+0.221, +0.510] | +0.522 [+0.438, +0.605] | +0.032 [-0.018, +0.082] | 0.637 | 0.469 | 0.625 | 0.337 |
| 4-5 receptors measured | 166 | 45 | +0.401 [+0.297, +0.481] | +0.006 [-0.178, +0.219] | +0.395 [+0.147, +0.633] | +0.401 [+0.247, +0.566] | -0.000 [-0.125, +0.123] | 0.530 | 0.265 | 0.566 | 0.245 |
| 6+ receptors measured | 79 | 29 | +0.309 [+0.152, +0.482] | +0.060 [-0.076, +0.208] | +0.250 [+0.138, +0.402] | +0.286 [+0.124, +0.457] | +0.023 [-0.076, +0.110] | 0.430 | 0.101 | 0.266 | 0.118 |
| novelty: nearest training Tanimoto < 0.3 | 22 | 6 | +0.078 [-0.348, +0.500] | +0.382 [+0.133, +0.586] | -0.304 [-0.795, +0.104] | +0.193 [-0.252, +0.602] | -0.115 [-0.287, +0.036] | 0.409 | 0.545 | 0.500 | 0.295 |
| novelty: 0.3 - 0.5 | 266 | 36 | +0.418 [+0.278, +0.566] | +0.262 [+0.033, +0.501] | +0.156 [-0.015, +0.329] | +0.353 [+0.178, +0.534] | +0.065 [-0.012, +0.147] | 0.560 | 0.545 | 0.526 | 0.300 |
| novelty: >= 0.5 | 918 | 73 | +0.557 [+0.483, +0.624] | +0.116 [-0.065, +0.313] | +0.441 [+0.278, +0.599] | +0.537 [+0.454, +0.618] | +0.020 [-0.035, +0.067] | 0.627 | 0.377 | 0.615 | 0.313 |

Novelty rows use the nearest-training Tanimoto (Morgan bits) of every evaluable test molecule. The kNN baseline averages the 25 most similar training molecules that have each receptor measured, weighted by similarity.

## Catalog drugs: zero-shot profile (chemotype held out) vs measured profile

Only drugs with >=3 measured receptors. rho is across the measured receptors.

| drug | measured receptors | rho zero-shot | rho baseline | measured top | top-1 | top-3 | nearest train Tanimoto |
|---|---:|---:|---:|---|---|---|---:|
| LSD | 4 | +0.80 | +1.00 | 5HT1a | yes | yes | 0.71 |
| haloperidol | 16 | +0.56 | -0.48 | D2 | yes | yes | 0.93 |
| ketamine | 14 | -0.03 | -0.38 | NMDA | no | no | 0.33 |
| methylphenidate | 12 | +0.11 | -0.43 | DAT | no | no | 0.57 |
| olanzapine | 8 | +0.93 | -0.07 | 5HT2a | yes | yes | 0.73 |
| risperidone | 15 | +0.88 | +0.20 | 5HT2a | yes | yes | 0.79 |

6 drugs: zero-shot rho beats the baseline for 5; measured top receptor is the predicted top-1 for 4 and in the predicted top-3 for 4.
