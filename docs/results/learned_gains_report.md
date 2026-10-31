# Learned receptor gains: pre-registered leave-one-drug-out test

Pre-registration: `docs/PREREGISTRATION.md` (SHA-256 `c49d8461cdd7fae305a9b884490093d57ed5e69e5e1ef877485b8567382604cb`). Parcellation `schaefer100`, 100 regions, 5000 receptor shuffles, 1000 spin rotations.

## Primary endpoint (concentration 100 nM, leave-one-drug-out)

**Tier A (measured affinity, the gate): CRITERION NOT MET** - 0/6 drugs beat the receptor-shuffle null (needed 4); median r +0.086; median gain over the mean-map baseline -0.010.

| drug | receptors | train drugs | r model | r mean-map | delta | shuffled mean r | p (receptor shuffle) | spin p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LSD | 4 | 5 | -0.359 | -0.274 | -0.085 | -0.349 | 0.702 | 0.922 |
| haloperidol | 16 | 5 | +0.326 | +0.262 | +0.065 | +0.232 | 0.413 | 0.088 |
| ketamine | 14 | 5 | -0.301 | +0.194 | -0.496 | -0.247 | 0.641 | 0.982 |
| methylphenidate | 12 | 5 | +0.108 | +0.004 | +0.105 | -0.067 | 0.104 | 0.384 |
| olanzapine | 8 | 5 | +0.338 | +0.053 | +0.285 | +0.055 | 0.173 | 0.086 |
| risperidone | 15 | 5 | +0.064 | +0.159 | -0.095 | -0.014 | 0.419 | 0.321 |

**Tier B (zero-shot): CRITERION NOT MET** - *not to be interpreted: Tier A did not meet the criterion.*

| drug | receptors | train drugs | r model | r mean-map | delta | shuffled mean r | p (receptor shuffle) | spin p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LSD | 19 | 5 | -0.386 | -0.274 | -0.113 | -0.245 | 0.971 | 0.936 |
| escitalopram | 19 | 6 | +0.429 | +0.144 | +0.286 | +0.129 | 0.201 | 0.015 |
| haloperidol | 19 | 5 | +0.454 | +0.262 | +0.192 | +0.277 | 0.258 | 0.023 |
| ketamine | 19 | 5 | -0.012 | +0.194 | -0.206 | +0.058 | 0.779 | 0.529 |
| methylphenidate | 19 | 5 | -0.366 | +0.004 | -0.370 | -0.254 | 0.709 | 0.949 |
| midazolam | 19 | 6 | +0.447 | -0.183 | +0.630 | +0.129 | 0.220 | 0.017 |
| olanzapine | 19 | 5 | +0.374 | +0.053 | +0.321 | +0.276 | 0.506 | 0.061 |
| risperidone | 19 | 5 | -0.054 | +0.159 | -0.213 | -0.021 | 0.630 | 0.553 |

- Sensitivity, tier A, 100 nM, leave-class-out: CRITERION NOT MET (1/6 beat null, median r +0.046).
- Sensitivity, tier B, 100 nM, leave-class-out: CRITERION NOT MET (0/8 beat null, median r +0.025).
- Sensitivity, tier A, 10 nM, leave-one-drug-out: CRITERION NOT MET (0/6 beat null, median r +0.207).
- Sensitivity, tier B, 10 nM, leave-one-drug-out: CRITERION NOT MET (1/8 beat null, median r +0.160).
- Sensitivity, tier A, 10 nM, leave-class-out: CRITERION NOT MET (0/6 beat null, median r +0.231).
- Sensitivity, tier B, 10 nM, leave-class-out: CRITERION NOT MET (1/8 beat null, median r +0.148).
- Sensitivity, tier A, 1000 nM, leave-one-drug-out: CRITERION NOT MET (0/6 beat null, median r +0.085).
- Sensitivity, tier B, 1000 nM, leave-one-drug-out: CRITERION NOT MET (0/8 beat null, median r +0.137).
- Sensitivity, tier A, 1000 nM, leave-class-out: CRITERION NOT MET (1/6 beat null, median r -0.158).
- Sensitivity, tier B, 1000 nM, leave-class-out: CRITERION NOT MET (0/8 beat null, median r -0.174).

Skipped (not silently dropped): A/100/leave-one-drug-out: {'escitalopram': 'only 2 receptors in the test profile (need 3)', 'midazolam': 'only 1 receptors in the test profile (need 3)'}; A/100/leave-class-out: {'escitalopram': 'only 2 receptors in the test profile (need 3)', 'midazolam': 'only 1 receptors in the test profile (need 3)'}; A/10/leave-one-drug-out: {'escitalopram': 'only 2 receptors in the test profile (need 3)', 'midazolam': 'only 1 receptors in the test profile (need 3)'}; A/10/leave-class-out: {'escitalopram': 'only 2 receptors in the test profile (need 3)', 'midazolam': 'only 1 receptors in the test profile (need 3)'}; A/1000/leave-one-drug-out: {'escitalopram': 'only 2 receptors in the test profile (need 3)', 'midazolam': 'only 1 receptors in the test profile (need 3)'}; A/1000/leave-class-out: {'escitalopram': 'only 2 receptors in the test profile (need 3)', 'midazolam': 'only 1 receptors in the test profile (need 3)'}

## Tier B held-out diagnostics

- LSD: 53310 training molecules, removed 3 same molecule and 29 same scaffold; nearest remaining Tanimoto 0.71
- escitalopram: 53324 training molecules, removed 6 same molecule and 12 same scaffold; nearest remaining Tanimoto 0.85
- haloperidol: 53293 training molecules, removed 2 same molecule and 47 same scaffold; nearest remaining Tanimoto 0.93
- ketamine: 53338 training molecules, removed 2 same molecule and 2 same scaffold; nearest remaining Tanimoto 0.33
- methylphenidate: 53285 training molecules, removed 5 same molecule and 52 same scaffold; nearest remaining Tanimoto 0.57
- midazolam: 53335 training molecules, removed 1 same molecule and 6 same scaffold; nearest remaining Tanimoto 0.53
- olanzapine: 53339 training molecules, removed 1 same molecule and 2 same scaffold; nearest remaining Tanimoto 0.73
- risperidone: 53334 training molecules, removed 2 same molecule and 6 same scaffold; nearest remaining Tanimoto 0.79
