# SMILES-Brain validation report

Generated 2026-10-31 10:16 UTC · commit `59a26a4` · parcellation `schaefer100` (100 regions) · weights `zscore` · primary projection **magnitude-only** (no agonist/antagonist direction) · seed 0

Each row correlates a drug's **predicted** map (affinity vector x receptor-density atlas) with its **observed** drug-minus-placebo CBF map across regions. p-values come from the null named in each row. Tier A and Tier B differ only in where the affinity vector comes from.

## Tier A - measured-affinity reproduction (sanity check)

Affinity vector = median of lab-measured pChEMBL (ChEMBL + PDSP) for the drug. This is the published method and should work before the zero-shot claim is trusted. Unmeasured receptors are dropped, not set to zero.

| drug | measure | n subj | receptors used | r | spin p | BH p | null | expected sign | sign agrees |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| methylphenidate | CBF | 18 | 12 | +0.019 | 0.884 | 0.966 | spin (1000) | - | no |
| ketamine | CBF | 26 | 14 | +0.064 | 0.722 | 0.966 | spin (1000) | + | yes |
| risperidone | CBF | 21 | 15 | -0.346 | 0.035 | 0.210 | spin (1000) | - | yes |
| haloperidol | CBF | 21 | 16 | -0.051 | 0.909 | 0.966 | spin (1000) | - | yes |
| LSD | CBF | 15 | 4 | +0.013 | 0.966 | 0.966 | spin (1000) | + | yes |
| olanzapine | CBF | 21 | 8 | +0.101 | 0.547 | 0.966 | spin (1000) | - | no |

1/6 with p < 0.05 (uncorrected); 0/6 after Benjamini-Hochberg across this tier; sign agreement 4/6.

## Tier B - zero-shot prediction (the claim)

Affinity vector = predicted from SMILES by a ridge model refit with the drug's molecule (any salt / stereoisomer) **and its whole Bemis-Murcko scaffold removed from training**. Checked by `assert_held_out` on the exact training SMILES; a leak would abort the run.

| drug | measure | n subj | receptors used | r | spin p | BH p | null | expected sign | sign agrees |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| methylphenidate | CBF | 18 | 19 | +0.132 | 0.276 | 0.649 | spin (1000) | - | no |
| ketamine | CBF | 26 | 19 | +0.077 | 0.628 | 0.718 | spin (1000) | + | yes |
| risperidone | CBF | 21 | 19 | -0.489 | 0.004 | 0.032 | spin (1000) | - | yes |
| haloperidol | CBF | 21 | 19 | +0.202 | 0.489 | 0.655 | spin (1000) | - | no |
| LSD | CBF | 15 | 19 | -0.143 | 0.492 | 0.655 | spin (1000) | + | no |
| escitalopram | CBF | 18 | 19 | -0.161 | 0.325 | 0.649 | spin (1000) | - | yes |
| olanzapine | CBF | 21 | 19 | +0.275 | 0.096 | 0.384 | spin (1000) | - | no |
| midazolam | CBF | 26 | 19 | +0.013 | 0.943 | 0.943 | spin (1000) | - | no |

1/8 with p < 0.05 (uncorrected); 1/8 after Benjamini-Hochberg across this tier; sign agreement 3/8.

## Pre-specified secondary analysis - direction-aware projection

The same two affinity sources, but projected through **only** the receptors whose agonist/antagonist direction is recorded in `configs/validation_maps.yaml`, with that sign applied and raw potency as the weight. Those directions were written from pharmacology references and committed before this analysis was run (see that file's git history). The magnitude-only tiers above were already known at that point, but no direction was chosen by looking at a correlation.

Raw potency is used instead of the z-scored profile because z-scoring across a one- to four-receptor subset is degenerate: a single receptor z-scores to exactly 0, which would give an all-zero map. That follows from the restriction, not from tuning.

### Tier A-dir - measured affinity, direction-aware

| drug | measure | n subj | receptors (signed) | r | spin p | BH p | null | expected sign | sign agrees |
|---|---|---:|---|---:|---:|---:|---|---|---|
| methylphenidate | CBF | 18 | DAT-, NAT- | +0.085 | 0.578 | 0.740 | spin (1000) | - | no |
| ketamine | CBF | 26 | NMDA- | -0.213 | 0.198 | 0.393 | spin (1000) | + | no |
| risperidone | CBF | 21 | D2-, 5HT2a- | +0.060 | 0.771 | 0.771 | spin (1000) | - | no |
| haloperidol | CBF | 21 | D2-, D1- | -0.440 | 0.109 | 0.393 | spin (1000) | - | yes |
| LSD | CBF | 15 | 5HT2a+, 5HT1a+, 5HT1b+, D2+ | -0.329 | 0.246 | 0.393 | spin (1000) | + | no |
| escitalopram | CBF | 18 | 5HTT- | -0.089 | 0.647 | 0.740 | spin (1000) | - | yes |
| olanzapine | CBF | 21 | D2-, 5HT2a-, M1- | -0.429 | 0.060 | 0.393 | spin (1000) | - | yes |
| midazolam | CBF | 26 | GABAa+ | -0.225 | 0.232 | 0.393 | spin (1000) | - | yes |

0/8 with p < 0.05 (uncorrected); 0/8 after Benjamini-Hochberg across this tier; sign agreement 4/8.

### Tier B-dir - zero-shot, direction-aware

| drug | measure | n subj | receptors (signed) | r | spin p | BH p | null | expected sign | sign agrees |
|---|---|---:|---|---:|---:|---:|---|---|---|
| methylphenidate | CBF | 18 | DAT-, NAT- | +0.094 | 0.531 | 0.709 | spin (1000) | - | no |
| ketamine | CBF | 26 | NMDA- | -0.213 | 0.198 | 0.464 | spin (1000) | + | no |
| risperidone | CBF | 21 | D2-, 5HT2a- | +0.055 | 0.784 | 0.784 | spin (1000) | - | no |
| haloperidol | CBF | 21 | D2-, D1- | -0.450 | 0.099 | 0.396 | spin (1000) | - | yes |
| LSD | CBF | 15 | 5HT2a+, 5HT1a+, 5HT1b+, D2+ | -0.211 | 0.487 | 0.709 | spin (1000) | + | no |
| escitalopram | CBF | 18 | 5HTT- | -0.089 | 0.647 | 0.740 | spin (1000) | - | yes |
| olanzapine | CBF | 21 | D2-, 5HT2a-, M1- | -0.424 | 0.063 | 0.396 | spin (1000) | - | yes |
| midazolam | CBF | 26 | GABAa+ | -0.225 | 0.232 | 0.464 | spin (1000) | - | yes |

0/8 with p < 0.05 (uncorrected); 0/8 after Benjamini-Hochberg across this tier; sign agreement 4/8.

### Zero-shot guarantee

| drug | training molecules | removed: same molecule | removed: same scaffold only | nearest remaining Tanimoto | murcko scaffold (parent) |
|---|---:|---:|---:|---:|---|
| methylphenidate | 53285 | 5 | 52 | 0.57 | `c1ccc(CC2CCCCN2)cc1` |
| ketamine | 53338 | 2 | 2 | 0.33 | `O=C1CCCCC1c1ccccc1` |
| risperidone | 53334 | 2 | 6 | 0.79 | `O=c1c(CCN2CCC(c3noc4ccccc34)CC2)cnc2n1CCCC2` |
| haloperidol | 53293 | 2 | 47 | 0.93 | `O=C(CCCN1CCC(c2ccccc2)CC1)c1ccccc1` |
| LSD | 53310 | 3 | 29 | 0.71 | `C1=C2c3cccc4[nH]cc(c34)CC2NCC1` |
| escitalopram | 53324 | 6 | 12 | 0.85 | `c1ccc(C2OCc3ccccc32)cc1` |
| olanzapine | 53339 | 1 | 2 | 0.73 | `c1ccc2c(c1)N=C(N1CCNCC1)c1ccsc1N2` |
| midazolam | 53335 | 1 | 6 | 0.53 | `c1ccc(C2=NCc3cncn3-c3ccccc32)cc1` |

The nearest-Tanimoto column is the honest caveat: removing a scaffold does not remove close analogues with a different scaffold, so "zero-shot" here means *held-out chemotype*, not *no similar training molecules*.

## Diagnostics - where the pipeline succeeds and where it fails

A spin test says whether a prediction matched; it does not say which step is at fault when it did not. These three localise it.

### 1. Does the observed map resemble the drug's own target receptor?

Correlation of each observed map with each single receptor density map, before any model. `rank` is where the drug's best pre-registered target sits among all 19 receptors, by |r|.

| drug | best target | r | rank | spin p | strongest receptor overall |
|---|---|---:|---:|---:|---|
| methylphenidate | NAT | -0.257 | 5/19 | 0.176 | A4B2 (-0.49) |
| ketamine | NMDA | +0.213 | 6/19 | 0.198 | GABAa (+0.43) |
| risperidone | 5HT2a | -0.281 | 7/19 | 0.126 | VAChT (+0.47) |
| haloperidol | D1 | +0.572 | 2/19 | 0.005 | H3 (+0.57) |
| LSD | 5HT1a | -0.485 | 3/19 | 0.031 | MU (-0.60) |
| escitalopram | 5HTT | +0.089 | 14/19 | 0.647 | MU (-0.57) |
| olanzapine | D2 | +0.565 | 2/19 | 0.012 | 5HT1a (+0.63) |
| midazolam | GABAa | -0.225 | 16/19 | 0.232 | 5HT4 (-0.63) |

### 2. Ceiling: the best any weighting of this atlas could do

Oracle, not a result. The weights are fit to the observed map itself (ridge, collinearity-aware) and scored on held-out regions. This is what a perfect affinity vector would be competing against. `weight stability` is the mean correlation between bootstrap weight vectors.

| drug | in-sample r (overfit) | cross-validated r | spin p | weight stability |
|---|---:|---:|---:|---:|
| methylphenidate | 0.655 | 0.456 | 0.106 | 0.14 |
| ketamine | 0.733 | 0.468 | 0.005 | 0.27 |
| risperidone | 0.775 | 0.685 | 0.001 | 0.26 |
| haloperidol | 0.859 | 0.742 | 0.001 | 0.33 |
| LSD | 0.891 | 0.810 | 0.002 | 0.58 |
| escitalopram | 0.722 | 0.589 | 0.005 | 0.19 |
| olanzapine | 0.758 | 0.674 | 0.002 | 0.03 |
| midazolam | 0.745 | 0.663 | 0.002 | 0.04 |

Median cross-validated ceiling r = **0.668**, spin-significant for 7/8 drugs.

### 3. Does it matter which receptor holds which affinity?

The decisive test of the claim. Keeps the drug's measured affinity *values* and the real atlas, and shuffles only which receptor each value belongs to. Immune to the collinearity that makes fitted weights uninterpretable, because it never reads a coefficient.

| drug | receptors | true r | mean \|r\| of shuffled assignments | p |
|---|---:|---:|---:|---:|
| methylphenidate | 12 | +0.019 | 0.200 | 0.946 |
| ketamine | 14 | +0.064 | 0.150 | 0.722 |
| risperidone | 15 | -0.346 | 0.245 | 0.279 |
| haloperidol | 16 | -0.051 | 0.219 | 0.870 |
| LSD | 4 | +0.013 | 0.336 | 1.000 |
| escitalopram | <3 measured | - | - | - |
| olanzapine | 8 | +0.101 | 0.260 | 0.806 |
| midazolam | <3 measured | - | - | - |

**0/6 drugs beat the receptor-permutation null at p < 0.05.**

Why the fitted weights cannot simply be read off instead: the 19 density maps are collinear (mean |r| = 0.34; 47/171 pairs above 0.5; first principal component 37% of variance). Many different weight vectors fit about equally well, which is why the ceiling in (2) is high while the bootstrap weight stability beside it is low.

## Limitations

- **n = 8 drugs.** Per-drug p-values are the evidence; pooled claims are not supported by this few. Benjamini-Hochberg is applied within each tier only.
- Observed maps are drug-induced **CBF** changes from two studies (Dukart 2018: 4 drugs; Carhart-Harris 2016: LSD, a T-map not an effect size), group-level, different n. CBF is an indirect readout of neural activity, not receptor occupancy.
- The projection is **magnitude-only**: it does not know agonist from antagonist, so a sign-correct result for an antagonist is partly luck. Directions can be supplied (`--direction`), not predicted.
- Receptor density says where a drug *can* act, not how much reaches the brain.
- A spin test controls for spatial autocorrelation of the cortical maps only; subcortex is not in this parcellation.
- Expected signs were fixed in `configs/validation_maps.yaml` before these results were computed.
