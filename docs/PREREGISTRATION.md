# Pre-registration v1: learned receptor gains (leave-one-drug-out)

Written 2026-10-31, **before** the model below has been run on any real drug map. Commit this
file before running `scripts/learned_gains.py`; the script records this file's SHA-256 in its
report and refuses to run if the file is missing. Any later change is a new version (v2, ...)
and results under it are labelled exploratory.

## Why this experiment

Established so far (`docs/VALIDATION.md`, `docs/REPORT.md`), on 8 open CBF drug maps:

- a weighted sum of the 19 receptor density maps can reproduce the maps (cross-validated
  r 0.46-0.81), so the atlas is expressive enough;
- but treating binding **affinity** as that weight does no better than randomly reassigning the
  same affinities among receptors (0 of 6 testable drugs beat that null), using lab-measured
  affinity, so a better structure-to-affinity model cannot fix it.

Two untested reasons affinity might fail as a weight: (1) binding saturates, so potency is not
occupancy; (2) receptors do not push CBF in the same direction or by the same amount, and that
receptor-specific effect is shared across drugs, so it can be learned from other drugs.

## Hypothesis

**H1.** For a held-out drug, `map(region) = sum_r gain_r * occupancy_r * density(region, r)`,
with gains fitted only on *other* drugs' observed maps, predicts that drug's observed map better
than the same model given its occupancy values on shuffled receptors, and better than the
average of the training maps.

**H0.** Which receptors a drug occupies carries no information about its map beyond what a
drug-independent pattern already gives.

## Fixed analysis (not to be tuned on the results)

- Occupancy `C / (C + Ki)` with `Ki = 10^(9 - pKi)` nM; unmeasured receptors contribute 0.
- **Primary concentration C = 100 nM**, one value for every drug. This is a convention, not a
  measured free concentration; per-drug pharmacokinetics are out of scope.
- Gains: ridge regression on stacked (drug, region) rows; each training map standardised;
  design columns centred per drug; ridge strength chosen by inner leave-one-drug-out among
  {0.01, 0.1, 1, 10, 100, 1000}, ties to the larger value.
- Held-out drug: correlation (Pearson r) between predicted and observed map over 100
  Schaefer-100 regions.
- Receptor-shuffle null: 5000 shuffles of the held-out drug's occupancy values among its own
  measured receptors, gains held fixed; **one-sided** `p = (1 + #{r_shuffled >= r}) / (1 + n)`.
- Mean-map baseline: average of the standardised training maps.
- A drug needs >= 3 measured receptors and >= 3 training drugs, else it is reported as skipped.
- Spin p-values are reported only when real spin permutations are supplied; they are secondary.

## Primary endpoint and decision rule

Tier A (measured affinity), leave-one-drug-out, 100 nM, the catalog's eight drugs that have an
observed map. With `n` evaluable drugs:

- **INCONCLUSIVE** if `n < 6`.
- **CRITERION MET** if at least `max(4, ceil(n/2))` drugs have receptor-shuffle `p < 0.05`
  **and** the median of `(r_model - r_mean_map)` is positive.
- **CRITERION NOT MET** otherwise.

Under H0 the chance of 4 or more of 8 drugs at p < 0.05 is about 0.0004 (binomial), so no
further multiplicity correction is applied to this single primary test.

**Tier B (zero-shot)** is interpreted only if Tier A is met, because a structure-predicted
profile cannot be expected to beat the measured profile it approximates. Hierarchical, so no
multiplicity cost.

## Sensitivity analyses (reported, not confirmatory, never used to redefine success)

- C = 10 nM and C = 1000 nM.
- Leave-class-out: haloperidol, risperidone and olanzapine never train each other. Related
  drugs have similar profiles and similar maps (risperidone vs haloperidol r = +0.53), so
  ordinary leave-one-out can flatter the model through class similarity.

## What the test can and cannot rule out (measured, synthetic)

`python scripts/lodo_power.py --seeds 40` plants a correctly specified gain model in synthetic
data (8 drugs, 8-16 receptors each) and adds noise to the maps:

| noise sd (signal sd = 1) | criterion met | median held-out r |
|---|---:|---:|
| none (maps unrelated to affinity) | 0% | -0.02 |
| 0.3 | 100% | +0.94 |
| 0.6 | 98% | +0.84 |
| 1.0 | 92% | +0.68 |
| 1.5 | 72% | +0.51 |
| 2.0 | 50% | +0.40 |
| 3.0 | 10% | +0.26 |

The verdict is threshold-sensitive near the boundary: one synthetic world with held-out r about
0.9 gave 3 of 8 drugs below p < 0.05 at 200 shuffles and 4 of 8 at 2000, because the shuffled
null is itself strong when the density maps are collinear (shuffled r about 0.7 there; the
synthetic atlas is about as collinear as the real one, mean |r| 0.36 vs 0.34). This is why the
primary run uses 5000 shuffles, and why a borderline count should be read as borderline.

Read this as: a NOT MET verdict rules out gain-model signals giving held-out r above roughly
0.5, but **not** weak ones (around r 0.3 or below). This is the best case: real maps have
unmodelled structure and the real occupancy is not exactly 100 nM for every drug, so real power
is lower.

## Interpretation, decided in advance

- **MET (Tier A):** affinity-derived occupancy plus shared receptor gains carries drug-specific
  information. Then read Tier B; if it is also met, the zero-shot claim has support and the
  D-MPNN work matters. Report the gains with the collinearity caveat (densities are collinear,
  individual gains are not interpretable one at a time).
- **NOT MET:** this transform does not rescue affinity as a weight on these data. Report as a
  second negative result. Do not tune concentration, ridge grid or groupings and re-run to
  reach MET.
- **INCONCLUSIVE:** state which maps were missing and stop.

## Known limitations fixed in advance

- Eight drugs, seven from one paper (one lab, one ASL protocol), all CBF; CBF is indirect.
- 19 shared gains from at most 7 training drugs; heavy regularisation is expected and a thin
  fit is itself informative.
- Cortex only (100 parcels); several of these drugs act subcortically.
- Direction (agonist or antagonist) enters only through the learned gain, not through the
  catalog's per-drug directions, which are a separate pre-registered analysis.
- A single global concentration ignores dose, bioavailability and brain penetration.

## What was already known when this was written

The magnitude-only results, the single-receptor correlation table and the receptor-shuffle
result for affinity-as-weight are in `docs/VALIDATION.md`. **No result of the gains model on any
real drug map had been seen.** Its behaviour on synthetic data (recovery of planted gains, a
false-positive rate of 0 of 40 studies on null data, the power table above) is covered by
`tests/test_learned_gains.py` and `scripts/lodo_power.py`.

## Deviations log

(none yet; append here with date and reason if anything changes after this is committed)
