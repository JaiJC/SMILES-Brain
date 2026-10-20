# Validation guide

How to turn the harness from DEMO into a real, defensible evaluation. This is the
critical-path work — do it early.

## What "validated" means here

For a drug, we compare its **predicted** whole-brain map (from SMILES → affinity → atlas
projection) against an **observed** drug-response map (from imaging). Agreement is measured by
a spatial correlation, p-valued with a **spin test** that preserves spatial autocorrelation.
We also check whether the correlation's **sign** matches the pharmacology (`expected_sign`).

Two levels of claim, in order of difficulty:

1. **Measured-affinity reproduction (sanity).** Use lab-measured affinities (PDSP) for a known
   drug, project, correlate with its observed map. This should work — it's the published
   method. If it doesn't, something is wrong with alignment or the atlas before you blame the
   affinity model.
2. **Zero-shot prediction (the claim).** Use the Stage-1 model's affinity *predicted from
   SMILES*, holding out that drug's chemotype from training. Correlate with the observed map.

## Steps

### 1. Observed maps (the hard part — start here)
For each drug in `configs/validation_maps.yaml`, obtain a parcellated response map in the
**same parcellation as the atlas** (Schaefer-100, 7 networks) and save it as a one-value-per-
region CSV at the entry's `file` path (e.g. `data/validation/ketamine.csv`).

- Check OpenNeuro and the papers in each `source` for available group-level maps.
- If you have volumetric maps, parcellate with the Schaefer-100 atlas so region order matches
  the receptor atlas:
  `python scripts/ingest_map.py --drug <name> --nifti map.nii.gz --source "..." --n-subjects N --contrast "..."`
  (writes the catalog entry's CSV plus a `.provenance.json` sidecar; add `--zero-as-missing` if
  the deposited map uses 0, not NaN, outside its analysis mask).
- Record provenance and n per map. A handful of solid maps beats many shaky ones.

**If open maps turn out to be scarce, this reshapes the project claim — find out in week 1.**

### 2. Spin nulls (do once)
Needs `netneurotools>=0.2.5,<0.3` (0.3 removed the spin API). Parcel centroids come from the
Schaefer-2018 fsaverage5 annotation:

```bash
python scripts/fetch_parcel_coords.py            # -> data/atlas/schaefer100_coords.csv
python scripts/make_spins.py --coords data/atlas/schaefer100_coords.csv \
    --n-perm 1000 --out data/atlas/spins_scale100.npy
```

The spin method is netneurotools' `original` (Alexander-Bloch) rotation, which assigns each
rotated parcel to its nearest unrotated one, so a spun map is not a strict permutation (parcels
can repeat). That is the standard behaviour, not a bug.

Without this file the harness uses a naive permutation null and labels every p-value as
`spin=False` — never report those as spin tests.

### 3. Run
```bash
python scripts/evaluate.py \
    --artifact artifacts/affinity_baseline.joblib \
    --atlas data/atlas/density_scale100.csv \
    --spins data/atlas/spins_scale100.npy
```

Output columns: `r`, `p`, `spin`, `n_perm`, `n_receptors_used`, `sign_ok`.

For the two-tier report (measured-affinity reproduction vs zero-shot with the drug's chemotype
held out of training, asserted in code):

```bash
python scripts/evaluate.py --report        # -> artifacts/validation_report.md + .png
```

## Reporting honestly
- State n, the parcellation, and the null for every number.
- Report zero-shot (held-out chemotype) results separately from measured-affinity sanity.
- A modest, properly spin-tested correlation with correct signs across several drug classes
  is a stronger result than a big correlation with a naive null — and reviewers know it.

## Provenance log (observed maps)

All five catalog maps were obtained and parcellated to Schaefer-2018 100 parcels / 7 networks
(label ids 1..100 = receptor-atlas rows 0..99) with `scripts/ingest_map.py`. Files live in
`data/validation/` (gitignored); each has a `<file>.provenance.json` with source, n, contrast and
SHA-256 of input and output. None of the 100 regions was missing for any drug.

| drug | measure | n | source | file used | notes |
|---|---|---:|---|---|---|
| methylphenidate | CBF, drug - placebo | 18 | Dukart et al. 2018, *Sci Rep*, doi:10.1038/s41598-018-22444-0, supplementary zip MOESM2 | `Zscore_treatmentMap_Methylphenidate.nii` | 30 mg; 3 mm MNI |
| ketamine | CBF, drug - placebo | 26 | same | `Zscore_treatmentMap_Ketamine.nii` | 0.25 mg/kg bolus + 0.25 mg/kg/h; 3 mm MNI |
| risperidone | CBF, drug - placebo | 21 | same | `Zscore_treatmentMap_Risperidone2mg.nii` | **2 mg** chosen a priori over 0.5 mg (both deposited); 3 mm MNI |
| haloperidol | CBF, drug - placebo | 21 | same | `Zscore_treatmentMap_Haloperidol.nii` | 3 mg; 3 mm MNI |
| LSD | CBF, LSD - placebo (T map) | 15 | Carhart-Harris et al. 2016, *PNAS*, doi:10.1073/pnas.1518377113; NeuroVault collection 1083, image 14479 "CBF Difference" | `CBF_Difference.nii.gz` | 75 ug i.v.; 2 mm MNI; unthresholded |

How the maps were checked: no NaNs; Dukart maps are 0 outside their analysis mask, so exact zeros
were treated as missing (otherwise edge regions shrink toward 0) and the lowest per-region voxel
coverage was 69% (median 94%); LSD sign sanity-checked (visual cortex CBF rises under LSD, as in the
paper); the risperidone and haloperidol maps (both D2 antagonists) correlate at r = +0.53 across regions.

### Finding: maps were not scarce, but they are narrow

Open group-level maps exist for all five catalog drugs, so the project claim does not collapse.
But the evidence base is thin in ways to state whenever results are quoted:

- **Four of five maps come from one paper** (one lab, one ASL protocol, one preprocessing
  pipeline); LSD comes from a second study with a different statistic (T map vs effect size).
- **Everything is CBF**, an indirect readout, and n = 5 drugs. The original catalog listed LSD as
  BOLD from Luppi 2023; no deposited map was found for that, so LSD uses the CBF map above and the
  catalog `measure`/`source` were updated.
- **Ambiguity in the Dukart supplement:** it ships both `Zscore_treatmentMap_*` and
  `sZscore_treatmentMap_*` files and the paper text does not say what the `s` means (probably
  smoothed). The unprefixed files were used; regional means make the choice unlikely to matter much
  but it has not been tested.
- The Dukart files are named z-score maps while the paper describes group effect sizes; either
  way they are signed (increase vs decrease under drug).
- Luppi et al. 2023 (10 drugs, BOLD-based) did not deposit parcellated maps that could be found;
  asking the authors is the route to more drugs.

## Atlas bug found by the Tier A audit (2026-10-20)

Tier A is the published method, so when it barely worked the first suspect was our own
pipeline, not the biology. It was the atlas.

`fetch_atlas.py` used to average **every** upstream file whose leading token matched a
receptor key, with equal weight. Hansen et al.'s own `code/make_receptor_matrix.py` instead
picks **one tracer per receptor** and, where a receptor has several maps of the same tracer,
averages them **weighted by number of healthy controls**. Our version silently mixed back in
the tracers the paper discards -- D2 raclopride (n=7), GABA-A flumazenil (n=6), 5-HT2A MDL
(n=3), DAT FEPE2I (n=6), and others -- giving each the same vote as the map the paper kept.

Effect: **10 of the 19 receptor columns disagreed with the published atlas**, worst
DAT r=0.876, NAT r=0.875, 5HT2a r=0.883, CB1 r=0.895, D2 r=0.907, GABAa r=0.925.

Fixed by transcribing the paper's tracer selection and weights into `TRACER_SELECTION`.
`python -m smilesbrain.data.fetch_atlas --verify` now rebuilds the atlas and checks it against
the paper's own `results/receptor_data_scale100.csv`: **all 19 columns at r = 1.0000**. The
check is wired into the CLI so this cannot silently regress.

What it changed in the results: it **removed a spurious significant finding**. LSD's Tier A
correlation was r = -0.411, spin p = 0.031 -- significant, and with the *wrong* sign for a
5-HT2A agonist. On the corrected atlas it is r = +0.013, p = 0.966. The headline count of
significant results went down, which is the right direction for a bug fix.

### Ingestion pipeline verified independently

Separately, the observed-map ingestion was checked end to end by parcellating three of
Hansen's own volumetric PET images with `scripts/ingest_map.py`'s code path and correlating
against their shipped parcellated vectors: 5HT2a r=0.998, MU r=0.988, D1 r=0.980 (rolling the
region order by 50 as a control drops these to 0.40-0.79, so the test can fail). This also
surfaced two real fixes: 4D NIfTIs with a single trailing volume are now accepted, and the
`--zero-as-missing` flag is confirmed necessary (5HT2a scored r=0.72 without it, 0.998 with).
The LSD drug map was checked and is unaffected by that choice (r=0.9999 either way).

## What the diagnostics establish (2026-10-20)

Three diagnostics in `smilesbrain.validation.diagnostics` localise the failure. Run them with
`scripts/evaluate.py --report`; they are in the report.

### 1. The observed maps mostly do not resemble the drug's own target

Correlating each observed map with each single receptor density map, before any model:

| drug | best pre-registered target | r | rank among 19 | strongest receptor overall |
|---|---|---:|---:|---|
| methylphenidate | NAT | -0.257 | 5 | A4B2 (-0.49) |
| ketamine | NMDA | +0.213 | 6 | GABAa (+0.43) |
| risperidone | 5HT2a | -0.281 | 7 | VAChT (+0.47) |
| haloperidol | D1 | +0.572 | 2 | H3 (+0.57) |
| LSD | 5HT1a | -0.485 | 3 | MU (-0.60) |
| escitalopram | 5HTT | +0.089 | 14 | MU (-0.57) |
| olanzapine | D2 | +0.565 | 2 | 5HT1a (+0.63) |
| midazolam | GABAa | -0.225 | 16 | 5HT4 (-0.63) |

Midazolam's GABA-A ranks 16th of 19; LSD's 5-HT2A ranks 18th. The two drugs whose target does
rank near the top (haloperidol, olanzapine) correlate **positively** with it, though both are
antagonists, which is the opposite of the pre-registered expectation.

### 2. But the architecture is not the problem -- the ceiling is high

Fitting the 19 density maps to each observed map (ridge, scored on held-out regions) gives a
**median cross-validated r of 0.43 (OLS) to 0.67 (ridge), spin-significant for 8/8 drugs**,
ranging to r = 0.81 for LSD. So a weighted sum of receptor densities *can* reproduce these
maps well. The atlas is rich enough and the maps are not noise.

### 3. The weight vector is what fails -- and affinity is no better than chance

The decisive test. Keep the drug's measured affinity values and the real atlas; shuffle only
**which receptor holds which value**. If the true assignment carries information, it should
beat its own shuffles.

| drug | receptors | true r | mean \|r\| of shuffled assignments | p |
|---|---:|---:|---:|---:|
| methylphenidate | 12 | +0.019 | 0.200 | 0.946 |
| ketamine | 14 | +0.064 | 0.149 | 0.714 |
| risperidone | 15 | -0.346 | 0.248 | 0.274 |
| haloperidol | 16 | -0.051 | 0.216 | 0.868 |
| LSD | 4 | +0.013 | 0.332 | 1.000 |
| olanzapine | 8 | +0.101 | 0.258 | 0.806 |

**0 of 6 drugs beat the receptor-permutation null.** For most, the true assignment does *worse*
than the average shuffle. This holds for Tier A, i.e. with **lab-measured** affinity, so it is
not a failure of the affinity model.

### Why we do not just read the fitted weights instead

The 19 density maps are collinear: mean |r| = 0.34, 47 of 171 pairs above 0.5, first principal
component 37.5% of variance. Bootstrapping the fitted weights gives between-resample
correlations as low as 0.0, so individual coefficients are not interpretable even where the
fitted predictions are stable. The permutation null in (3) never reads a coefficient, which is
exactly why it is the test to trust here.

### The conclusion

> Receptor density maps can explain drug-induced CBF maps well (ceiling r up to 0.81), but
> **binding affinity is not the weight that does it**. Knowing which receptor a drug binds, and
> how tightly, predicts these maps no better than randomly reassigning those same affinities
> among receptors.

This is a negative result for the project's central premise, and it is not fixable by a better
affinity model: it already fails with measured affinity. What it does not rule out is that some
*other* transform of the affinity profile -- occupancy at clinical dose, efficacy, direction,
weighted by PK -- is the right weight. The high ceiling says such a weight vector exists; the
permutation null says plain affinity is not it. That is the question worth attacking next, and
it is the same inverse problem as the roadmap's inverse-design task.

## Results: first real run (2026-10-20, `artifacts/validation_report.md`)

Schaefer-100, 1000 spin rotations, z-score weights, magnitude-only projection, ridge baseline
(alpha 100 from the validated baseline, not re-tuned per drug), n = 5 drugs.

| | significant (spin p < 0.05) | after BH within tier | expected sign correct |
|---|---|---|---|
| Tier A, measured affinity | 2/5 (risperidone r = -0.43, LSD r = -0.41) | 0/5 | 4/5 (LSD wrong) |
| Tier B, zero-shot | 0/5 | 0/5 | 2/5 |

**Plain reading: this is a null result.** The zero-shot claim is not supported by these five
maps, and the sanity tier is only weakly reproduced (the measured-affinity method that the
project builds on is itself not clearly recovered here). Do not quote a correlation from this
report without these numbers next to it.

Things that make the null less damning than it looks, none of which is tested yet:

- Tier A for LSD rests on only 4 measured receptors, and its significant correlation has the
  *wrong* sign for a 5-HT2A agonist.
- The projection is magnitude-only; for antagonists (risperidone, haloperidol, methylphenidate
  as a reuptake inhibitor) the published approach relies on sign. Supplying directions is a fair
  next experiment, but they must come from pharmacology written down *before* looking at these
  results; picking directions to improve r would be fitting to five data points.
- Only the cortex (100 parcels) is tested; the largest CBF effects of several of these drugs are
  subcortical. `cammoun033` has an atlas now but no matching observed-map ingestion or spin null.
- Haloperidol's nearest remaining training analogue has Tanimoto 0.93 even after its scaffold was
  removed, so its "zero-shot" row is the least held-out of the five.
