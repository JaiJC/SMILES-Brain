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
- If you have volumetric maps, parcellate with the Schaefer-100 atlas (nilearn
  `NiftiLabelsMasker`) so region order matches the receptor atlas.
- Record provenance and n per map. A handful of solid maps beats many shaky ones.

**If open maps turn out to be scarce, this reshapes the project claim — find out in week 1.**

### 2. Spin nulls (do once)
Get Schaefer-100 parcel spherical centroids + hemisphere, save as CSV `x,y,z,hemi`, then:

```bash
python scripts/make_spins.py --coords data/atlas/schaefer100_coords.csv \
    --n-perm 1000 --out data/atlas/spins_scale100.npy
```

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

## Reporting honestly
- State n, the parcellation, and the null for every number.
- Report zero-shot (held-out chemotype) results separately from measured-affinity sanity.
- A modest, properly spin-tested correlation with correct signs across several drug classes
  is a stronger result than a big correlation with a naive null — and reviewers know it.
