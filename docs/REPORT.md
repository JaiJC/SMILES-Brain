# SMILES-Brain: write-up

Status: **negative result on the central premise; infrastructure complete.** Every number below
is traced to a file in this repository. Numbers that depend on artifacts not committed here (they
are gitignored) are marked and reproduced by the command shown.

## Why this matters (one paragraph)

Predicting where in the brain a molecule acts, from its structure alone, would let chemists
prioritise compounds before any assay or scan. Prior work does this with *lab-measured* binding
affinity, so it only covers drugs that have already been tested. SMILES-Brain tried to replace
the measured affinity with a structure-based prediction and to run the mapping backwards
(inverse design). The useful outcome so far is not the map; it is a tested, reproducible harness
that shows **plain binding affinity is not the right weight for receptor-density maps**, so that
the open question can be attacked directly.

## What was done

1. **Affinity models (Stage 1).** Multi-task regression of pChEMBL for 19 receptors/transporters
   from ChEMBL + PDSP Ki, with masked losses for missing labels and **scaffold-split**
   evaluation. Baselines: ridge and gradient-boosted trees on Morgan fingerprints. A Chemprop
   D-MPNN backend exists; its production CUDA run is not reported (see Limitations).
2. **Projection (Stage 2).** `effect(region) = sum_r weight_r * density(region, r)` with
   Hansen et al. PET receptor densities (Schaefer-100; other parcellations configurable).
   Magnitude-only by default; optional per-receptor agonist/antagonist signs.
3. **Validation (Stage 3).** Spatial correlation with observed drug maps, tested against
   **spin nulls** (1000 rotations; a naive permutation null is labelled `spin=False` and may not
   be reported as a spin test). Two tiers: *A* measured affinity (sanity) and *B* zero-shot, in
   which every training molecule sharing the drug's InChIKey connectivity layer or Bemis-Murcko
   scaffold is removed and `assert_held_out` re-derives the keys from the SMILES actually used
   for training, raising if any overlap remains.
4. **Inverse design.** Solve receptor weights that best produce a target regional profile,
   score molecules by pattern match, screen libraries (`scripts/design_screen.py`) and propose
   new structures with a mutation search (`scripts/design_generate.py`).
5. **Demo and figures.** `streamlit run app/app.py`; `scripts/plot_map.py`;
   `scripts/make_figures.py` regenerates figures and headline counts from saved artifacts.

## Data

| item | source | n |
|---|---|---|
| receptor density atlas | Hansen et al. 2022 (`netneurolab/hansen_receptors`), rebuilt with the paper's tracer selection; `fetch_atlas --verify` matches the paper's own table at r = 1.0000 for all 19 columns | 19 receptors, 100 parcels |
| observed drug maps | CBF, drug minus placebo. Seven from Dukart et al. 2018 (*Sci Rep*): methylphenidate (n=18), ketamine (26), risperidone 2 mg (21), haloperidol (21), escitalopram (18), olanzapine (21), midazolam (26). LSD from Carhart-Harris et al. 2016 (*PNAS*), NeuroVault collection 1083 (n=15) | 8 drugs |
| affinity labels | ChEMBL + PDSP Ki (pKi normalised; values outside 0-14 rejected) | per-receptor counts: `artifacts/label_audit.md` |

Full provenance, parcellation steps and known ambiguities: `docs/VALIDATION.md`.

## Results

### 1. The receptor-density basis is expressive enough
Fitting the 19 density maps to each observed map (ridge, scored on held-out regions) gives a
cross-validated r of **0.46 to 0.81 (median 0.67)**, spin-significant for **7 of 8** drugs
(methylphenidate is the exception, p = 0.106). The atlas and the maps are not the problem.

### 2. Affinity is not the weight that does it
Keeping each drug's *measured* affinity values and the real atlas, then shuffling which receptor
holds which value, the true assignment should beat its own shuffles if affinity
carried the signal. **0 of 6 testable drugs did** (true r vs shuffled mean |r|: methylphenidate
+0.019 vs 0.200, ketamine +0.064 vs 0.149, risperidone -0.346 vs 0.248, haloperidol -0.051 vs
0.216, LSD +0.013 vs 0.332, olanzapine +0.101 vs 0.258; all p > 0.27). This uses lab
affinity, so a better structure-to-affinity model cannot fix it.

### 3. The observed maps mostly do not resemble the drug's own target
Correlating each observed map with single receptor densities, midazolam's GABA-A ranks 16th of
19 and LSD's 5-HT2A 18th. The two drugs whose pre-registered target ranks near the top
(haloperidol, olanzapine) correlate *positively* with it despite being antagonists.

### 4. Affinity model comparison
On a scaffold split the repository records GBDT 0.387 > ridge 0.367 (mean held-out R², README).
The D-MPNN result is intentionally not reported until its CUDA/SLURM run completes.

### 5. Two-tier correlations
An early five-drug run (2026-10-20, `docs/VALIDATION.md`, "first real run") found Tier B
zero-shot 0/5 spin-significant and Tier A weak. **That run predates the atlas bug fix** (10 of
19 receptor columns were wrong; the fix removed a spurious significant LSD result), so its
numbers are superseded and should not be quoted. The corrected eight-drug tier table is
produced by:

```bash
python scripts/evaluate.py --report            # writes artifacts/validation_tiers.csv
python scripts/make_figures.py                  # writes artifacts/figures/numbers.md + fig_validation_r.png
```

It is deliberately not typed into this document: paste `numbers.md` here after running it, with
its null type (spin, 1000 rotations) intact.

### What the evidence supports
> Receptor density maps can explain drug-induced CBF maps well, but binding affinity is not the
> weight that does it: knowing which receptor a drug binds, and how tightly, predicts these
> maps no better than randomly reassigning those same affinities among receptors.

This does not rule out another transform of the affinity profile (occupancy at clinical dose,
efficacy, direction, pharmacokinetics) being the right weight. The high ceiling says such a
weight exists.

## Inverse design and generation: status

- A defect was found and fixed while building this stage: `score_affinity` compared raw pChEMBL
  values (about 4-10) with unitless least-squares weights using mean squared error. On random
  molecules that score correlated -0.995 with mean affinity and 0.02 with pattern match, i.e. it
  ranked the *weakest* binders first. It is now the correlation between the affinity vector and
  the solved weights (scale- and offset-invariant, in [-1, 1]); tests pin both properties.
- On a synthetic model the mutation search beat a random-mutation baseline with the same
  evaluation budget (that is a check that the search optimises its objective, not evidence of
  anything biological). It has not been run against a real trained artifact in this repository.
- Because affinity does not currently predict brain maps (Result 2), a high design score means
  "matches the weight vector solved from the target", **not** "will produce that regional effect".
  All generated structures are labelled `UNVALIDATED HYPOTHESIS`; the plausibility filter only
  removes peroxides and sulfur with no S=O/S=N, not unstable, toxic or unsynthesisable
  molecules.

## Limitations

- **n = 8 drugs, seven from one paper** (one lab, one ASL protocol, one pipeline), all CBF, an
  indirect readout. Luppi et al. 2023 BOLD maps were not found deposited.
- Cortex only (100 parcels); several of these drugs act mostly subcortically. `cammoun033` has an
  atlas but no observed-map ingestion or spin null.
- Density is not exposure (no pharmacokinetics); affinity is not function (direction is
  user-supplied, never predicted; pre-registered directions for catalog drugs are in
  `configs/validation_maps.yaml`).
- Haloperidol's nearest remaining training analogue has Tanimoto 0.93 after scaffold removal, so
  its "zero-shot" row is the least held out.
- The Dukart supplement ships `Zscore_` and `sZscore_` maps; the unprefixed ones were used and
  the alternative was not tested.
- Spin tests assign rotated parcels to their nearest unrotated one, so spun maps are not strict
  permutations (standard behaviour of the method).
- D-MPNN, real-artifact design runs and the corrected eight-drug tier table need real data and
  a GPU and were not run in this repository's offline environment.

## Reproduce

```bash
pip install -e ".[all]"
python -m smilesbrain.data.fetch_atlas --verify
python -m smilesbrain.data.fetch_affinity && python -m smilesbrain.data.assemble
python scripts/train_affinity.py --config configs/affinity_baseline.yaml
python scripts/fetch_parcel_coords.py && python scripts/make_spins.py --coords data/atlas/schaefer100_coords.csv --n-perm 1000 --out data/atlas/spins_scale100.npy
python scripts/evaluate.py --report --spins data/atlas/spins_scale100.npy
python scripts/make_figures.py
streamlit run app/app.py
```

Every script accepts `--seed` and writes a `*_manifest.json` (config, git commit, seed, input
hashes) beside its output.
