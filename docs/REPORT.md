# SMILES-Brain: final report

**Bottom line.** A molecule's receptor-binding profile can be predicted from structure on
held-out scaffolds, well above a molecule-blind baseline, **but no better than simply copying the
profile of its nearest training analogues**. Turning a profile into a whole-brain map that
matches measured drug effects is **not supported** by the available data, in two independent
pre-planned tests. Every number below is traced to a file under `docs/results/` or reproduced
by the command shown.

## Summary of findings

| # | Finding | Status |
|---|---|---|
| 1 | Structure -> 19-receptor profile, scaffold held out: within-molecule rho +0.52 vs +0.15 for a receptor-mean baseline (difference +0.36, CI [+0.23, +0.49]), **but +0.49 for a nearest-analogue (kNN) baseline (difference +0.03, CI [-0.02, +0.08])** | Supported against a molecule-blind baseline only; **no advantage over kNN** (exploratory) |
| 2 | A weighted sum of the 19 receptor density maps reproduces the 8 drug CBF maps (cross-validated r median 0.67, 7/8 spin-significant) | Supported |
| 3 | Binding affinity is not the weight: 0 of 6 drugs beat a receptor-shuffle null | Established (negative) |
| 4 | Occupancy + learned shared receptor gains does not rescue it: 0 of 6 beat the null; pre-registered criterion not met | Established (negative); rules out strong effects only |
| 5 | Zero-shot brain-map correlation: 1 of 8 drugs survives multiple-testing correction (risperidone) | **Not supported** |
| 6 | More validation maps are not readily available on NeuroVault (psilocybin 0, MDMA 0, benzodiazepines 0 matching collections) | Established for NeuroVault text search only |

**What can be claimed:** "a reproducible, pre-registered harness showing that receptor density
explains drug CBF maps while affinity does not", and "a scaffold-validated receptor-profile
predictor that matches a nearest-analogue lookup, with uncertainty and applicability-domain
flags". **What cannot:** that the model adds information beyond close analogues in ChEMBL, or
that predicted brain maps for new molecules match real drug effects.

## 1. Question and pipeline

SMILES -> 19-receptor affinity vector (Stage 1) -> receptor density atlas projection (Stage 2)
-> region map, validated against observed drug maps with spin nulls (Stage 3). Prior work does
this with lab-measured affinity; the proposed novelty was replacing it with a structure-based
prediction (zero-shot molecules) and running it backwards (inverse design).

## 2. Data

| item | source | n |
|---|---|---|
| receptor density atlas | Hansen et al. 2022, rebuilt with the paper's tracer selection; `fetch_atlas --verify` matches the paper's own table at r = 1.0000 for all 19 columns | 19 receptors, 100 parcels |
| observed drug maps | CBF, drug minus placebo. Seven from Dukart et al. 2018 (*Sci Rep*): methylphenidate (n=18), ketamine (26), risperidone 2 mg (21), haloperidol (21), escitalopram (18), olanzapine (21), midazolam (26). LSD from Carhart-Harris et al. 2016 (*PNAS*), NeuroVault 1083 (n=15) | 8 drugs |
| affinity labels | ChEMBL + PDSP Ki, pChEMBL (values outside 0-14 rejected) | 53,342 molecules x 19 receptors, 7.2% observed |

Provenance, hashes and how the maps were rebuilt offline: `docs/VALIDATION.md`.

## 3. Results

### 3.1 Stage 1: the receptor profile is predictable, mostly via close analogues (`docs/results/profile_eval.md`)
Ridge on Morgan fingerprints, scaffold split (42,674 train / 10,668 test, no scaffold shared,
asserted; the saved artifact is verified to have been trained on exactly this fold). Mean
held-out R^2 0.379 over non-sparse receptors (0.408 over the 16 receptors with n_test >= 200).

For the 1,206 test molecules with at least three measured receptors (78 scaffolds), comparing
predicted with measured profile across those receptors (95% CI by resampling whole scaffolds).
Two baselines: the receptor-mean (molecule-blind) and kNN, the similarity-weighted mean of the
25 nearest training molecules measured at each receptor.

| | model | receptor-mean | model - receptor-mean | kNN (k=25) | model - kNN |
|---|---|---|---|---|---|
| within-molecule rho | +0.518 [+0.448, +0.592] | +0.153 [-0.006, +0.334] | **+0.364 [+0.231, +0.494]** | +0.490 [+0.405, +0.581] | **+0.027 [-0.019, +0.075]** |
| top receptor correct | 0.609 | 0.417 | | 0.594 | |

(chance for top receptor: 0.310.) **The model clearly beats a molecule-blind baseline and is
statistically indistinguishable from nearest-analogue lookup.** By nearest-training Tanimoto:
>= 0.5 (918 molecules, 73 scaffolds) model +0.557, receptor-mean +0.116, kNN +0.537; 0.3-0.5
(266, 36) model +0.418, kNN +0.353 (difference +0.065 [-0.012, +0.147]); below 0.3 only 22
molecules from 6 scaffolds, so no conclusion is possible (model +0.078, kNN +0.193). A scaffold
split still leaves close analogues, so this measures generalisation to near-neighbour
chemotypes; whether the model beats kNN on genuinely novel chemistry is untested.

Catalog drugs, chemotype removed from training, predicted vs measured profile (6 drugs with
three or more measured receptors): the measured top receptor is the predicted top receptor for
4 of 6 (LSD, haloperidol, olanzapine, risperidone) and not for ketamine (NMDA; nearest training
molecule Tanimoto 0.33) or methylphenidate (DAT; 0.57). rho above 0.5 for the same four. This
was compared with the receptor-mean only, not kNN.

Caveats: designed once and run once (the kNN baseline was added after seeing the first run,
because it was listed as missing), **not pre-registered**. Molecules profiled on three or more
receptors are not a random sample of ChEMBL (assay panels are chosen by chemists), which can
make within-molecule ranking easier or harder than for an arbitrary compound.

### 3.2 The receptor-density basis is expressive enough
Fitting the 19 density maps to each observed map (ridge, scored on held-out regions) gives a
cross-validated r of 0.46 to 0.81 (median 0.67), spin-significant for 7 of 8 drugs
(methylphenidate is the exception, p = 0.106).

### 3.3 Affinity is not the weight that does it
Keeping each drug's measured affinity values and the real atlas, then shuffling which receptor
holds which value: **0 of 6 testable drugs** beat their shuffles (true r vs shuffled mean |r|:
methylphenidate +0.019 vs 0.200, ketamine +0.064 vs 0.149, risperidone -0.346 vs 0.248,
haloperidol -0.051 vs 0.216, LSD +0.013 vs 0.332, olanzapine +0.101 vs 0.258; all p > 0.27).
This uses lab affinity, so a better structure-to-affinity model cannot fix it. The observed
maps also mostly do not resemble the drug's own target (midazolam's GABA-A ranks 16th of 19,
LSD's 5-HT2A 18th), which points at CBF as an indirect readout.

### 3.4 Second experiment: learned receptor gains (pre-registered, run, not supported)
Hypothesis: affinity fails because binding saturates (potency is not occupancy) and receptors
differ in how they move blood flow. `docs/PREREGISTRATION.md` (committed 2026-10-31 15:28 IST,
before any map was downloaded or any model run; SHA-256 `c49d8461...`) fixed the test: one
shared gain per receptor applied to occupancy, fitted leave-one-drug-out, against a
receptor-shuffle null, a mean-map baseline, and a hierarchical Tier A -> Tier B rule. Run as
registered (5000 shuffles, 1000 spin rotations, 100 nM): `docs/results/learned_gains_report.md`.

**CRITERION NOT MET.** Tier A: 0 of 6 drugs beat the shuffle null (4 needed), median held-out r
+0.086, median gain over the mean-map baseline -0.010; Tier B is not interpreted by rule; all
sensitivity analyses (10 and 1000 nM, leave-class-out) are also not met (at most 1 drug beat
the null). Power at the size available (6 evaluable drugs, 5 training drugs each), measured
post hoc by `scripts/lodo_power.py --n-drugs 6`: 75% at a true held-out r of about 0.67, 45% at
0.51, 12% at 0.38, 0% false positives on null data. This **rules out a strong gain-model signal
but not a weak one.** Not tested: per-drug dose and pharmacokinetics, non-linear transforms,
subcortex, non-CBF readouts, richer profiles for escitalopram and midazolam (1-2 measured
receptors).

### 3.5 Two-tier brain-map correlations (corrected atlas, 8 drugs, 1000 spin rotations)
From `docs/results/validation_tiers.csv` (`validation_report.md`):

| tier | n drugs | median r | p<0.05 | BH p<0.05 | expected sign correct |
|---|---:|---:|---:|---:|---:|
| A: measured affinity | 6 | +0.016 | 1/6 | 0/6 | 4/6 |
| B: zero-shot | 8 | +0.045 | 1/8 | 1/8 | 3/8 |
| A-dir: pre-registered directions | 8 | -0.219 | 0/8 | 0/8 | 4/8 |
| B-dir: pre-registered directions | 8 | -0.212 | 0/8 | 0/8 | 4/8 |

The one finding that survives correction (risperidone, Tier B r = -0.489, BH p = 0.032) is one
result among 24 drug-tier tests and is not corroborated by the shuffle diagnostic. An earlier
five-drug table predates the atlas fix (which removed a spurious LSD result) and is superseded.

### 3.6 Is more validation data available? (`docs/results/neurovault_scouting.md`)
All 18,062 NeuroVault collections were downloaded and matched on name and description text.
Matching collections: psilocybin 0, MDMA 0, olanzapine 0, benzodiazepines 0, LSD 1 (the one
used), ketamine 4, haloperidol 2, methylphenidate 1; the few hits are mostly task-fMRI or
meta-analyses, not resting drug-vs-placebo maps. This is a keyword search of collection text
(image-level metadata was not searched, and some names such as `LSD_gradients` are missed), so
it bounds what is *easily* findable, not what exists. Maps for more drugs would more likely
come from asking authors (Luppi 2023 and others) than from NeuroVault.

## 4. Inverse design and generation

- A defect was fixed here: the original design score compared raw pChEMBL values (4-10) with
  unitless least-squares weights, so on random molecules it correlated -0.995 with mean
  affinity and 0.02 with pattern match, i.e. it ranked the weakest binders first. It is now the
  scale-invariant correlation between affinity vector and solved weights.
- On a synthetic model the mutation search beats a random-mutation baseline at equal budget
  (a check that it optimises its objective, not biology). Not run on the real artifact.
- Given 3.3-3.5, a design score that targets a **regional brain effect** is not validated.
  Designing toward a **receptor profile** is the use that 3.1 supports. All generated
  structures are labelled `UNVALIDATED HYPOTHESIS`; the plausibility filter removes only
  peroxides and sulfur without S=O/S=N.

## 5. Limitations

- **n = 8 drugs, seven from one paper** (one lab, one ASL protocol), all CBF; only 6 evaluable
  for measured-affinity tests. Strong claims in either direction need more drugs.
- Cortex only (100 parcels); several of these drugs act subcortically.
- Density is not exposure; affinity is not function; direction is supplied, never predicted.
- Haloperidol's nearest remaining training analogue has Tanimoto 0.93 after scaffold removal.
- The Dukart supplement ships `Zscore_` and `sZscore_` maps; the unprefixed ones were used.
- The D-MPNN production run needs a GPU and was not run (a CPU D-MPNN recorded in an earlier
  commit message scored below ridge and GBDT: 0.247 vs 0.367 and 0.387). 3.3-3.4 show affinity
  quality is not the bottleneck for brain maps; for the profile task (3.1) it could matter.
- Everything runs offline on synthetic data for testing; real-data numbers above were produced
  once on this repository's code and are reproducible with the commands below.

## 6. Recommended next steps

1. **Ship the supported claim, stated precisely.** A scaffold-validated receptor-profile
   predictor with uncertainty and out-of-domain flags that matches a nearest-analogue lookup;
   present the brain map as exploratory. Do not claim it adds information beyond analogues.
2. **If a brain-map claim is still wanted:** obtain more and better observed maps (more drugs,
   a second lab, PET occupancy or receptor-specific readouts instead of CBF), pre-register the
   test as in `docs/PREREGISTRATION.md`, and size it with `scripts/lodo_power.py`.
3. **Test whether a learned model can beat kNN at all:** run the D-MPNN on GPU against the kNN
   baseline, and pre-register a confirmatory test on a time split or newly deposited ChEMBL
   data, restricted to molecules with nearest-training Tanimoto below 0.5 where the two differ.

## 7. Reproduce

```bash
pip install -e ".[all]"
python -m smilesbrain.data.fetch_atlas --verify
python -m smilesbrain.data.assemble --from-merged data/affinity_merged   # or fetch_affinity + assemble
python scripts/train_affinity.py --config configs/affinity_baseline.yaml
python scripts/eval_profiles.py                                          # section 3.1
# maps + nulls: docs/VALIDATION.md ("Reproducing the data without ChEMBL/OSF")
python scripts/evaluate.py --report --spins data/atlas/spins_scale100.npy   # 3.2, 3.3, 3.5
python scripts/learned_gains.py --spins data/atlas/spins_scale100.npy       # 3.4
python scripts/lodo_power.py --n-drugs 6
python scripts/scout_neurovault.py --out docs/results/neurovault_scouting.md  # 3.6
python scripts/make_figures.py
streamlit run app/app.py
```

Every script accepts `--seed` and writes a `*_manifest.json` (config, git commit, seed, input
hashes) beside its output.
