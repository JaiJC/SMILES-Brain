# SMILES → Brain

**Predict a molecule's receptor-binding profile from its chemical structure, and project it onto
PET receptor densities as an *exploratory* brain map.** The profile prediction matches a
nearest-analogue lookup on held-out scaffolds. The brain map is **not** validated: see
"Validation status" below.

You type a SMILES string. The model predicts the molecule's binding affinity across 19
neurotransmitter receptors/transporters, then projects that receptor-affinity vector onto
open PET-derived receptor density atlases to produce a predicted whole-brain effect map.
Because affinity is predicted from structure, this works on molecules that have **never been
synthesised or tested**.

```
        SMILES                 19-receptor                  receptor density            predicted
   "CN1CCC[C@H]1c1cccnc1"  ──►  affinity vector  ──►  ×  atlas (region × receptor)  ──►  brain map
      (structure)              (learned, pKi)          (Hansen et al. 2022, open)      (region scores)
```

## Why this is different from prior work

Prior work (Dukart 2018; Hansen 2022; Luppi 2023; and an April 2026 psychedelics preprint)
relates receptor atlases to brain scans using **lab-measured** binding affinities — so it only
works for drugs that already have assay data. SMILES-Brain replaces the measured-affinity input
with a **structure-to-affinity model**, unlocking two things nobody has shipped end-to-end:

1. **Zero-shot molecules** — predict a brain map for a compound with no binding data.
2. **Inverse design** — "light up region X, spare region Y → which molecules fit?"

## Validation status (full write-up: `docs/REPORT.md`)

| Claim | Evidence | Status |
|---|---|---|
| Structure -> receptor profile on held-out scaffolds | within-molecule rho +0.52 vs +0.15 (receptor-mean baseline), but +0.49 for a nearest-analogue kNN baseline (`docs/results/profile_eval.md`) | **Supported vs a molecule-blind baseline; no advantage over kNN** |
| Receptor densities can explain drug CBF maps | cross-validated r 0.46-0.81, 7 of 8 spin-significant | Supported |
| Binding affinity is the weight that does it | 0 of 6 drugs beat a receptor-shuffle null (lab affinity, so not a model problem) | **Not supported** |
| Occupancy + learned shared receptor gains does | pre-registered; 0 of 6 beat the null; criterion not met (`docs/results/learned_gains_report.md`) | **Not supported** (rules out strong effects only, 6 drugs) |
| Predicted brain map matches real drug maps (zero-shot) | 1 of 8 drugs survives multiple-testing correction | **Not supported** |

So the honest product today is a receptor-profile predictor (as good as nearest-analogue
lookup) with uncertainty and out-of-domain flags; the brain map is a labelled hypothesis view. The repo is built so that
better data can re-open the question: the tests, nulls, pre-registration and power tool are in
place (`docs/PREREGISTRATION.md`, `scripts/lodo_power.py`).

## Honest limitations (read before trusting any output)

- Receptor **density** says where a drug *can* act, not how much drug arrives (pharmacokinetics).
  Affinity ≠ functional effect: by default the projection is **magnitude-only** and treats every
  bound receptor as if the molecule were an agonist. If you know the direction, pass it
  (`--direction "5HT2a=+1,D2=-1"` or `project_affinity(..., direction=...)`; +1 agonist, −1
  antagonist) and the map reflects it. Receptors left out of `direction` stay magnitude-only
  and are reported as such. Directions are never predicted — you supply them.
- Results can be robustness-checked on other parcellations (`--parcellation`: `schaefer100`,
  `schaefer400`, and `cammoun033`, which adds subcortex). Spin tests apply to cortical
  parcellations only; subcortical parcels cannot be spun on the sphere.
- Predicted maps are hypotheses for prioritisation, **not** clinical or preclinical claims.
- Validation is correlational against a small set of open drug-response maps.

## Pipeline

| Stage | Module | Input | Output |
|-------|--------|-------|--------|
| 1. Affinity | `smilesbrain.models` | SMILES | 19 × pKi vector |
| 2. Projection | `smilesbrain.projection` | pKi vector + atlas | region × 1 effect map |
| 3. Validation | `smilesbrain.validation` | predicted vs. real drug maps | correlation, spin test |

## Quickstart

### Reproducible environment

The full development environment, including optional chemistry, PyTorch,
validation, and visualisation dependencies, is defined in
[`environment.yml`](environment.yml):

```bash
conda env create -f environment.yml
conda activate smilesbrain
pip install -e .
```

For a lightweight CPU-only install without optional heavy dependencies, use
the pinned core lock instead:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e .
```

The lock is intended for Python 3.10. The conda file includes a CPU-capable
PyTorch build. On an HPC with NVIDIA GPUs, remove the `pytorch` dependency
before creating the environment and install the CUDA-matched build from
[`pytorch.org`](https://pytorch.org/get-started/locally/), or use the
corresponding `pytorch` and `pytorch-cuda` conda packages for the cluster's
CUDA version.

All pipeline scripts accept `--seed`. Training uses the configured scaffold
split seed by default, and every script writes a `*_manifest.json` beside its
output containing the seed, configuration, git commit, timestamp, and hashes
of available input files. This makes offline reruns auditable and
reproducible.

```bash
pip install -e ".[dev]"
python -m smilesbrain.data.fetch_atlas          # Hansen receptor atlas (open)
python -m smilesbrain.data.fetch_atlas --parcellation schaefer400   # alternative parcellation
python -m smilesbrain.data.fetch_affinity        # ChEMBL + PDSP Ki training data
python scripts/train_affinity.py --config configs/affinity_baseline.yaml
python scripts/predict.py --smiles "CN1CCC[C@H]1c1cccnc1"  # nicotine example
```

For the production Chemprop D-MPNN run on a CUDA SLURM node:

```bash
sbatch slurm/train_dmpnn.sbatch configs/affinity_dmpnn_slurm.yaml
```

The batch script performs a CUDA/Chemprop preflight and refuses to fall back to CPU.
PDSP normalization rejects pKi values outside 0-14 and understands nM, pM, uM, mM,
and molar units; rejected rows are excluded rather than silently used for training.

## Demo, plotting, inverse design

```bash
pip install -e ".[app]"
streamlit run app/app.py                          # type a SMILES -> affinities + brain map
python scripts/plot_map.py artifacts/prediction.csv --html   # cortical-surface PNG (+ HTML)
python scripts/design_screen.py --smiles-file candidates.txt --high-regions 1,2,3 --low-regions 98,99
python scripts/design_generate.py --seed-smiles-file seeds.txt --high-regions 1,2,3 --low-regions 98,99
python scripts/make_figures.py                    # figures + headline counts from artifacts/
```

The demo shows a `DEMO MODE` banner whenever the trained artifact or real atlas is missing, and
always shows the validation caveat above. Design scores rank how well a molecule's predicted
affinity *pattern* matches the receptor weights solved from a target regional profile; generated
structures are **unvalidated hypotheses**, and given the validation result they are not evidence
of a regional effect. The full write-up is `docs/REPORT.md`.

## Data (all open)

- **Receptor atlas:** `netneurolab/hansen_receptors` (PET, 19 receptors/transporters, Schaefer-100).
- **Affinity training:** ChEMBL + PDSP Ki database.
- **Validation maps:** open drug-response maps (OpenNeuro / published parcellated maps).

## Status

Complete as a negative-result release. Built and tested: data pipeline, affinity models,
projection, validation harness with spin nulls, pre-registered learned-gains test, receptor-profile
evaluation, inverse-design screen and generator, demo app, CI (Python 3.10 and 3.12). Five of the
seven definition-of-done items are met (`docs/REPORT.md`, section 5); the two that are not need a
GPU. The scientific goal, a validated structure-to-brain-map predictor, is not achieved: see
"Validation status" above. All results are under `docs/results/`.
