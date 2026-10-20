# SMILES → Brain

**Predict the whole-brain action map of a molecule from its chemical structure alone — no lab assay, no scan, no animal required.**

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

## Validation status: the central premise does not hold on the data we have

Read this before the pitch above. On 8 drugs with open, group-level CBF response maps
(Schaefer-100, 1000 spin rotations), three diagnostics give a consistent answer
(`docs/VALIDATION.md`, `artifacts/validation_report.md`):

1. A weighted sum of the 19 receptor density maps **can** reproduce these drug maps well:
   cross-validated ceiling r of 0.46-0.81, spin-significant for 7 of 8 drugs. The atlas and
   the maps are fine.
2. But **binding affinity is not that weight**. Shuffling which receptor holds which measured
   affinity value does *as well or better* than the true assignment: **0 of 6 testable drugs
   beat that receptor-permutation null**.
3. This is measured with **lab affinity**, not predicted, so a better affinity model does not
   fix it. (For the record the model work was done anyway: GBDT 0.387 > ridge 0.367 >
   D-MPNN 0.247 mean held-out R² on a scaffold split.)

So the honest claim today is: *the receptor-density basis is expressive enough, and finding
the transform from a molecule to the right weight vector is an open problem.* Affinity
magnitude is not it. The repo is built so that question can be attacked and the answer
checked -- which is the point of the harness, the pre-registered directions and the nulls.

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

On an HPC: see `slurm/` for ready SLURM batch scripts.

## Data (all open)

- **Receptor atlas:** `netneurolab/hansen_receptors` (PET, 19 receptors/transporters, Schaefer-100).
- **Affinity training:** ChEMBL + PDSP Ki database.
- **Validation maps:** open drug-response maps (OpenNeuro / published parcellated maps).

## Status

Early, actively built. Stage 1 baseline (Morgan fingerprint multi-task MLP) + Stage 2 projection
run end-to-end; a D-MPNN affinity model and inverse design are in progress. See `docs/ROADMAP.md`.
