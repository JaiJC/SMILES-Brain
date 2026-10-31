# SMILES-Brain — LLM Build Playbook

A complete, step-by-step instruction set you can feed to a coding LLM to build the entire
project. It is written to be executed **one task at a time**, in order. Each task says what to
do, which files to touch, and how to know it worked.

---

## 0. How to use this document with an LLM

Give the LLM this whole file plus access to the repository, then drive it like this:

1. Paste the **Ground rules** section first and tell it: *"Follow these rules for every task."*
2. Work **one task at a time**, in order. Paste a single task block (e.g. `P2-T3`) and say:
   *"Do exactly this task. Show me the diff and the test output. Do not start the next task."*
3. After each task, run the stated **acceptance check**. Only move on when it passes.
4. If the LLM wants to change files outside the task's listed files, make it explain why first.

Do not ask the LLM to "build the whole project" in one shot — it will hallucinate structure
and skip the honesty safeguards that make this project credible. The value is in the
discipline: scaffold splits, masked losses, spin tests, and labelled demo modes.

---

## Ground rules (apply to every task)

- **Language/stack:** Python ≥3.10. Core deps: numpy, pandas, scipy, scikit-learn, pyyaml,
  tqdm, requests. Heavy deps are **optional extras**: `rdkit` (chem), `torch` (torch),
  `matplotlib`/`nibabel` (viz). Never make the core package import torch or rdkit at top level.
- **Canonical receptor order:** `configs/receptors.yaml` is the single source of truth. Every
  matrix column, prediction vector, and atlas column must follow `ReceptorConfig.atlas_keys`
  order. Never hardcode a receptor list anywhere else.
- **Honesty is a feature, not a nicety.** Keep every safeguard:
  - scaffold splits are the default evaluation;
  - multi-task losses are **masked** (missing labels contribute no gradient);
  - the spin test refuses to pass off a naive permutation null as a real spin test;
  - any synthetic/fallback path prints a clearly-labelled `DEMO MODE` banner.
- **Offline-first.** Everything must run end-to-end on synthetic data with no network and no
  GPU, so tests and CI stay green. Real runs swap in fetched data and trained models.
- **Tests after every task.** `python -m pytest -q` must stay green. New behaviour needs a new
  test that does not require rdkit/torch/network.
- **Commit style:** one task = one commit, imperative subject, body explains *why*. Do not add
  model identifiers to commits.
- **No fabricated results.** The LLM must never invent correlation numbers, affinities, or
  brain maps. If data is missing, it writes code that fails loudly with guidance.

---

## Current state (already built — do not rebuild)

The repo already contains a working end-to-end scaffold. Read these before starting so you
extend rather than duplicate:

```
src/smilesbrain/
  config.py                 # ReceptorConfig loader (19 receptors, canonical order)
  features.py               # SMILES -> Morgan fingerprint (rdkit, guarded)
  splits.py                 # random + scaffold splits
  models/baseline.py        # MaskedMultiTaskRidge (sparse-label aware, sklearn)
  models/mlp.py             # MaskedMultiTaskMLP (torch, masked multi-task loss)
  projection/project.py     # affinity vector x atlas -> region map
  validation/spin.py        # map_correlation + spin_test (naive-null guard)
  validation/catalog.py     # drug-map catalog loader
  validation/maps.py        # observed-map loader (strict alignment)
  validation/nulls.py       # spin-null generation (netneurotools, cached)
  validation/harness.py     # evaluate predicted vs observed maps
  data/fetch_affinity.py    # ChEMBL affinities per receptor
  data/fetch_atlas.py       # Hansen PET atlas (parcellated)
  data/assemble.py          # wide (molecule x receptor) label table
  data/synthetic.py         # offline synthetic generators (DEMO only)
configs/   receptors.yaml, affinity_baseline.yaml, validation_maps.yaml
scripts/   train_affinity.py, predict.py, evaluate.py, make_spins.py
slurm/     train_affinity.sbatch, fetch_data.sbatch
tests/     test_pipeline.py, test_validation.py   (12 passing)
```

The tasks below finish the project: real data, a stronger model, real validation data, inverse
design, a demo, and reproducibility.

---

## Phase P0 — Environment & reproducibility

### P0-T1 — Pin the environment
**Files:** `environment.yml` (new), `requirements-lock.txt` (new), `README.md` (edit).
**Do:** Create a conda `environment.yml` with python=3.10 and the core + `[all]` extras
(rdkit, torch CPU+CUDA note, matplotlib, nibabel, netneurotools, chemprop). Add a comment on
how to pick the CUDA build of torch on the HPC. Generate a pip lock for the core deps.
**Check:** `conda env create -f environment.yml` succeeds on a clean machine (or document the
exact commands if conda is unavailable); `python -c "import smilesbrain"` works.

### P0-T2 — Continuous integration
**Files:** `.github/workflows/ci.yml` (new).
**Do:** GitHub Actions: on push/PR, set up Python 3.10 and 3.12, `pip install -e ".[dev]"`,
run `ruff check` and `pytest -q`. Do **not** install rdkit/torch in CI — the suite is designed
to pass without them.
**Check:** Workflow is valid YAML; locally `ruff check .` and `pytest -q` pass.

### P0-T3 — Global seeding & run manifests
**Files:** `src/smilesbrain/utils/seed.py` (new), wire into the three scripts.
**Do:** Add `set_global_seed(seed)` (python, numpy, torch if present). Each script writes a
small JSON manifest next to its output: config used, git commit, seed, timestamp, input file
hashes.
**Check:** Running any script produces a `*_manifest.json`; re-running with the same seed and
inputs reproduces identical outputs (for the sklearn baseline).

---

## Phase P1 — Real data

### P1-T1 — Verify the atlas filename mapping
**Files:** `src/smilesbrain/data/fetch_atlas.py` (edit `KEY_ALIASES` if needed).
**Do:** Run `python -m smilesbrain.data.fetch_atlas --list`. Compare the printed upstream
filenames against our 19 `atlas_key`s. Fix `KEY_ALIASES` so every receptor maps. Then fetch.
**Check:** `python -m smilesbrain.data.fetch_atlas` writes `data/atlas/density_scale100.csv`
with shape `100 x 19` (or prints a precise WARNING naming any receptor with no map).

### P1-T2 — Fetch ChEMBL affinities
**Files:** none (run the existing fetcher).
**Do:** `python -m smilesbrain.data.fetch_affinity` (login node, internet). It is resumable.
**Check:** `data/affinity/<key>.csv` exists for each receptor; print per-receptor molecule
counts. Flag receptors with <200 molecules (expect VAChT, maybe NMDA/GABAa subunits sparse).

### P1-T3 — Assemble and audit the label table
**Files:** `scripts/audit_labels.py` (new).
**Do:** Run `python -m smilesbrain.data.assemble`. Then write an audit script that reports, per
receptor: n molecules, pChEMBL distribution (min/median/max), and % of the full matrix that is
observed. Save a short markdown report to `artifacts/label_audit.md`.
**Check:** `data/labels.parquet` exists; audit report generated; label density printed.

### P1-T4 — Add PDSP Ki as a second affinity source (optional but valuable)
**Files:** `src/smilesbrain/data/fetch_pdsp.py` (new), `data/assemble.py` (edit to merge).
**Do:** Pull Ki values from the PDSP Ki database for the same UniProt targets, convert to
pKi, and merge with ChEMBL (median-collapse duplicates per molecule+receptor). Keep a `source`
column. PDSP has data for receptors ChEMBL is thin on.
**Check:** Merged table has ≥ as many labels as ChEMBL alone; no duplicate (smiles, receptor)
rows; a test asserts the merge is idempotent.

---

## Phase P2 — Affinity models

### P2-T1 — Baseline on real data
**Files:** none (run), `configs/affinity_baseline.yaml` (tune alpha).
**Do:** Train the Ridge baseline on real labels with a **scaffold split**. Grid-search `alpha`
on a validation fold.
**Check:** `scripts/train_affinity.py` prints per-receptor held-out R²; saves artifact +
report. Record the mean R² over non-sparse receptors as the number to beat.

### P2-T2 — Add a random-forest / GBDT baseline on fingerprints
**Files:** `src/smilesbrain/models/trees.py` (new), `train_affinity.py` (add backend).
**Do:** Implement a NaN-aware per-receptor `HistGradientBoostingRegressor` (it tolerates NaN
features; here targets are masked per receptor as in the Ridge baseline). Add `backend: gbdt`.
**Check:** New backend trains and reports; compare R² to Ridge in `artifacts/model_compare.csv`.

### P2-T3 — D-MPNN graph model (the headline model)
**Files:** `src/smilesbrain/models/dmpnn.py` (new), `configs/affinity_dmpnn.yaml` (new),
`train_affinity.py` (add backend), `slurm/train_dmpnn.sbatch` (new).
**Do:** Wrap `chemprop` (directed message-passing NN) as a multi-task regressor with masked
targets. Train from SMILES (not fingerprints), scaffold split, GPU. Keep the same report
interface so it drops into the existing comparison.
**Check:** On the scaffold split, D-MPNN mean R² ≥ the best baseline on non-sparse receptors.
Save `artifacts/affinity_dmpnn.pt` + report. A tiny synthetic smoke test runs on CPU.

### P2-T4 — Uncertainty & applicability domain
**Files:** `src/smilesbrain/models/uncertainty.py` (new), `predict.py` (surface it).
**Do:** Add per-prediction uncertainty: ensemble (train k D-MPNNs with different seeds) or MC
dropout, plus a Tanimoto-to-training-set "applicability domain" distance. `predict.py` prints
a confidence per receptor and flags out-of-domain molecules.
**Check:** `predict.py` output includes uncertainty; a test asserts OOD molecules get flagged.

---

## Phase P3 — Projection

### P3-T1 — Direction-aware weighting (agonist vs antagonist)
**Files:** `src/smilesbrain/projection/project.py` (extend), `configs/receptors.yaml` (add a
`functional_direction` hook if data allows).
**Do:** The current projection uses affinity magnitude only. Add an optional functional-effect
sign per receptor (agonist +1 / antagonist −1) when known for a molecule, so the map reflects
*direction* of modulation, not just presence. Default to magnitude-only with a clear note when
direction is unknown.
**Check:** With explicit directions supplied, flipping a receptor's sign flips its contribution
to the map; a test covers it. The README limitation note is updated to reflect the new option.

### P3-T2 — Subcortex & alternative parcellations
**Files:** `configs/receptors.yaml` (parcellation entries), `data/fetch_atlas.py` (edit).
**Do:** Support at least one additional parcellation (e.g. Schaefer-400, and a subcortex
atlas) so results are robustness-checkable. Keep `n_regions` wired from config everywhere.
**Check:** Pipeline runs unchanged on a second parcellation; a test loads both configs.

---

## Phase P4 — Validation (critical path)

### P4-T1 — Observed-map ingestion helper
**Files:** `src/smilesbrain/data/parcellate.py` (new), `scripts/ingest_map.py` (new).
**Do:** Given a volumetric statistical map (NIfTI) and a Schaefer atlas, produce a
one-value-per-region CSV at the path a `validation_maps.yaml` entry expects (nilearn
`NiftiLabelsMasker`). Record provenance (source, n, contrast) in a sidecar JSON.
**Check:** Feeding a NIfTI yields a CSV the harness's `load_observed_map` accepts (length ==
n_regions). A test uses a tiny synthetic NIfTI.

### P4-T2 — Source the real drug maps
**Files:** `data/validation/*.csv` (new, gitignored), `docs/VALIDATION.md` (log provenance).
**Do:** For each catalog drug, locate an open group-level response map (OpenNeuro; the papers
in each `source`). Parcellate with P4-T1. Log exactly where each came from and its n.
**Check:** `scripts/evaluate.py` leaves DEMO mode (finds real maps). **If maps are scarce,
record that finding prominently — it changes the project's claim.**

### P4-T3 — Real spin nulls
**Files:** `scripts/fetch_parcel_coords.py` (new), run `make_spins.py`.
**Do:** Obtain Schaefer-100 parcel spherical centroids + hemisphere (neuromaps/nilearn
surface annotation), save as `x,y,z,hemi` CSV, generate `data/atlas/spins_scale100.npy`.
**Check:** `evaluate.py --spins ...` reports `spin=True`; p-values now valid.

### P4-T4 — Two-tier evaluation report
**Files:** `scripts/evaluate.py` (extend), `src/smilesbrain/validation/report.py` (new).
**Do:** Produce a report that clearly separates (a) **measured-affinity reproduction** (sanity)
from (b) **zero-shot prediction** (held-out chemotype). For (b), ensure each test drug's
scaffold is excluded from affinity-model training. Output a markdown table + a figure.
**Check:** `artifacts/validation_report.md` shows both tiers with n, null type, r, spin-p,
sign agreement. Zero-shot drugs are provably absent from training (assert in code).

---

## Phase P5 — Inverse design (the "turn heads" feature)

### P5-T1 — Target-profile scoring
**Files:** `src/smilesbrain/design/score.py` (new).
**Do:** Given a desired regional profile (e.g. "high in region set A, low in B"), derive the
receptor-weight vector that best produces it (least squares against the atlas), then score any
molecule by how well its predicted affinity matches that weight vector.
**Check:** A molecule whose affinity equals the solved weights scores ~best; a test covers it.

### P5-T2 — Library screen
**Files:** `scripts/design_screen.py` (new).
**Do:** Score a candidate library (ChEMBL subset or a provided SMILES list) against a target
profile; output a ranked table with predicted maps and uncertainty, flagging OOD hits.
**Check:** Script ranks a library; top hits are reproducible under a fixed seed.

### P5-T3 — (Stretch) generative proposal
**Files:** `src/smilesbrain/design/generate.py` (new).
**Do:** Use a simple, dependency-light generator (e.g. SELFIES random/mutation or a small
pretrained model if available) to propose novel SMILES optimised toward the target profile.
Clearly mark outputs as **unvalidated hypotheses**.
**Check:** Proposals are valid molecules (parse under rdkit) and improve the target score over
random baselines.

---

## Phase P6 — Demo, docs, outreach assets

### P6-T1 — Brain-map visualisation
**Files:** `scripts/plot_map.py` (new), `src/smilesbrain/viz/surface.py` (new).
**Do:** Render a region-score vector on a cortical surface (nilearn plotting) to PNG/HTML.
**Check:** `plot_map.py artifacts/prediction.csv` produces a figure.

### P6-T2 — Interactive demo
**Files:** `app/` (new — streamlit or a small FastAPI+JS page).
**Do:** Text box for SMILES → show predicted affinity bars + an interactive 3D brain + the
honesty/limitations banner. Load the trained model + atlas; DEMO banner when absent.
**Check:** `streamlit run app/app.py` serves locally; typing nicotine renders a map.

### P6-T3 — Reproducible paper/report figures
**Files:** `notebooks/` or `scripts/make_figures.py` (new).
**Do:** One command regenerates every figure/number in the writeup from artifacts, so results
are auditable.
**Check:** Running it reproduces the figures from committed artifacts.

### P6-T4 — Writeup & outreach
**Files:** `docs/REPORT.md` (new).
**Do:** Methods, honest results (two-tier, spin-tested), limitations, and a one-paragraph
"why this matters". Keep claims proportional to evidence.
**Check:** A domain reader can tell exactly what was done, on what data, with what null.

---

## Definition of done (project level)

- `pytest -q` green without rdkit/torch/network; a GPU smoke test green with them.
- Real atlas + real labels fetched and audited; label density reported per receptor.
- D-MPNN beats baselines on a scaffold split (non-sparse receptors), with uncertainty.
- Validation report shows **zero-shot** (held-out chemotype) correlations with **real** drug
  maps, **spin-tested**, with sign agreement — and states n and nulls for every number.
- Inverse-design screen ranks a library toward a target regional profile.
- A demo anyone can run, and a report that states limitations as plainly as results.
