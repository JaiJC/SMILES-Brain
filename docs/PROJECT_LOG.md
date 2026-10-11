# SMILES-Brain — What We've Built So Far, and Why It Matters

A plain-language record of the project to date. It explains the idea, then walks through every
piece we built, what it does, and why that choice matters — both scientifically and for how the
work will be judged by reviewers, labs, and interviewers.

---

## 1. The idea in one paragraph

You type a molecule's chemical structure (a SMILES string). The system predicts where in the
brain that molecule will act — a whole-brain map — **without any lab assay, brain scan, or
animal study**. It does this in two moves: first predict how strongly the molecule binds each
of 19 neurotransmitter receptors/transporters (from structure alone), then overlay those
binding strengths on published maps of where each receptor physically sits in the brain. A
brain region "lights up" when it is rich in the receptors the molecule binds.

**Analogy:** the brain is a city; receptors are different kinds of door-locks spread unevenly
across its neighbourhoods. The receptor-density atlas (already published) is the map of which
locks are where. A drug is a key. We predict which locks the key fits (from the key's shape
alone), then colour the city by how many fitting locks each neighbourhood has.

## 2. Why it's novel (and where the honest line is)

Prior work — Dukart (2018), Hansen (2022), Luppi (2023), and an April-2026 psychedelics
preprint — relates receptor atlases to brain scans using **lab-measured** binding affinities.
That only works for drugs that already have assay data. Our twist is to **predict** the
affinities from structure, which unlocks two things nobody has shipped end-to-end:

1. **Zero-shot molecules** — a brain map for a compound nobody has ever tested.
2. **Inverse design** — "light up region X, spare region Y → which molecules fit?"

We are deliberate about not overclaiming: the April-2026 preprint is close prior art, so the
novelty is the *structure-predicted, assay-free, end-to-end* pipeline plus inverse design — not
the receptor-map projection itself.

---

## 3. The pipeline, stage by stage

```
   SMILES  ──►  19-receptor affinity  ──►  × receptor-density atlas  ──►  brain map  ──►  validate
 (structure)      (Stage 1: learned)         (Stage 2: projection)        (region scores)  (Stage 3)
```

---

## 4. What we built, step by step

Each subsection = a real piece of the repository. "Why it matters" is the part to internalise
for interviews — it's where the engineering judgement shows.

### Step 1 — Receptor configuration as the single source of truth
**File:** `configs/receptors.yaml`, `src/smilesbrain/config.py`.
We list the 19 Hansen-atlas receptors once, each tied to its protein target (UniProt accession,
used to pull training data) and its neurotransmitter system. Two caveat flags are baked in:
`complex` (a multi-subunit channel, so single-subunit affinity is only a proxy) and `sparse`
(little public ligand data expected). The config enforces a **canonical ordering**.

**Why it matters:** in a pipeline where a vector of 19 numbers flows through a model, a
projection, and a validation step, the single most dangerous bug is silent **column
misalignment** — receptor 5 of the model lining up with receptor 7 of the atlas. Nothing
crashes; the results are just quietly wrong. Centralising the order kills that whole class of
bug. Encoding the `complex`/`sparse` caveats in the data (not a comment) means every downstream
report can flag unreliable receptors automatically — honesty by construction.

### Step 2 — Turning molecules into numbers (featurisation)
**File:** `src/smilesbrain/features.py`.
We convert each SMILES into a Morgan fingerprint — a fixed-length binary vector marking which
chemical substructures are present. Molecules that don't parse are dropped (and their labels
dropped with them).

**Why it matters:** fingerprints are the honest, fast **baseline** representation. They let us
stand up the whole pipeline and get a number to beat before investing in a graph neural network.
RDKit is an *optional* dependency, imported lazily — so the rest of the codebase, the tests, and
CI stay importable on a laptop with nothing heavy installed.

### Step 3 — Honest train/test splitting
**File:** `src/smilesbrain/splits.py`.
We default to **scaffold splits**: molecules are grouped by their core chemical skeleton
(Bemis-Murcko scaffold) and whole skeletons are held out for testing.

**Why it matters:** this is the difference between a project that survives scrutiny and one that
doesn't. A random split lets near-identical analogues sit in both train and test, so a model
that merely memorises scores beautifully — and meaninglessly. A scaffold split measures
generalisation to **new chemotypes**, which is exactly the zero-shot setting we claim. We chose
the harder evaluation on purpose; a sharp interviewer checks for this first.

### Step 4 — The affinity models (Stage 1)
**Files:** `src/smilesbrain/models/baseline.py` (Ridge), `src/smilesbrain/models/mlp.py` (MLP).
Both are **multi-task** (one model predicts all 19 receptors) and both handle **sparse labels**:
real binding data is patchy — most molecules are tested against only a few receptors. The Ridge
baseline fits one regressor per receptor on just that receptor's observed rows. The PyTorch MLP
shares a trunk across receptors and uses a **masked loss**, so a missing label contributes no
gradient.

**Why it matters:** sparse, masked multi-task learning is the correct response to how ChEMBL
actually looks. Training 19 independent models throws away the shared chemistry; a naive
multi-output model can't even ingest the missing values. The masked multi-task design lets one
model learn transferable chemistry from the whole patchy table — and the MLP is GPU-ready for
your SLURM cluster while the Ridge baseline keeps a cheap, honest reference point.

### Step 5 — Projection: affinities to a brain map (Stage 2)
**File:** `src/smilesbrain/projection/project.py`.
The core model is deliberately simple and interpretable:
`effect(region) = Σ_r weight_r × density(region, r)`. The weights come from the molecule's
predicted affinities; three weighting modes (`raw`, `zscore`, `softmax`) let you ask for
absolute potency, relative receptor preference, or selectivity emphasis. Receptors with no
prediction or no atlas map are dropped, and the set actually used is returned.

**Why it matters:** keeping Stage 2 simple and linear means every prediction is **explainable**
— you can point to exactly which receptors drove a region's score. That interpretability is
worth more here than a black-box gain, both scientifically and for trust. Returning the
"receptors used" makes coverage honest: you always know how much of the map is real signal
versus missing data.

### Step 6 — Data acquisition
**Files:** `data/fetch_affinity.py` (ChEMBL), `data/fetch_atlas.py` (Hansen), `data/assemble.py`.
`fetch_affinity` pulls Ki/Kd/IC50 from ChEMBL per target and harmonises them to pChEMBL (a
single −log-molar affinity scale), median-collapsing duplicate measurements. `fetch_atlas`
downloads the Hansen PET receptor maps, z-scores and averages multiple tracers per receptor.
`assemble` merges everything into one wide (molecule × receptor) table with NaNs where unmeasured.

**Why it matters:** these are resumable, polite, and defensive (clear warnings when an upstream
filename or target doesn't resolve). pChEMBL harmonisation is the quiet but crucial step that
makes Ki, Kd and IC50 comparable on one axis — without it the training targets would be
apples-and-oranges. All sources are open, which keeps your data-availability score high.

### Step 7 — Offline synthetic generators
**File:** `src/smilesbrain/data/synthetic.py`.
Generators for fake-but-learnable affinity labels, a smooth synthetic atlas, and a "noised
true projection" observed map. Every one is loudly labelled as **not scientifically
meaningful**.

**Why it matters:** this is what lets the *entire* pipeline — train, project, validate — run
with no network and no downloads. That means tests are deterministic, CI is green, and you can
develop on a plane. The "noised true projection" is a neat trick: a correct validation harness
*must* recover a strong correlation from it, so it doubles as a self-test of the harness.

### Step 8 — The validation harness (Stage 3) — the critical path
**Files:** `validation/catalog.py`, `validation/maps.py`, `validation/nulls.py`,
`validation/harness.py`, `validation/spin.py`, `configs/validation_maps.yaml`.
For each drug we predict its map from SMILES, correlate it with a real observed drug-response
map, and test significance with a **spin test** (rotating the map on the cortical sphere to
build a null that respects spatial autocorrelation). We also check whether the correlation's
**sign** matches known pharmacology, and report how many receptors contributed.

Three deliberate safeguards:
- **Strict alignment:** the observed-map loader fails loudly if a map isn't in the same
  parcellation — it refuses to correlate misaligned vectors.
- **No fake spin tests:** without real spherical permutations the code uses a naive null and
  **labels every result `spin=False`** with a warning. It will not let you report a naive
  p-value as a spin test.
- **Two tiers:** the harness separates *measured-affinity reproduction* (a sanity check using
  real assay values) from *zero-shot prediction* (the actual novel claim).

**Why it matters:** this is where projects like this live or die. Comparing brain maps with a
naive p-value is a classic, fatal statistics error — neighbouring regions are correlated, so
naive p-values are wildly over-optimistic. The spin test is the field-standard fix, and
refusing to fake it is exactly the discipline a PI or reviewer looks for. Separating the two
tiers means that if the zero-shot result is modest, you can still show the pipeline is sound —
and a modest, properly spin-tested result beats a flashy, wrong one.

### Step 9 — Scripts, HPC, and tests
**Files:** `scripts/train_affinity.py`, `scripts/predict.py`, `scripts/evaluate.py`,
`scripts/make_spins.py`, `slurm/*.sbatch`, `tests/*`.
Command-line entry points for training, single-molecule prediction, and validation — each with
a labelled DEMO fallback so they run out of the box. SLURM batch scripts for the cluster. A test
suite (currently **12 passing**) that exercises training, projection, alignment, and the spin
logic without needing rdkit, torch, or a network.

**Why it matters:** the DEMO fallbacks mean a collaborator (or an interviewer) can clone and run
`predict.py` in thirty seconds and *see* a brain map — the single best thing for getting people
to engage. The SLURM scripts show you can scale to real training. The tests mean you can refactor
fast without silently breaking the alignment that the whole result depends on.

---

## 5. The honesty stance (and why it's a competitive advantage)

Across every stage we chose the harder, more honest option: scaffold splits over random, masked
multi-task over 19 separate models, spin tests over naive p-values, labelled demo modes over
silent fakery, caveat flags baked into the data. In the README and docs we state plainly that
**receptor density ≠ drug exposure ≠ agonist/antagonist direction**, and that outputs are
prioritisation hypotheses, not clinical claims.

This isn't just good manners. In computational neuroscience and ML-for-biology, the fastest way
to lose a reviewer or a hiring manager is an inflated claim they can puncture in one question.
Demonstrating that you know your method's limits — and built guardrails so the code can't quietly
overclaim — is what signals research maturity. It's the part that makes people want to talk to you.

---

## 6. Where we are, and what's next

**Built:** the full scaffold — data fetchers, featurisation, two affinity models, projection,
and a complete validation harness — all running end-to-end on synthetic data, with 12 passing
tests. See `git log` for the per-commit history.

**Immediate critical path:** sourcing and parcellating **real** observed drug maps into
`data/validation/` (see `docs/VALIDATION.md`). The harness is ready and waiting for them; if
open maps turn out to be scarce, that finding reshapes the project's claim and we want to know
within week 1.

**Then:** real data fetch + audit → a D-MPNN model to beat the baseline → real spin nulls → the
two-tier zero-shot validation report → inverse design → an interactive demo. The full, ordered,
LLM-executable task list is in `docs/BUILD_PLAYBOOK.md`.
