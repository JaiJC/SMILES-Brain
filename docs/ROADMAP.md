# Roadmap — 2-week sprint

Goal: a working, honest end-to-end pipeline + a shareable demo, with scaffold-split
generalisation numbers you can defend in an interview.

## Done (scaffold shipped)
- Receptor config (single source of truth, canonical ordering).
- SMILES → Morgan fingerprint featurisation (RDKit, optional-dep guarded).
- Sparse-label-aware multi-task baseline (Ridge) + masked multi-task MLP (PyTorch/GPU).
- Projection: affinity vector × Hansen atlas → region effect map.
- Validation: map correlation + spin-test scaffold (with a loud warning when spins absent).
- Data fetchers: ChEMBL affinities, Hansen atlas; wide-table assembly.
- Scaffold/random splits; synthetic generators so everything runs offline.
- Tests (no heavy deps) + SLURM scripts.

## Week 1
- [ ] Run `fetch_data` on HPC login node; sanity-check label density per receptor.
- [ ] Confirm atlas filename→receptor mapping with `fetch_atlas --list`; fix `KEY_ALIASES`.
- [ ] Train baseline; record per-receptor scaffold-split R². Flag `sparse`/`complex` receptors.
- [ ] Train D-MPNN (chemprop) and compare to baseline on the same split.
- [ ] Reproduce the *measured-affinity* projection for 3–4 known drugs vs. published maps.

## Week 2
- [ ] Assemble a validation set of open drug-response maps (OpenNeuro / parcellated).
  **This is the critical-path risk — do it first, in parallel with Week 1.**
- [ ] Proper spin test via `neuromaps` spherical permutations.
- [ ] Zero-shot eval: hold out whole drug classes; predict → project → correlate with real maps.
- [x] Inverse design mode: library screen + mutation-based generator (unvalidated; see REPORT).
- [x] Web demo (type SMILES → 3D brain) + write-up (`docs/REPORT.md`). 2-min video: not done.
- [ ] Outreach: psychedelic/CNS drug-discovery teams; Hansen & Luppi labs.

## Known risks (state these openly)
- Density ≠ drug exposure ≠ agonist/antagonist direction. Correlational validation only.
- Some receptors (VAChT, GABA-A subunit, NMDA) have sparse or proxy ligand data.
- Spin test must use real spherical nulls before any p-value is reported.
