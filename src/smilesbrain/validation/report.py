"""Two-tier validation report: measured-affinity reproduction vs zero-shot prediction.

Tier A -- *measured-affinity reproduction* (sanity). The drug's affinity profile is taken from
lab measurements (ChEMBL + PDSP rows in the label table), projected through the atlas and
correlated with the observed map. This is the published method; if it fails, suspect atlas /
alignment before blaming the affinity model.

Tiers A-dir / B-dir -- the same two affinity sources, but projected through **only** the
receptors whose functional direction is recorded in the catalog, with that sign applied
(pre-registered; see configs/validation_maps.yaml). These use *raw* potency as the weight
rather than the z-scored profile: z-scoring across a one- to four-receptor subset is
degenerate (a single receptor z-scores to exactly 0, giving an all-zero map), so the raw
weight follows from the restriction rather than being chosen for performance.

Tier B -- *zero-shot prediction* (the claim). The affinity profile is predicted from SMILES by a
model retrained with every molecule related to the test drug removed from training, so the model
has never seen the drug or its chemotype.

What "related" means here is deliberately stricter than the repo's scaffold split:
  * Keys are computed on the *parent* (largest) fragment, so a salted copy of the drug (the
    label table contains many, e.g. "Br.Br.<molecule>") cannot slip past the check -- the
    whole-SMILES Murcko scaffold of a salt is a different string from the free base's.
  * A training molecule is removed if it shares the drug's InChIKey connectivity layer (same
    molecule, any stereoisomer / salt form) OR its Bemis-Murcko scaffold (includeChirality=False).
  * `assert_held_out` then re-derives those keys from the SMILES actually used for training and
    raises if any overlap remains. The report prints the counts and the nearest remaining
    Tanimoto similarity, because scaffold exclusion does not remove close analogues with a
    different scaffold; that residual similarity is reported, not hidden.
"""

from __future__ import annotations

import functools
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from smilesbrain.validation.harness import evaluate_one


# --------------------------------------------------------------------------- structure keys
@dataclass(frozen=True)
class StructureKeys:
    connectivity: str   # InChIKey block 1 of the parent fragment (stereo/salt agnostic)
    stereo: str         # InChIKey blocks 1+2 (adds stereochemistry) of the parent fragment
    scaffold: str       # Bemis-Murcko scaffold SMILES of the parent fragment ("" if acyclic)
    has_defined_stereo: bool


@functools.lru_cache(maxsize=None)
def structure_keys(smiles: str) -> StructureKeys | None:
    """Keys for one molecule, computed on its largest fragment. None if it does not parse.

    Memoised: it is a pure function of the SMILES string, so caching does not change any
    result -- `assert_held_out` still derives its answer from the training SMILES it is given.
    """
    from rdkit import Chem
    from rdkit.Chem import inchi
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    parent = max(Chem.GetMolFrags(mol, asMols=True), key=lambda m: m.GetNumHeavyAtoms())
    key = inchi.MolToInchiKey(parent)
    if not key:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=parent, includeChirality=False)
    centers = Chem.FindMolChiralCenters(parent, includeUnassigned=False, useLegacyImplementation=False)
    return StructureKeys(
        connectivity=key[:14], stereo=key[:25], scaffold=scaffold, has_defined_stereo=bool(centers)
    )


def label_keys(smiles_list: list[str]) -> list[StructureKeys | None]:
    return [structure_keys(s) for s in smiles_list]


def _related(a: StructureKeys, b: StructureKeys) -> tuple[bool, bool]:
    same_molecule = a.connectivity == b.connectivity
    same_scaffold = bool(a.scaffold) and a.scaffold == b.scaffold
    return same_molecule, same_scaffold


def held_out_mask(train_keys: list[StructureKeys | None], drug: StructureKeys) -> tuple[np.ndarray, int, int]:
    """Boolean mask of training rows to KEEP, plus counts removed as (same molecule, scaffold-only)."""
    keep = np.ones(len(train_keys), dtype=bool)
    n_mol = n_scaf = 0
    for i, k in enumerate(train_keys):
        if k is None:
            continue
        same_mol, same_scaf = _related(k, drug)
        if same_mol:
            keep[i] = False
            n_mol += 1
        elif same_scaf:
            keep[i] = False
            n_scaf += 1
    return keep, n_mol, n_scaf


def assert_held_out(train_smiles: list[str], drug_smiles: str) -> None:
    """Raise AssertionError if any training molecule is the drug or shares its scaffold.

    Recomputes everything from the SMILES passed in (not from any precomputed mask), so it
    checks what the model was actually fitted on.
    """
    drug = structure_keys(drug_smiles)
    assert drug is not None, f"drug SMILES does not parse: {drug_smiles!r}"
    offenders = []
    for smi in train_smiles:
        k = structure_keys(smi)
        if k is not None and any(_related(k, drug)):
            offenders.append(smi)
    assert not offenders, (
        f"zero-shot leak: {len(offenders)} training molecule(s) share the drug's molecule "
        f"or scaffold, e.g. {offenders[:3]}"
    )


# ---------------------------------------------------------------------- measured affinity
def measured_affinity(
    labels: pd.DataFrame, keys: list[StructureKeys | None], drug_smiles: str
) -> tuple[dict[str, float], int]:
    """Median measured pChEMBL per receptor over all label rows that are this drug.

    Matching uses the stereo-aware key when the catalog SMILES defines stereochemistry, else
    the connectivity key, so e.g. racemic methylphenidate matches any stereoisomer but LSD
    does not match its diastereomer. Returns (profile, n_matching_rows).
    """
    drug = structure_keys(drug_smiles)
    if drug is None:
        raise ValueError(f"cannot parse {drug_smiles!r}")
    field_ = "stereo" if drug.has_defined_stereo else "connectivity"
    rows = [i for i, k in enumerate(keys) if k is not None and getattr(k, field_) == getattr(drug, field_)]
    if not rows:
        return {}, 0
    sub = labels.iloc[rows]
    med = sub.median(axis=0, skipna=True)
    return {c: float(v) for c, v in med.items() if np.isfinite(v)}, len(rows)


# ------------------------------------------------------------------------------ statistics
def benjamini_hochberg(p: list[float]) -> list[float]:
    p = np.asarray(p, dtype=float)
    n = p.size
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out.tolist()


# ---------------------------------------------------------------------------- evaluation
@dataclass
class TierRow:
    tier: str                      # "A" | "B" | "A-dir" | "B-dir"
    drug: str
    measure: str
    n_subjects: int | None
    n_receptors_used: int
    r: float
    p: float
    spin: bool
    n_perm: int
    expected_sign: int | None
    sign_ok: bool | None
    # tier B diagnostics (None for tier A)
    n_train: int | None = None
    n_removed_molecule: int | None = None
    n_removed_scaffold: int | None = None
    max_tanimoto: float | None = None
    scaffold: str | None = None
    # tier A diagnostics
    n_label_rows: int | None = None
    directed: dict[str, int] | None = None
    p_bh: float = float("nan")
    scores: np.ndarray = field(default=None, repr=False)
    observed: np.ndarray = field(default=None, repr=False)


def tier_rows_to_frame(rows: list[TierRow]) -> pd.DataFrame:
    """Scalar columns of every TierRow (no per-region arrays), for scripts/make_figures.py."""
    records = []
    for r in rows:
        rec = {k: v for k, v in vars(r).items() if k not in ("scores", "observed")}
        rec["directed"] = (
            ";".join(f"{k}={v:+d}" for k, v in sorted(r.directed.items())) if r.directed else ""
        )
        records.append(rec)
    return pd.DataFrame(records)


def ridge_refit(alpha: float, receptor_keys: list[str]):
    """Default zero-shot backend: refit the ridge baseline on the filtered training set."""
    from smilesbrain.models.baseline import MaskedMultiTaskRidge

    def refit(X, Y, smiles):
        model = MaskedMultiTaskRidge(receptor_keys=receptor_keys, alpha=alpha).fit(X, Y)
        return lambda smi, x: model.predict(x[None, :])[0]

    return refit


def gbdt_refit(receptor_keys: list[str], **kwargs):
    from smilesbrain.models.trees import MaskedMultiTaskGBDT

    def refit(X, Y, smiles):
        model = MaskedMultiTaskGBDT(receptor_keys=receptor_keys, **kwargs).fit(X, Y)
        return lambda smi, x: model.predict(x[None, :])[0]

    return refit


def dmpnn_refit(receptor_keys: list[str], **kwargs):
    """Zero-shot backend that retrains a D-MPNN per drug. Slow (minutes to hours per drug)."""
    from smilesbrain.models.dmpnn import MaskedDMPNNRegressor

    def refit(X, Y, smiles):
        model = MaskedDMPNNRegressor(receptor_keys=receptor_keys, **kwargs).fit(smiles, Y)
        return lambda smi, x: model.predict([smi])[0]

    return refit


def two_tier_evaluation(
    catalog,
    observed: dict[str, np.ndarray],
    atlas: pd.DataFrame,
    labels: pd.DataFrame,
    fp_cfg,
    alpha: float,
    receptor_keys: list[str],
    perm_indices: np.ndarray | None = None,
    weight_mode: str = "zscore",
    n_subjects: dict[str, int | None] | None = None,
    refit=None,
) -> list[TierRow]:
    """Run tier A and tier B for every catalog drug that has an observed map.

    `labels`: (molecule x receptor) pChEMBL table indexed by SMILES.
    `alpha`: ridge strength carried over from the validated baseline (not re-tuned per drug).
    `refit`: zero-shot backend, `fn(X, Y, smiles) -> fn(smiles, fingerprint) -> vector`.
             Defaults to the ridge baseline. Whatever it is, it is refit per drug on the
             chemotype-filtered training set, so the held-out guarantee is unchanged.
    """
    refit = refit or ridge_refit(alpha, receptor_keys)
    from smilesbrain.features import featurize_many, morgan_fingerprint
    from smilesbrain.models.uncertainty import tanimoto_to_training_set
    from smilesbrain.projection import project_affinity

    n_subjects = n_subjects or {}
    entries = [e for e in catalog if e.drug in observed]
    if not entries:
        return []

    smiles_all = labels.index.tolist()
    keys_all = label_keys(smiles_all)
    X_all, kept = featurize_many(smiles_all, fp_cfg)
    Y_all = labels[receptor_keys].to_numpy(dtype=float)[kept]
    keys_kept = [keys_all[i] for i in kept]
    smiles_kept = [smiles_all[i] for i in kept]
    labels_kept = labels.iloc[kept]

    rows: list[TierRow] = []
    for entry in entries:
        obs = observed[entry.drug]
        drug_keys = structure_keys(entry.smiles)
        if drug_keys is None:
            raise ValueError(f"{entry.drug}: catalog SMILES does not parse")
        exp = entry.expected_sign_value

        def _row(tier, profile, direction, **extra):
            """One evaluated row; direction-restricted tiers use raw potency weights."""
            mode = "raw" if direction else weight_mode
            if direction:
                profile = {k: v for k, v in profile.items() if k in direction}
                if not profile:
                    return None
            proj = project_affinity(profile, atlas, weight_mode=mode, direction=direction)
            res = evaluate_one(entry.drug, profile, obs, atlas, perm_indices, mode, exp,
                               direction=direction)
            return TierRow(tier, entry.drug, entry.measure, n_subjects.get(entry.drug),
                           res.n_receptors_used, res.r, res.p, res.spin, res.n_perm, exp,
                           res.sign_ok, directed=direction, scores=proj.region_scores,
                           observed=obs, **extra)

        # ---- Tier A: measured affinity
        profile, n_rows = measured_affinity(labels_kept, keys_kept, entry.smiles)
        if len(profile) >= 3:
            rows.append(_row("A", profile, None, n_label_rows=n_rows))
        if entry.direction and profile:
            r_ = _row("A-dir", profile, entry.direction, n_label_rows=n_rows)
            if r_ is not None:
                rows.append(r_)

        # ---- Tier B: zero-shot prediction with the drug's chemotype removed from training
        keep, n_mol, n_scaf = held_out_mask(keys_kept, drug_keys)
        train_smiles = [s for s, k in zip(smiles_kept, keep) if k]
        assert_held_out(train_smiles, entry.smiles)               # provable, not assumed
        predict_fn = refit(X_all[keep], Y_all[keep], train_smiles)
        x = morgan_fingerprint(entry.smiles, fp_cfg)
        pred = np.asarray(predict_fn(entry.smiles, x), dtype=float)
        affinity = {k: float(v) for k, v in zip(receptor_keys, pred)}
        diag = dict(n_train=int(keep.sum()), n_removed_molecule=n_mol,
                    n_removed_scaffold=n_scaf,
                    max_tanimoto=float(tanimoto_to_training_set(x, X_all[keep])),
                    scaffold=drug_keys.scaffold)
        rows.append(_row("B", affinity, None, **diag))
        if entry.direction:
            r_ = _row("B-dir", affinity, entry.direction, **diag)
            if r_ is not None:
                rows.append(r_)

    for tier in ("A", "B", "A-dir", "B-dir"):
        idx = [i for i, r in enumerate(rows) if r.tier == tier]
        for i, q in zip(idx, benjamini_hochberg([rows[i].p for i in idx])):
            rows[i].p_bh = q
    return rows


# -------------------------------------------------------------------------------- report
def _git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def _fmt_sign(row: TierRow) -> str:
    if row.sign_ok is None:
        return "n/a"
    return "yes" if row.sign_ok else "no"


def _table(rows: list[TierRow], tier: str) -> list[str]:
    sel = [r for r in rows if r.tier == tier]
    if not sel:
        return ["_No drugs in this tier (no observed map / no measured profile)._", ""]
    def null(r):
        return f"spin ({r.n_perm})" if r.spin else f"NAIVE permutation ({r.n_perm}) - not a spin test"

    directed = tier.endswith("-dir")
    extra_h = " receptors (signed) |" if directed else " receptors used |"
    head = ["| drug | measure | n subj |" + extra_h + " r | spin p | BH p | null | expected sign | sign agrees |",
            "|---|---|---:|---|---:|---:|---:|---|---|---|" if directed
            else "|---|---|---:|---:|---:|---:|---:|---|---|---|"]
    body = []
    for r in sel:
        exp = {1: "+", -1: "-", None: "-"}[r.expected_sign] if r.expected_sign else "none"
        if directed:
            cell = ", ".join(f"{k}{'+' if v > 0 else '-'}" for k, v in (r.directed or {}).items())
        else:
            cell = str(r.n_receptors_used)
        body.append(
            f"| {r.drug} | {r.measure} | {r.n_subjects if r.n_subjects is not None else '?'} | "
            f"{cell} | {r.r:+.3f} | {r.p:.3f} | {r.p_bh:.3f} | {null(r)} | {exp} | {_fmt_sign(r)} |"
        )
    agree = [r.sign_ok for r in sel if r.sign_ok is not None]
    sig = sum(r.p < 0.05 for r in sel)
    foot = (f"\n{sig}/{len(sel)} with p < 0.05 (uncorrected); "
            f"{sum(r.p_bh < 0.05 for r in sel)}/{len(sel)} after Benjamini-Hochberg across this tier; "
            f"sign agreement {sum(agree)}/{len(agree)}.")
    return head + body + [foot, ""]


def build_diagnostics(catalog, observed, atlas, labels, keys_kept, labels_kept,
                      perm_indices=None, weight_mode="zscore", n_perm=5000) -> dict:
    """Run the three localising diagnostics for every drug with an observed map."""
    from smilesbrain.validation.diagnostics import (
        architecture_ceiling,
        atlas_collinearity,
        receptor_permutation_null,
        target_receptor_ranking,
    )

    out = {"collinearity": atlas_collinearity(atlas), "rows": []}
    for entry in catalog:
        if entry.drug not in observed:
            continue
        obs = observed[entry.drug]
        profile, _ = measured_affinity(labels_kept, keys_kept, entry.smiles)
        out["rows"].append({
            "drug": entry.drug,
            "target": target_receptor_ranking(entry.drug, obs, atlas, entry.direction,
                                              perm_indices),
            "ceiling": architecture_ceiling(entry.drug, obs, atlas, perm_indices),
            "perm": receptor_permutation_null(entry.drug, profile, obs, atlas,
                                              n_perm=n_perm, weight_mode=weight_mode),
        })
    return out


def _diagnostics_section(diag: dict) -> list[str]:
    if not diag or not diag["rows"]:
        return []
    c = diag["collinearity"]
    L = ["## Diagnostics - where the pipeline succeeds and where it fails", "",
         "A spin test says whether a prediction matched; it does not say which step is at "
         "fault when it did not. These three localise it.", "",
         "### 1. Does the observed map resemble the drug's own target receptor?", "",
         "Correlation of each observed map with each single receptor density map, before any "
         "model. `rank` is where the drug's best pre-registered target sits among all "
         f"{c['n_pairs'] and len(diag['rows'][0]['target'].ranking)} receptors, by |r|.", "",
         "| drug | best target | r | rank | spin p | strongest receptor overall |",
         "|---|---|---:|---:|---:|---|"]
    for row in diag["rows"]:
        t = row["target"]
        top, top_r = t.ranking[0]
        L.append(f"| {t.drug} | {t.best_target or 'n/a'} | {t.r:+.3f} | "
                 f"{t.rank if t.rank else 'n/a'}/{t.n_receptors} | {t.p:.3f} | {top} ({top_r:+.2f}) |")
    L += ["", "### 2. Ceiling: the best any weighting of this atlas could do", "",
          "Oracle, not a result. The weights are fit to the observed map itself (ridge, "
          "collinearity-aware) and scored on held-out regions. This is what a perfect affinity "
          "vector would be competing against. `weight stability` is the mean correlation "
          "between bootstrap weight vectors.", "",
          "| drug | in-sample r (overfit) | cross-validated r | spin p | weight stability |",
          "|---|---:|---:|---:|---:|"]
    for row in diag["rows"]:
        k = row["ceiling"]
        L.append(f"| {k.drug} | {k.r_in_sample:.3f} | {k.r_cv:.3f} | {k.p_cv:.3f} | "
                 f"{k.weight_stability:.2f} |")
    med = np.median([r["ceiling"].r_cv for r in diag["rows"]])
    sig = sum(r["ceiling"].p_cv < 0.05 for r in diag["rows"])
    L += ["", f"Median cross-validated ceiling r = **{med:.3f}**, spin-significant for "
          f"{sig}/{len(diag['rows'])} drugs.", ""]
    L += ["### 3. Does it matter which receptor holds which affinity?", "",
          "The decisive test of the claim. Keeps the drug's measured affinity *values* and the "
          "real atlas, and shuffles only which receptor each value belongs to. Immune to the "
          "collinearity that makes fitted weights uninterpretable, because it never reads a "
          "coefficient.", "",
          r"| drug | receptors | true r | mean \|r\| of shuffled assignments | p |",
          "|---|---:|---:|---:|---:|"]
    n_beat = n_tested = 0
    for row in diag["rows"]:
        pn = row["perm"]
        if pn is None:
            L.append(f"| {row['drug']} | <3 measured | - | - | - |")
            continue
        n_tested += 1
        n_beat += pn.p < 0.05
        L.append(f"| {pn.drug} | {pn.n_receptors} | {pn.r:+.3f} | {pn.null_mean_abs_r:.3f} | "
                 f"{pn.p:.3f} |")
    L += ["", f"**{n_beat}/{n_tested} drugs beat the receptor-permutation null at p < 0.05.**", "",
          f"Why the fitted weights cannot simply be read off instead: the {len(diag['rows'][0]['target'].ranking)} "
          f"density maps are collinear (mean |r| = {c['mean_abs_r']:.2f}; "
          f"{c['pairs_above_0.5']}/{c['n_pairs']} pairs above 0.5; first principal component "
          f"{c['pc1_variance']:.0%} of variance). Many different weight vectors fit about "
          "equally well, which is why the ceiling in (2) is high while the bootstrap weight "
          "stability beside it is low.", ""]
    return L


def build_markdown(rows: list[TierRow], meta: dict, diag: dict | None = None) -> str:
    L: list[str] = []
    L += ["# SMILES-Brain validation report", "",
          f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · commit `{_git_commit()}` · "
          f"parcellation `{meta['parcellation']}` ({meta['n_regions']} regions) · weights `{meta['weight_mode']}` · "
          f"primary projection **magnitude-only** (no agonist/antagonist direction) · seed {meta['seed']}", ""]
    L += ["Each row correlates a drug's **predicted** map (affinity vector x receptor-density atlas) with its "
          "**observed** drug-minus-placebo CBF map across regions. p-values come from the null named in each "
          "row. Tier A and Tier B differ only in where the affinity vector comes from.", ""]

    L += ["## Tier A - measured-affinity reproduction (sanity check)", "",
          "Affinity vector = median of lab-measured pChEMBL (ChEMBL + PDSP) for the drug. This is the published "
          "method and should work before the zero-shot claim is trusted. Unmeasured receptors are dropped, "
          "not set to zero.", ""]
    L += _table(rows, "A")

    L += ["## Tier B - zero-shot prediction (the claim)", "",
          "Affinity vector = predicted from SMILES by a ridge model refit with the drug's molecule (any salt / "
          "stereoisomer) **and its whole Bemis-Murcko scaffold removed from training**. Checked by "
          "`assert_held_out` on the exact training SMILES; a leak would abort the run.", ""]
    L += _table(rows, "B")

    if any(r.tier.endswith("-dir") for r in rows):
        L += ["## Pre-specified secondary analysis - direction-aware projection", "",
              "The same two affinity sources, but projected through **only** the receptors whose "
              "agonist/antagonist direction is recorded in `configs/validation_maps.yaml`, with that "
              "sign applied and raw potency as the weight. Those directions were written from "
              "pharmacology references and committed before this analysis was run (see that file's "
              "git history). The magnitude-only tiers above were already known at that point, but no "
              "direction was chosen by looking at a correlation.", "",
              "Raw potency is used instead of the z-scored profile because z-scoring across a one- "
              "to four-receptor subset is degenerate: a single receptor z-scores to exactly 0, which "
              "would give an all-zero map. That follows from the restriction, not from tuning.", "",
              "### Tier A-dir - measured affinity, direction-aware", ""]
        L += _table(rows, "A-dir")
        L += ["### Tier B-dir - zero-shot, direction-aware", ""]
        L += _table(rows, "B-dir")

    b = [r for r in rows if r.tier == "B"]
    if b:
        L += ["### Zero-shot guarantee", "",
              "| drug | training molecules | removed: same molecule | removed: same scaffold only | "
              "nearest remaining Tanimoto | murcko scaffold (parent) |", "|---|---:|---:|---:|---:|---|"]
        for r in b:
            L.append(f"| {r.drug} | {r.n_train} | {r.n_removed_molecule} | {r.n_removed_scaffold} | "
                     f"{r.max_tanimoto:.2f} | `{r.scaffold or '(acyclic)'}` |")
        L += ["", "The nearest-Tanimoto column is the honest caveat: removing a scaffold does not remove close "
              "analogues with a different scaffold, so \"zero-shot\" here means *held-out chemotype*, "
              "not *no similar training molecules*.", ""]

    if diag:
        L += _diagnostics_section(diag)

    L += ["## Limitations", "",
          f"- **n = {len({r.drug for r in rows})} drugs.** Per-drug p-values are the evidence; pooled claims are not "
          "supported by this few. Benjamini-Hochberg is applied within each tier only.",
          "- Observed maps are drug-induced **CBF** changes from two studies (Dukart 2018: 4 drugs; "
          "Carhart-Harris 2016: LSD, a T-map not an effect size), group-level, different n. CBF is an indirect "
          "readout of neural activity, not receptor occupancy.",
          "- The projection is **magnitude-only**: it does not know agonist from antagonist, so a sign-correct "
          "result for an antagonist is partly luck. Directions can be supplied (`--direction`), not predicted.",
          "- Receptor density says where a drug *can* act, not how much reaches the brain.",
          "- A spin test controls for spatial autocorrelation of the cortical maps only; subcortex is not "
          "in this parcellation.",
          "- Expected signs were fixed in `configs/validation_maps.yaml` before these results were computed.", ""]
    return "\n".join(L)


def plot_report(rows: list[TierRow], path: str | Path) -> Path:
    """Top: spin-tested r per drug and tier. Bottom: region-wise predicted vs observed, per drug."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # reference palette slots 1-2 (blue, orange); validated for CVD + contrast, light surface.
    COL = {"A": "#2a78d6", "B": "#eb6834"}
    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    NAME = {"A": "Tier A: measured affinity", "B": "Tier B: zero-shot (chemotype held out)"}

    drugs = list(dict.fromkeys(r.drug for r in rows))
    n = max(len(drugs), 1)
    panels = [("magnitude-only (primary)", ("A", "B"))]
    if any(r.tier.endswith("-dir") for r in rows):
        panels.append(("direction-aware (pre-registered)", ("A-dir", "B-dir")))

    fig = plt.figure(figsize=(min(20.0, max(11.0, 1.05 * n * len(panels))), 8.2), facecolor=SURF)
    gs = fig.add_gridspec(2, max(n, len(panels)), height_ratios=[1.3, 1], hspace=0.6, wspace=0.42)
    top = gs[0, :].subgridspec(1, len(panels), wspace=0.18)

    width = 0.36
    axes = []
    for pi, (ptitle, tiers) in enumerate(panels):
        ax = fig.add_subplot(top[0, pi], facecolor=SURF)
        axes.append(ax)
        for j, tier in enumerate(tiers):
            for i, d in enumerate(drugs):
                r = next((x for x in rows if x.tier == tier and x.drug == d), None)
                if r is None:
                    continue
                base = tier[0]
                sig = r.p < 0.05
                ax.bar(i + (j - 0.5) * (width + 0.04), r.r, width,
                       color=COL[base] if sig else SURF, edgecolor=COL[base], linewidth=1.6, zorder=3)
                if sig:
                    ax.text(i + (j - 0.5) * (width + 0.04), r.r + (0.03 if r.r >= 0 else -0.03), "*",
                            ha="center", va="bottom" if r.r >= 0 else "top", color=INK, fontsize=13)
        ax.axhline(0, color=INK2, linewidth=0.8, zorder=2)
        ax.margins(y=0.18)
        exp = {d: next((x.expected_sign for x in rows if x.drug == d), None) for d in drugs}
        ax.set_xticks(range(len(drugs)))
        ax.set_xticklabels([f"{d}\n({'+' if exp[d] == 1 else '-' if exp[d] == -1 else '?'})"
                            for d in drugs], color=INK, fontsize=8.5, rotation=30, ha="right")
        if pi == 0:
            ax.set_ylabel("spatial correlation r (predicted vs observed)", color=INK2, fontsize=9)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, length=0, labelsize=8)
        ax.set_title(ptitle, color=INK, fontsize=10, loc="left", pad=6)
    lo = min(a.get_ylim()[0] for a in axes)
    hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(lo, hi)                      # one shared scale: panels are comparable
    for a in axes[1:]:
        a.set_yticklabels([])

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COL[t], edgecolor=COL[t]) for t in "AB"]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor=INK2, edgecolor=INK2),
                plt.Rectangle((0, 0), 1, 1, facecolor=SURF, edgecolor=INK2, linewidth=1.6)]
    axes[0].legend(handles, [NAME["A"], NAME["B"], "spin p < 0.05 (filled, *)", "not significant (outline)"],
                   frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.10), ncol=2, fontsize=8.5,
                   labelcolor=INK2)
    null_txt = "spin null" if all(r.spin for r in rows) else "NAIVE null (not a spin test)"
    fig.suptitle(f"Per-drug agreement with observed CBF maps · {null_txt} · x-label shows the "
                 f"expected sign", color=INK, fontsize=11, x=0.01, ha="left", y=0.995)

    for i, d in enumerate(drugs):
        sub = fig.add_subplot(gs[1, i], facecolor=SURF)
        for tier in ("A", "B"):
            r = next((x for x in rows if x.tier == tier and x.drug == d), None)
            if r is None:
                continue
            z = (r.scores - np.nanmean(r.scores)) / (np.nanstd(r.scores) + 1e-12)
            sub.scatter(z, r.observed, s=14, color=COL[tier], alpha=0.65, edgecolor=SURF,
                        linewidth=0.5, zorder=3)
        sub.set_title(d, color=INK, fontsize=10)
        sub.set_xlabel("predicted (z)", color=INK2, fontsize=8)
        if i == 0:
            sub.set_ylabel("observed (map units)", color=INK2, fontsize=8)
        sub.grid(True, color=GRID, linewidth=0.6)
        for s in ("top", "right"):
            sub.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            sub.spines[s].set_color(GRID)
        sub.tick_params(colors=INK2, labelsize=6.5, length=0)
    fig.text(0.01, 0.005, "Each dot is one of the parcellation's regions. Observed units differ per drug "
             "(effect size vs T-map); axes are free per panel.", color=INK2, fontsize=8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    return path
