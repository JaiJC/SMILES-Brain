#!/usr/bin/env python
"""Run the pre-registered learned-gains experiment on real data (leave-one-drug-out).

    python scripts/learned_gains.py --spins data/atlas/spins_scale100.npy

Needs the same real inputs as `scripts/evaluate.py --report` (trained affinity artifact, real
atlas, label table, observed drug maps) and `docs/PREREGISTRATION.md`, which must exist and
should be committed *before* this is run: its SHA-256 is recorded in the output so the order
can be audited. Refuses DEMO MODE. Writes artifacts/learned_gains.csv and
artifacts/learned_gains_report.md.

Reading order: Tier A (measured affinity) is the gate. Tier B (zero-shot) is only interpreted
if Tier A meets the criterion, because a structure-predicted profile cannot do better than the
measured one it approximates. Concentration and group variants are sensitivity analyses.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import joblib
import pandas as pd

from smilesbrain.config import load_receptors
from smilesbrain.utils import set_global_seed, write_run_manifest
from smilesbrain.validation import load_catalog, load_observed_map
from smilesbrain.validation.lodo import PREREG, lodo_evaluate, rows_to_frame, summarise_lodo
from smilesbrain.validation.nulls import load_spins
from smilesbrain.validation.profiles import measured_profiles, zero_shot_profiles
from smilesbrain.validation.report import ridge_refit

# Fixed in docs/PREREGISTRATION.md: drugs that share pharmacology are never in each other's
# training set for the leave-class-out sensitivity analysis.
CLASS_GROUPS = {"haloperidol": "antipsychotic", "risperidone": "antipsychotic",
                "olanzapine": "antipsychotic"}
SENSITIVITY_CONC_NM = (10.0, 1000.0)


def _fmt(x: float) -> str:
    return "n/a" if x != x else f"{x:+.3f}"


def _table(rows) -> list[str]:
    lines = ["| drug | receptors | train drugs | r model | r mean-map | delta | shuffled mean r | "
             "p (receptor shuffle) | spin p |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        sp = f"{r.p_spin:.3f}" if r.spin else "n/a (no spin nulls)"
        lines.append(f"| {r.drug} | {r.n_receptors} | {r.n_train} | {_fmt(r.r_model)} | "
                     f"{_fmt(r.r_mean_map)} | {_fmt(r.delta_vs_mean_map)} | {_fmt(r.r_null_mean)} | "
                     f"{r.p_perm:.3f} | {sp} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="artifacts/affinity_baseline.joblib", type=Path)
    ap.add_argument("--atlas", default=None, type=Path)
    ap.add_argument("--spins", default=None, type=Path)
    ap.add_argument("--labels", default="data/labels.parquet", type=Path)
    ap.add_argument("--parcellation", default=None)
    ap.add_argument("--prereg", default="docs/PREREGISTRATION.md", type=Path)
    ap.add_argument("--n-perm", default=5000, type=int)
    ap.add_argument("--out", default="artifacts/learned_gains.csv", type=Path)
    ap.add_argument("--seed", default=0, type=int)
    args = ap.parse_args()
    set_global_seed(args.seed)

    if not args.prereg.exists():
        raise SystemExit(f"{args.prereg} not found. The success criterion must be written down, "
                         "and committed, before this analysis is run.")
    prereg_hash = hashlib.sha256(args.prereg.read_bytes()).hexdigest()

    rc = load_receptors(parcellation=args.parcellation)
    atlas_path = args.atlas or Path(rc.parcellation.atlas_file)
    for needed in (args.artifact, atlas_path, args.labels):
        if not needed.exists():
            raise SystemExit(f"{needed} not found; refusing DEMO MODE for a scientific test.")
    atlas = pd.read_csv(atlas_path, index_col=0)
    if list(atlas.columns) != rc.atlas_keys or len(atlas) != rc.n_regions:
        raise SystemExit("atlas does not match the configured parcellation and receptor order")
    catalog = load_catalog()
    observed = {e.drug: load_observed_map(e.file, rc.n_regions)
                for e in catalog if Path(e.file).exists()}
    if not observed:
        raise SystemExit("no observed drug maps found (see docs/VALIDATION.md)")
    perms = load_spins(args.spins, rc.n_regions) if args.spins and args.spins.exists() else None
    labels = (pd.read_parquet(args.labels) if args.labels.suffix == ".parquet"
              else pd.read_csv(args.labels, index_col=0))
    bundle = joblib.load(args.artifact)
    alpha = getattr(bundle["model"], "alpha", 1.0)

    catalog = [e for e in catalog if e.drug in observed]
    measured, n_rows = measured_profiles(catalog, labels, rc.atlas_keys)
    zero_shot, zs_diag = zero_shot_profiles(catalog, labels, bundle["fp"], rc.atlas_keys,
                                            ridge_refit(alpha, rc.atlas_keys))

    runs = {}   # (tier, conc, grouping) -> (rows, skipped)
    for conc in (PREREG["primary_conc_nM"], *SENSITIVITY_CONC_NM):
        for grouping, groups in (("leave-one-drug-out", None), ("leave-class-out", CLASS_GROUPS)):
            for tier, test in (("A", measured), ("B", zero_shot)):
                runs[(tier, conc, grouping)] = lodo_evaluate(
                    measured, test, observed, atlas, conc_nM=conc, tier=tier, groups=groups,
                    n_perm=args.n_perm, perm_indices=perms, seed=args.seed)

    frames = [rows_to_frame(rows).assign(grouping=g) for (t, c, g), (rows, _) in runs.items()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(args.out, index=False)

    primary = PREREG["primary_conc_nM"]
    sa = summarise_lodo(runs[("A", primary, "leave-one-drug-out")][0])
    sb = summarise_lodo(runs[("B", primary, "leave-one-drug-out")][0])
    md = ["# Learned receptor gains: pre-registered leave-one-drug-out test", "",
          f"Pre-registration: `{args.prereg}` (SHA-256 `{prereg_hash}`). "
          f"Parcellation `{rc.parcellation.key}`, {rc.n_regions} regions, "
          f"{args.n_perm} receptor shuffles, "
          + (f"{perms.shape[1]} spin rotations." if perms is not None
             else "**no spin nulls supplied; spin p-values are not reported.**"), "",
          f"## Primary endpoint (concentration {primary:g} nM, leave-one-drug-out)", "",
          f"**Tier A (measured affinity, the gate): {sa['verdict']}** - "
          f"{sa['n_beating_null']}/{sa['n_drugs']} drugs beat the receptor-shuffle null "
          f"(needed {sa['needed']}); median r {_fmt(sa['median_r_model'])}; median gain over the "
          f"mean-map baseline {_fmt(sa['median_delta_vs_mean_map'])}.", "",
          *_table(runs[("A", primary, "leave-one-drug-out")][0]), "",
          f"**Tier B (zero-shot): {sb['verdict']}**"
          + ("" if sa["verdict"] == "CRITERION MET"
             else " - *not to be interpreted: Tier A did not meet the criterion.*"), "",
          *_table(runs[("B", primary, "leave-one-drug-out")][0]), ""]
    for (tier, conc, grouping), (rows, skipped) in runs.items():
        if (conc, grouping) == (primary, "leave-one-drug-out"):
            continue
        s = summarise_lodo(rows)
        md.append(f"- Sensitivity, tier {tier}, {conc:g} nM, {grouping}: {s['verdict']} "
                  f"({s['n_beating_null']}/{s['n_drugs']} beat null, median r "
                  f"{_fmt(s['median_r_model'])}).")
    skipped_all = {f"{t}/{c:g}/{g}": sk for (t, c, g), (_, sk) in runs.items() if sk}
    if skipped_all:
        md += ["", "Skipped (not silently dropped): " + "; ".join(
            f"{k}: {v}" for k, v in skipped_all.items())]
    md += ["", "## Tier B held-out diagnostics", "",
           *[f"- {d}: {v['n_train']} training molecules, removed {v['n_removed_molecule']} same "
             f"molecule and {v['n_removed_scaffold']} same scaffold; nearest remaining Tanimoto "
             f"{v['max_tanimoto']:.2f}" for d, v in sorted(zs_diag.items())], ""]
    report = args.out.with_name("learned_gains_report.md")
    report.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"table -> {args.out}\nreport -> {report}")
    manifest = write_run_manifest(
        report, config={"prereg_sha256": prereg_hash, "primary_conc_nM": primary,
                        "n_perm": args.n_perm, "alpha": alpha, "prereg": PREREG,
                        "spins": None if perms is None else str(args.spins)},
        seed=args.seed, input_paths=[args.prereg, args.artifact, atlas_path, args.labels]
        + [e.file for e in catalog])
    print(f"manifest -> {manifest}")


if __name__ == "__main__":
    main()
