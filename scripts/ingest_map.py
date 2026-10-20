#!/usr/bin/env python
"""Parcellate a volumetric drug-response map into the CSV a validation-catalog entry expects.

    python scripts/ingest_map.py --drug ketamine --nifti ketamine_group_tmap.nii.gz \
        --source "Author et al. YEAR, doi:..." --n-subjects 24 --contrast "ketamine > placebo"

Writes the catalog entry's `file` (configs/validation_maps.yaml) as one value per region in the
selected parcellation (Schaefer only), plus `<file>.provenance.json` recording source, n,
contrast, input hash and how many regions had no data. The CSV is round-trip checked with the
harness's own `load_observed_map`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from smilesbrain.config import load_receptors
from smilesbrain.data.parcellate import (
    fetch_schaefer_labels_img,
    parcellate_nifti,
    write_observed_map,
    write_provenance,
)
from smilesbrain.validation import load_catalog


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug", required=True, help="drug name in configs/validation_maps.yaml")
    ap.add_argument("--nifti", required=True, type=Path)
    ap.add_argument("--source", required=True, help="citation / URL / dataset id")
    ap.add_argument("--n-subjects", type=int, default=None)
    ap.add_argument("--contrast", required=True, help="what the map contrasts, e.g. drug > placebo")
    ap.add_argument("--parcellation", default=None)
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--out", type=Path, default=None, help="override the catalog entry's file")
    ap.add_argument("--zero-as-missing", action="store_true",
                    help="treat exact-0 voxels as outside the analysis mask (not data)")
    ap.add_argument("--note", default=None, help="free-text note stored in the provenance")
    ap.add_argument("--labels", type=Path, default=None,
                    help="label NIfTI (default: Schaefer-2018 7-network 2mm from nilearn)")
    ap.add_argument("--data-dir", default=None, help="nilearn cache dir")
    args = ap.parse_args()

    rc = load_receptors(parcellation=args.parcellation)
    parc = rc.parcellation
    if parc.subcortex or not parc.name.startswith("Schaefer"):
        raise SystemExit(f"{parc.key}: ingestion supports Schaefer parcellations only.")

    entries = {e.drug: e for e in load_catalog(args.catalog)}
    if args.drug not in entries:
        raise SystemExit(f"{args.drug!r} not in catalog; known: {sorted(entries)}")
    out = args.out or Path(entries[args.drug].file)

    labels = args.labels or fetch_schaefer_labels_img(rc.n_regions, data_dir=args.data_dir)
    values = parcellate_nifti(args.nifti, labels, rc.n_regions, args.zero_as_missing)
    write_observed_map(values, out, rc.n_regions)
    side = write_provenance(
        out, values, drug=args.drug, source=args.source, n_subjects=args.n_subjects,
        contrast=args.contrast, parcellation=parc.name, input_nifti=args.nifti,
        measure=entries[args.drug].measure, zero_as_missing=args.zero_as_missing,
        extra={"note": args.note} if args.note else None,
    )
    missing = int(sum(v != v for v in values))
    print(f"{args.drug}: {rc.n_regions} regions ({missing} without data) -> {out}")
    print(f"provenance -> {side}")


if __name__ == "__main__":
    main()
