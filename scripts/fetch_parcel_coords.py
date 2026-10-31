#!/usr/bin/env python
"""Write Schaefer parcel spherical centroids + hemisphere for spin-test generation.

    python scripts/fetch_parcel_coords.py                       # default parcellation
    python scripts/fetch_parcel_coords.py --parcellation schaefer400

Downloads the Schaefer-2018 7-network annotation on fsaverage5 (via netneurotools) and
computes each parcel's centroid on the spherical surface. Output is a CSV of `x,y,z,hemi`
(hemi 0 = left, 1 = right), one row per parcel in the same order as the receptor atlas
(left hemisphere first), ready for scripts/make_spins.py.

Only Schaefer parcellations are supported: spin tests rotate cortical parcels on the sphere,
so a parcellation with subcortex (cammoun033) has no valid spin null.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from smilesbrain.config import load_receptors

MEDIAL_WALL = "Background+FreeSurfer_Defined_Medial_Wall"


def schaefer_centroids(n_regions: int, data_dir: str | None = None) -> pd.DataFrame:
    try:
        from netneurotools import datasets, freesurfer
    except ImportError as exc:  # pragma: no cover - env dependent
        raise SystemExit(
            "netneurotools is required: pip install 'netneurotools>=0.2.5,<0.3' "
            "(0.3 removed the spin-test API this project uses)."
        ) from exc
    annots = datasets.fetch_schaefer2018("fsaverage5", data_dir=data_dir)
    key = f"{n_regions}Parcels7Networks"
    if key not in annots:
        raise SystemExit(f"No Schaefer annotation for {key}; available: {sorted(annots)}")
    lh, rh = annots[key]
    coords, hemi = freesurfer.find_parcel_centroids(
        lhannot=lh, rhannot=rh, version="fsaverage5", surf="sphere", drop=[MEDIAL_WALL]
    )
    if coords.shape[0] != n_regions:
        raise SystemExit(f"Expected {n_regions} parcels, annotation gave {coords.shape[0]}")
    df = pd.DataFrame(np.asarray(coords, dtype=float), columns=["x", "y", "z"])
    df["hemi"] = np.asarray(hemi, dtype=int)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parcellation", default=None,
                    help="key under `parcellations` in configs/receptors.yaml")
    ap.add_argument("--out", default=None, type=Path,
                    help="default: data/atlas/schaefer<N>_coords.csv")
    ap.add_argument("--data-dir", default=None, help="cache dir for downloaded annotations")
    args = ap.parse_args()

    parc = load_receptors(parcellation=args.parcellation).parcellation
    if parc.subcortex or not parc.name.startswith("Schaefer"):
        raise SystemExit(
            f"{parc.key}: spin tests need a cortical Schaefer parcellation; "
            "subcortical parcels cannot be rotated on the sphere."
        )
    out = args.out or Path(f"data/atlas/schaefer{parc.n_regions}_coords.csv")
    df = schaefer_centroids(parc.n_regions, args.data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"coords[{parc.key}]: {len(df)} parcels "
          f"({int((df.hemi == 0).sum())} L / {int((df.hemi == 1).sum())} R) -> {out}")
    print(f"next: python scripts/make_spins.py --coords {out} --out {parc.spins_file}")


if __name__ == "__main__":
    main()
