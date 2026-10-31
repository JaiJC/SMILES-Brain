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


def centroids_from_annotations(lh_labels, lh_names, lh_sphere, rh_labels, rh_names, rh_sphere,
                               drop=(MEDIAL_WALL,)) -> pd.DataFrame:
    """Parcel centroids on the sphere, replicating netneurotools' default `method='surface'`.

    For each non-dropped parcel (in annotation label order, left hemisphere first): average the
    sphere coordinates of its vertices, then snap to the nearest vertex of the sphere. Needs no
    netneurotools and no download beyond the annotation and a sphere mesh.
    """
    from scipy.spatial.distance import cdist

    centroids, hemi = [], []
    for h, (labels, names, sphere) in enumerate(
            [(lh_labels, lh_names, lh_sphere), (rh_labels, rh_names, rh_sphere)]):
        labels = np.asarray(labels)
        sphere = np.asarray(sphere, dtype=float)
        if len(labels) != len(sphere):
            raise ValueError(f"annotation has {len(labels)} vertices, sphere has {len(sphere)}")
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
        for lab in np.unique(labels):
            if lab < 0 or names[lab] in drop:
                continue
            mean = sphere[labels == lab].mean(axis=0, keepdims=True)
            centroids.append(sphere[int(np.argmin(cdist(sphere, mean)[:, 0]))])
            hemi.append(h)
    df = pd.DataFrame(np.vstack(centroids), columns=["x", "y", "z"])
    df["hemi"] = np.asarray(hemi, dtype=int)
    return df


def offline_schaefer_centroids(n_regions: int, annot_dir: str, data_dir: str | None = None
                               ) -> pd.DataFrame:
    """Centroids from local `{lh,rh}.Schaefer2018_<N>Parcels_7Networks_order.annot` files and
    nilearn's bundled fsaverage5 sphere (for hosts where the OSF download is blocked)."""
    from pathlib import Path as _P

    import nibabel as nib
    from nilearn import datasets, surface

    fsavg = datasets.fetch_surf_fsaverage("fsaverage5", data_dir=data_dir)
    args = []
    for hemi in ("lh", "rh"):
        annot = _P(annot_dir) / f"{hemi}.Schaefer2018_{n_regions}Parcels_7Networks_order.annot"
        if not annot.exists():
            raise SystemExit(f"missing {annot}")
        labels, _ctab, names = nib.freesurfer.read_annot(str(annot))
        sphere = surface.load_surf_mesh(fsavg["sphere_left" if hemi == "lh" else "sphere_right"])
        args += [labels, names, np.asarray(sphere.coordinates)]
    df = centroids_from_annotations(*args)
    if len(df) != n_regions:
        raise SystemExit(f"Expected {n_regions} parcels, annotation gave {len(df)}")
    return df


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
    ap.add_argument("--annot-dir", default=None,
                    help="folder with local {lh,rh}.Schaefer2018_<N>Parcels_7Networks_order.annot "
                         "(CBIG repo, FreeSurfer5.3/fsaverage5/label); skips the OSF download")
    args = ap.parse_args()

    parc = load_receptors(parcellation=args.parcellation).parcellation
    if parc.subcortex or not parc.name.startswith("Schaefer"):
        raise SystemExit(
            f"{parc.key}: spin tests need a cortical Schaefer parcellation; "
            "subcortical parcels cannot be rotated on the sphere."
        )
    out = args.out or Path(f"data/atlas/schaefer{parc.n_regions}_coords.csv")
    df = (offline_schaefer_centroids(parc.n_regions, args.annot_dir, args.data_dir)
          if args.annot_dir else schaefer_centroids(parc.n_regions, args.data_dir))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"coords[{parc.key}]: {len(df)} parcels "
          f"({int((df.hemi == 0).sum())} L / {int((df.hemi == 1).sum())} R) -> {out}")
    print(f"next: python scripts/make_spins.py --coords {out} --out {parc.spins_file}")


if __name__ == "__main__":
    main()
