#!/usr/bin/env python
"""Precompute spin-test permutation indices for the atlas parcellation.

Requires parcel spherical centroid coordinates. Supply a CSV with columns [x, y, z, hemi]
(one row per parcel, hemi in {0,1} or {L,R}); these come from the parcellation's surface
annotation (e.g. Schaefer-100 on fsaverage5 via neuromaps/nilearn).

    python scripts/make_spins.py --coords data/atlas/schaefer100_coords.csv \
        --n-perm 1000 --out data/atlas/spins_scale100.npy

Run once; the evaluation harness reuses the cached file for all drugs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from smilesbrain.validation.nulls import gen_spins_from_coords, save_spins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", required=True, type=Path,
                    help="CSV with columns x,y,z,hemi (one row per parcel)")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/atlas/spins_scale100.npy", type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.coords)
    cols = {c.lower(): c for c in df.columns}
    coords = df[[cols["x"], cols["y"], cols["z"]]].to_numpy(dtype=float)
    hemi_raw = df[cols["hemi"]].astype(str).str.upper()
    hemiid = hemi_raw.map({"L": 0, "R": 1, "0": 0, "1": 1}).to_numpy()
    if np.isnan(hemiid.astype(float)).any():
        raise SystemExit("hemi column must be in {L,R,0,1}")

    spins = gen_spins_from_coords(coords, hemiid.astype(int), n_perm=args.n_perm, seed=args.seed)
    save_spins(spins, args.out)
    print(f"spins: {spins.shape[0]} regions x {spins.shape[1]} perms -> {args.out}")


if __name__ == "__main__":
    main()
