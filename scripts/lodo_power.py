#!/usr/bin/env python
"""How strong must a real affinity->map signal be for the pre-registered test to see it?

Plants a known gain model in synthetic data (`--n-drugs` drugs, random 8-16 receptors each), adds noise of
increasing size to the maps, and reports how often the pre-registered criterion is met. Also
reports the false-positive rate when maps carry no affinity information. This is a property of
the *test*, not a result about any real drug: read it as "what a NOT MET verdict can and cannot
rule out".

    python scripts/lodo_power.py --seeds 40
"""

from __future__ import annotations

import argparse

import numpy as np

from smilesbrain.config import load_receptors
from smilesbrain.data.synthetic import synthetic_atlas
from smilesbrain.projection.gains import occupancy, predict_map, standardize
from smilesbrain.validation.lodo import lodo_evaluate, summarise_lodo


def one_study(rc, atlas, rng, noise: float | None, n_perm: int,
              n_drugs: int = 8) -> tuple[str, float]:
    profiles = {}
    for i in range(n_drugs):
        keys = rng.choice(rc.atlas_keys, size=int(rng.integers(8, 17)), replace=False)
        profiles[f"d{i}"] = {k: float(rng.uniform(5.5, 9.0)) for k in keys}
    gains = rng.standard_normal(rc.n_receptors)
    density = atlas.to_numpy()
    maps = {}
    for d, prof in profiles.items():
        if noise is None:   # null: a density combination unrelated to affinity
            maps[d] = standardize(density @ rng.standard_normal(rc.n_receptors))
        else:
            clean = standardize(predict_map(occupancy(prof, rc.atlas_keys, 100.0), atlas, gains))
            maps[d] = clean + noise * rng.standard_normal(clean.shape)
    rows, _ = lodo_evaluate(profiles, profiles, maps, atlas, n_perm=n_perm,
                            seed=int(rng.integers(1_000_000)))
    s = summarise_lodo(rows)
    return s["verdict"], s["median_r_model"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--n-perm", type=int, default=300)
    ap.add_argument("--n-drugs", type=int, default=8,
                    help="evaluable drugs per study (the real data had only 6 with >=3 receptors)")
    args = ap.parse_args()
    rc = load_receptors()
    atlas = synthetic_atlas(rc, seed=4)
    print("map noise sd (signal sd = 1) | criterion met | median held-out r")
    for noise in (None, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0):
        out = [one_study(rc, atlas, np.random.default_rng(1000 * i + 7), noise, args.n_perm,
                          args.n_drugs)
               for i in range(args.seeds)]
        met = np.mean([v == "CRITERION MET" for v, _ in out])
        label = "none (null: no affinity info)" if noise is None else f"{noise:.1f}"
        print(f"{label:>30} | {met:6.0%}        | {np.nanmedian([r for _, r in out]):+.2f}")


if __name__ == "__main__":
    main()
