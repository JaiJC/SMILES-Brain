"""Download the Hansen et al. (2022) PET receptor density atlas (parcellated).

The atlas lives in the public repo `netneurolab/hansen_receptors`. Parcellated receptor maps
are distributed per parcellation as plain-text vectors (one value per region) under the repo's
PET data folder; which folder, output path and expected region count come from the
`parcellations` entry in configs/receptors.yaml (`--parcellation`, default schaefer100).

Construction follows the paper's own `code/make_receptor_matrix.py` exactly, because the
projection is only as good as the atlas it multiplies:

  * **One tracer is chosen per receptor**, not an average of everything on offer. The repo
    ships several tracers for many receptors, but the paper keeps the best-powered / best
    validated one and discards the rest -- e.g. D2 uses FLB457 (n=37 + n=55) and *not*
    fallypride or raclopride (n=7); GABAa uses the flumazenil BZ-site map from Norgaard and
    not Dukart's n=6 map; DAT uses the n=174 SPECT map and not FEPE2I (n=6).
  * Where a receptor has several maps of the **same** tracer, they are z-scored and averaged
    **weighted by the number of healthy controls**, so a small cohort cannot outvote a large one.

An earlier version of this file averaged every file whose leading token matched a receptor
key, with equal weight. That silently mixed discarded tracers back in and moved 10 of the 19
columns away from the published atlas (DAT r=0.88, NAT r=0.88, 5HT2a r=0.88, CB1 r=0.90,
D2 r=0.91, GABAa r=0.93 against the published matrix). `--verify` checks the rebuild against
the published matrix so that can never silently regress again.

Run `--list` to print the upstream files and which are selected, which is the fastest way to
debug a mapping drift.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from smilesbrain.config import load_receptors

GH_API = "https://api.github.com/repos/netneurolab/hansen_receptors/contents"
RAW = "https://raw.githubusercontent.com/netneurolab/hansen_receptors/main"
# The paper's own region x receptor matrices, used by --verify. Cortex only: for the Cammoun
# scales these have fewer rows than the parcellated vectors (scale033 -> 68, not 83).
PUBLISHED = RAW + "/results/receptor_data_{scale}.csv"
PUBLISHED_ORDER = ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2", "CB1", "D1",
                   "D2", "DAT", "GABAa", "H3", "M1", "mGluR5", "MU", "NAT", "NMDA", "VAChT"]

# atlas_key -> [(upstream file stem, n healthy controls), ...]. Transcribed from
# hansen_receptors/code/make_receptor_matrix.py. A receptor with one entry uses that map
# as-is; several entries are z-scored and averaged weighted by n.
TRACER_SELECTION: dict[str, list[tuple[str, int]]] = {
    "5HT1a":  [("5HT1a_way_hc36_savli", 36)],
    "5HT1b":  [("5HT1b_p943_hc22_savli", 22), ("5HT1b_p943_hc65_gallezot", 65)],
    "5HT2a":  [("5HT2a_cimbi_hc29_beliveau", 29)],
    "5HT4":   [("5HT4_sb20_hc59_beliveau", 59)],
    "5HT6":   [("5HT6_gsk_hc30_radhakrishnan", 30)],
    "5HTT":   [("5HTT_dasb_hc100_beliveau", 100)],
    "A4B2":   [("A4B2_flubatine_hc30_hillmer", 30)],
    "CB1":    [("CB1_omar_hc77_normandin", 77)],
    "D1":     [("D1_SCH23390_hc13_kaller", 13)],
    "D2":     [("D2_flb457_hc37_smith", 37), ("D2_flb457_hc55_sandiego", 55)],
    "DAT":    [("DAT_fpcit_hc174_dukart_spect", 174)],
    "GABAa":  [("GABAa-bz_flumazenil_hc16_norgaard", 16)],
    "H3":     [("H3_cban_hc8_gallezot", 8)],
    "M1":     [("M1_lsn_hc24_naganawa", 24)],
    "mGluR5": [("mGluR5_abp_hc22_rosaneto", 22), ("mGluR5_abp_hc28_dubois", 28),
               ("mGluR5_abp_hc73_smart", 73)],
    "MU":     [("MU_carfentanil_hc204_kantonen", 204)],
    "NAT":    [("NAT_MRB_hc77_ding", 77)],
    "NMDA":   [("NMDA_ge179_hc29_galovic", 29)],
    "VAChT":  [("VAChT_feobv_hc3_spreng", 3), ("VAChT_feobv_hc4_tuominen", 4),
               ("VAChT_feobv_hc5_bedard_sum", 5), ("VAChT_feobv_hc18_aghourian_sum", 18)],
}

# Upstream quirk, reproduced deliberately so this matches the published matrix: in
# make_receptor_matrix.py the VAChT average multiplies only the first two maps by their n
# while still dividing by the full weight sum (3+4+5+18), i.e. the Bedard and Aghourian maps
# enter with weight 1 instead of 5 and 18. Reproducing the reference beats silently
# "improving" it; set --fix-vacht to use the intended weights instead.
VACHT_UPSTREAM_WEIGHTS = [3, 4, 1, 1]


def list_pet_files(pet_dir: str) -> list[str]:
    resp = requests.get(f"{GH_API}/{pet_dir}", timeout=60)
    resp.raise_for_status()
    return [item["name"] for item in resp.json() if item["name"].endswith(".csv")]


def _selected_stems() -> dict[str, str]:
    """file stem -> atlas_key, for the maps the paper actually uses."""
    return {stem: key for key, maps in TRACER_SELECTION.items() for stem, _ in maps}


def _key_from_filename(fname: str) -> str | None:
    """Which receptor a selected upstream file belongs to (None if the paper discards it)."""
    return _selected_stems().get(fname.removesuffix(".csv"))


def _download_vector(fname: str, pet_dir: str) -> np.ndarray:
    resp = requests.get(f"{RAW}/{pet_dir}/{fname}", timeout=60)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text), header=None).values.ravel().astype(float)


def _zscore(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() + 1e-12)


def build_atlas(
    config: str | None = None, parcellation: str | None = None, fix_vacht: bool = False
) -> pd.DataFrame:
    """Region x receptor density matrix, z-scored per column, in canonical receptor order."""
    cfg = load_receptors(config, parcellation)
    pet_dir = cfg.parcellation.pet_dir

    columns: dict[str, np.ndarray] = {}
    for key in cfg.atlas_keys:
        maps = TRACER_SELECTION.get(key)
        if not maps:
            continue
        weights = [n for _, n in maps]
        if key == "VAChT" and not fix_vacht:
            weights = VACHT_UPSTREAM_WEIGHTS
        vectors = []
        for stem, _ in maps:
            vec = _download_vector(f"{stem}.csv", pet_dir)
            if vec.shape[0] != cfg.n_regions:
                raise RuntimeError(
                    f"{stem}.csv has {vec.shape[0]} regions but parcellation "
                    f"{cfg.parcellation.key!r} expects n_regions={cfg.n_regions}; "
                    "check pet_dir/n_regions in configs/receptors.yaml."
                )
            vectors.append(vec)
        if len(vectors) == 1:
            columns[key] = vectors[0]
        else:
            # z-score each map first so tracers on different scales can be averaged
            stacked = np.vstack([_zscore(v) * w for v, w in zip(vectors, weights)])
            columns[key] = stacked.sum(axis=0) / sum(n for _, n in maps)

    if not columns:
        raise RuntimeError(
            "No atlas maps matched. Run with --list and update TRACER_SELECTION to match "
            "the upstream filenames."
        )
    # Final z-score per column: the projection compares receptors to each other, so every
    # column must be on one scale (the paper z-scores the matrix before its analyses too).
    df = pd.DataFrame({k: _zscore(v) for k, v in columns.items()})
    df.index.name = "region"
    return df[[k for k in cfg.atlas_keys if k in df.columns]]


def fetch_published_matrix(scale: str) -> pd.DataFrame:
    """The paper's own region x receptor matrix, z-scored, with our column names."""
    resp = requests.get(PUBLISHED.format(scale=scale), timeout=60)
    resp.raise_for_status()
    mat = np.genfromtxt(io.StringIO(resp.text), delimiter=",")
    df = pd.DataFrame(mat, columns=PUBLISHED_ORDER)  # MOR/NET renamed to our MU/NAT
    return df.apply(_zscore)


def verify_against_published(df: pd.DataFrame, scale: str) -> pd.DataFrame:
    """Per-receptor correlation between a rebuilt atlas and the published matrix."""
    pub = fetch_published_matrix(scale)
    if len(pub) != len(df):
        raise RuntimeError(
            f"published matrix for {scale} has {len(pub)} regions but the rebuilt atlas has "
            f"{len(df)}. The published matrices are cortex-only, so the Cammoun scales differ "
            "(scale033 -> 68 vs 83). Verify on a Schaefer parcellation."
        )
    rows = [{"receptor": k, "r": float(np.corrcoef(df[k], pub[k])[0, 1])}
            for k in df.columns if k in pub.columns]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Hansen PET receptor atlas.")
    ap.add_argument("--out", default=None, type=Path,
                    help="default: the parcellation's atlas_file from the config")
    ap.add_argument("--config", default=None)
    ap.add_argument("--parcellation", default=None,
                    help="key under `parcellations` in the config (default: config default)")
    ap.add_argument("--list", action="store_true", help="list upstream files and matched keys")
    ap.add_argument("--verify", action="store_true",
                    help="compare the rebuild against the paper's published matrix")
    ap.add_argument("--fix-vacht", action="store_true",
                    help="use intended VAChT weights instead of reproducing the upstream quirk")
    args = ap.parse_args()

    cfg = load_receptors(args.config, args.parcellation)
    scale = cfg.parcellation.pet_dir.rsplit("/", 1)[-1]
    if args.list:
        selected = _selected_stems()
        for f in sorted(list_pet_files(cfg.parcellation.pet_dir)):
            key = selected.get(f.removesuffix(".csv"))
            print(f"{f:50s} -> {key if key else '(discarded by the paper)'}")
        return

    out = args.out or Path(cfg.parcellation.atlas_file)
    df = build_atlas(args.config, args.parcellation, fix_vacht=args.fix_vacht)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"atlas[{cfg.parcellation.key}]: {df.shape[0]} regions x {df.shape[1]} receptors -> {out}")
    missing = [k for k in cfg.atlas_keys if k not in df.columns]
    if missing:
        print(f"WARNING: no atlas map for: {missing}")

    if args.verify:
        report = verify_against_published(df, scale)
        print("\nagreement with the published receptor matrix:")
        print(report.to_string(index=False))
        worst = report["r"].min()
        print(f"\nworst receptor r = {worst:.4f}")
        if worst < 0.999:
            off = report.loc[report["r"] < 0.999, "receptor"].tolist()
            print(f"MISMATCH in: {off} -- the rebuild does not reproduce the published atlas.")
            raise SystemExit(1)
        print("rebuild reproduces the published atlas.")


if __name__ == "__main__":
    main()
