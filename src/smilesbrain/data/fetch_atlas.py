"""Download the Hansen et al. (2022) PET receptor density atlas (parcellated).

The atlas lives in the public repo `netneurolab/hansen_receptors`. Parcellated receptor
maps for the Schaefer-100 parcellation are distributed as plain-text vectors (one value per
region) under the repo's PET data folder. Because several receptors have more than one
tracer map, the repo ships multiple files per receptor; we average the z-scored maps for a
receptor, matching the atlas paper's construction.

We resolve filenames by listing the folder via the GitHub contents API and matching each
file's leading token against our `atlas_key`s. The result is written as a tidy
(region x receptor) matrix `data/atlas/density_scale100.csv`, columns in canonical order.

If filenames in the upstream repo change, adjust `KEY_ALIASES`. Run `--list` to print what
the fetcher found without downloading, which is the fastest way to debug a mapping drift.
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
# Folder holding Schaefer-100 parcellated PET vectors in the upstream repo.
PET_DIR = "data/PET_parcellated/scale100"

# Map upstream filename leading-token -> our atlas_key, where they differ.
KEY_ALIASES = {
    "5HT1a": "5HT1a", "5HT1b": "5HT1b", "5HT2a": "5HT2a", "5HT4": "5HT4", "5HT6": "5HT6",
    "5HTT": "5HTT", "A4B2": "A4B2", "CB1": "CB1", "D1": "D1", "D2": "D2", "DAT": "DAT",
    "GABAa": "GABAa", "H3": "H3", "M1": "M1", "mGluR5": "mGluR5", "MU": "MU",
    "NAT": "NAT", "NMDA": "NMDA", "VAChT": "VAChT",
}


def list_pet_files() -> list[str]:
    resp = requests.get(f"{GH_API}/{PET_DIR}", timeout=60)
    resp.raise_for_status()
    return [item["name"] for item in resp.json() if item["name"].endswith(".csv")]


def _key_from_filename(fname: str) -> str | None:
    token = fname.split("_")[0]
    return KEY_ALIASES.get(token)


def _download_vector(fname: str) -> np.ndarray:
    resp = requests.get(f"{RAW}/{PET_DIR}/{fname}", timeout=60)
    resp.raise_for_status()
    vec = pd.read_csv(io.StringIO(resp.text), header=None).values.ravel().astype(float)
    return vec


def build_atlas(config: str | None = None) -> pd.DataFrame:
    cfg = load_receptors(config)
    files = list_pet_files()

    by_key: dict[str, list[np.ndarray]] = {k: [] for k in cfg.atlas_keys}
    for fname in files:
        key = _key_from_filename(fname)
        if key in by_key:
            by_key[key].append(_download_vector(fname))

    columns: dict[str, np.ndarray] = {}
    for key in cfg.atlas_keys:
        vecs = by_key[key]
        if not vecs:
            continue
        # z-score each map, then average across tracers for the receptor.
        z = [(v - v.mean()) / (v.std() + 1e-8) for v in vecs]
        columns[key] = np.mean(z, axis=0)

    if not columns:
        raise RuntimeError(
            "No atlas maps matched. Run with --list and update KEY_ALIASES to match "
            "the upstream filenames."
        )
    df = pd.DataFrame(columns)
    df.index.name = "region"
    return df[[k for k in cfg.atlas_keys if k in df.columns]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Hansen PET receptor atlas.")
    ap.add_argument("--out", default="data/atlas/density_scale100.csv", type=Path)
    ap.add_argument("--config", default=None)
    ap.add_argument("--list", action="store_true", help="list upstream files and matched keys")
    args = ap.parse_args()

    if args.list:
        for f in list_pet_files():
            print(f"{f:50s} -> {_key_from_filename(f)}")
        return

    df = build_atlas(args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out)
    print(f"atlas: {df.shape[0]} regions x {df.shape[1]} receptors -> {args.out}")
    missing = [k for k in load_receptors(args.config).atlas_keys if k not in df.columns]
    if missing:
        print(f"WARNING: no atlas map for: {missing}")


if __name__ == "__main__":
    main()
