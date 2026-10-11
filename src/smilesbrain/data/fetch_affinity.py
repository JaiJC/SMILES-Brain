"""Download structure-activity (affinity) data from ChEMBL for each atlas receptor.

For each receptor's UniProt accession we:
  1. resolve the ChEMBL target id(s) for that accession,
  2. pull bioactivities (Ki / Kd / IC50) for small molecules against those targets,
  3. keep (canonical SMILES, pChEMBL) rows.

pChEMBL (= -log10 of the activity in molar) is ChEMBL's harmonised affinity scale, so Ki,
Kd and IC50 become comparable on one axis. We train in this unit.

Output: one CSV per receptor under <out_dir>/<atlas_key>.csv with columns
[canonical_smiles, pchembl, standard_type, chembl_id]. These are later merged into one wide
(molecule x receptor) label table by `assemble.py`.

Network note: the session routes HTTPS through a proxy; ChEMBL is a public API and should be
reachable. The fetcher is paginated, resumable (skips receptors whose CSV already exists),
and polite (small sleeps). Run it on a login node, not inside a GPU job.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from smilesbrain.config import load_receptors

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
ACTIVITY_TYPES = ("Ki", "Kd", "IC50")
SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json", "User-Agent": "smiles-brain/0.1"})


def _get(url: str, params: dict) -> dict:
    for attempt in range(5):
        try:
            resp = SESSION.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def target_ids_for_uniprot(uniprot: str) -> list[str]:
    data = _get(
        f"{CHEMBL}/target.json",
        {"target_components__accession": uniprot, "limit": 100},
    )
    return [t["target_chembl_id"] for t in data.get("targets", [])]


def activities_for_target(target_chembl_id: str) -> pd.DataFrame:
    rows: list[dict] = []
    params = {
        "target_chembl_id": target_chembl_id,
        "standard_type__in": ",".join(ACTIVITY_TYPES),
        "limit": 1000,
        "offset": 0,
    }
    while True:
        data = _get(f"{CHEMBL}/activity.json", params)
        acts = data.get("activities", [])
        for a in acts:
            pchembl = a.get("pchembl_value")
            smiles = a.get("canonical_smiles")
            if pchembl is None or not smiles:
                continue
            rows.append(
                {
                    "canonical_smiles": smiles,
                    "pchembl": float(pchembl),
                    "standard_type": a.get("standard_type"),
                    "chembl_id": a.get("molecule_chembl_id"),
                }
            )
        meta = data.get("page_meta", {})
        if not meta.get("next"):
            break
        params["offset"] += params["limit"]
        time.sleep(0.2)
    return pd.DataFrame(rows)


def fetch_receptor(uniprot: str) -> pd.DataFrame:
    frames = []
    for tid in target_ids_for_uniprot(uniprot):
        df = activities_for_target(tid)
        if not df.empty:
            frames.append(df)
        time.sleep(0.2)
    if not frames:
        return pd.DataFrame(columns=["canonical_smiles", "pchembl", "standard_type", "chembl_id"])
    out = pd.concat(frames, ignore_index=True)
    # Median-collapse duplicate measurements of the same molecule.
    out = (
        out.groupby("canonical_smiles", as_index=False)
        .agg(pchembl=("pchembl", "median"),
             standard_type=("standard_type", "first"),
             chembl_id=("chembl_id", "first"))
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch ChEMBL affinity data per receptor.")
    ap.add_argument("--out-dir", default="data/affinity", type=Path)
    ap.add_argument("--config", default=None, help="receptors.yaml (default: packaged)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_receptors(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for rec in tqdm(cfg.receptors, desc="receptors"):
        dest = args.out_dir / f"{rec.atlas_key}.csv"
        if dest.exists() and not args.overwrite:
            continue
        df = fetch_receptor(rec.uniprot)
        df.to_csv(dest, index=False)
        tqdm.write(f"{rec.atlas_key:8s} {rec.uniprot}: {len(df):6d} molecules -> {dest}")


if __name__ == "__main__":
    main()
