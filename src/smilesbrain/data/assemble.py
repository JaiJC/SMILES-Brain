"""Merge per-receptor affinity CSVs into one wide multi-task label table.

Result: a DataFrame indexed by canonical SMILES with one column per receptor (atlas_key),
holding pChEMBL where measured and NaN elsewhere. Multi-task training masks the NaNs, so a
molecule assayed against only a few receptors still contributes to those columns -- this
sparse-label regime is exactly what ChEMBL gives you and is the whole reason to train one
multi-task model rather than 19 independent ones.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from smilesbrain.config import load_receptors


def assemble(affinity_dir: Path, config: str | None = None) -> pd.DataFrame:
    cfg = load_receptors(config)
    merged: pd.DataFrame | None = None
    for rec in cfg.receptors:
        path = affinity_dir / f"{rec.atlas_key}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)[["canonical_smiles", "pchembl"]].rename(
            columns={"pchembl": rec.atlas_key}
        )
        df = df.dropna(subset=["canonical_smiles"]).set_index("canonical_smiles")
        merged = df if merged is None else merged.join(df, how="outer")
    if merged is None:
        raise FileNotFoundError(f"No per-receptor CSVs found in {affinity_dir}")
    # Enforce canonical column order so column i always means receptor i downstream.
    cols = [k for k in cfg.atlas_keys if k in merged.columns]
    return merged[cols]


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble wide affinity label table.")
    ap.add_argument("--affinity-dir", default="data/affinity", type=Path)
    ap.add_argument("--out", default="data/labels.parquet", type=Path)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    table = assemble(args.affinity_dir, args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".parquet":
        table.to_parquet(args.out)
    else:
        table.to_csv(args.out)
    density = table.notna().mean().mean()
    print(f"{len(table)} molecules x {table.shape[1]} receptors; "
          f"label density {density:.1%} -> {args.out}")


if __name__ == "__main__":
    main()
