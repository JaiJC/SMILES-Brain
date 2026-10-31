#!/usr/bin/env python
"""Audit the assembled sparse affinity label table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from smilesbrain.config import load_receptors


def audit_labels(path: Path, config: str | None = None) -> tuple[pd.DataFrame, float]:
    """Return per-receptor coverage statistics and whole-table density."""
    table = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, index_col=0)
    cfg = load_receptors(config)
    rows = []
    for key in cfg.atlas_keys:
        values = pd.to_numeric(table[key], errors="coerce") if key in table else pd.Series(
            dtype=float
        )
        observed = values.dropna().to_numpy(dtype=float)
        rows.append(
            {
                "receptor": key,
                "n_molecules": int(observed.size),
                "min_pchembl": float(np.min(observed)) if observed.size else np.nan,
                "median_pchembl": float(np.median(observed)) if observed.size else np.nan,
                "max_pchembl": float(np.max(observed)) if observed.size else np.nan,
                "observed_pct": float(observed.size / len(table) * 100) if len(table) else 0.0,
            }
        )
    stats = pd.DataFrame(rows)
    density = float(table.reindex(columns=cfg.atlas_keys).notna().mean().mean())
    return stats, density


def write_report(stats: pd.DataFrame, density: float, output: Path, labels_path: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Affinity label audit",
        "",
        f"Input: `{labels_path}`",
        f"Overall observed label density: **{density:.1%}**",
        "",
        "| Receptor | Molecules | Min pChEMBL | Median pChEMBL | Max pChEMBL | Observed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in stats.itertuples(index=False):
        fmt = lambda value: "n/a" if pd.isna(value) else f"{value:.2f}"
        lines.append(
            f"| {row.receptor} | {row.n_molecules} | {fmt(row.min_pchembl)} | "
            f"{fmt(row.median_pchembl)} | {fmt(row.max_pchembl)} | {row.observed_pct:.1f}% |"
        )
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit an assembled affinity label table.")
    ap.add_argument("--labels", default="data/labels.parquet", type=Path)
    ap.add_argument("--out", default="artifacts/label_audit.md", type=Path)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    if not args.labels.exists():
        raise SystemExit(
            f"Missing {args.labels}; run `python -m smilesbrain.data.assemble` first."
        )
    stats, density = audit_labels(args.labels, args.config)
    print(stats.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"\noverall label density: {density:.1%}")
    write_report(stats, density, args.out, args.labels)
    print(f"audit report -> {args.out}")


if __name__ == "__main__":
    main()
